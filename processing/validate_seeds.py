"""Validate everything Claude Code writes: segments, hard negatives, fillers and lexicons.

Run after every seed batch (PLAN §3.4):

    uv run python -m processing.validate_seeds            # errors fail, coverage is reported
    uv run python -m processing.validate_seeds --strict-coverage   # also fail on shortfalls

Errors make the exit code 1. Warnings are printed but never fail the run. Thresholds come from
the ``seeds`` and ``lexicons`` sections of ``configs/generation.yaml``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from generation.config import (
    DEFAULT_CONFIG,
    DEFAULT_COVERAGE,
    REPO_ROOT,
    Coverage,
    GenerationConfig,
    load_config,
    load_coverage,
)
from generation.schema import (
    Domain,
    Filler,
    HNCategory,
    MeetingType,
    SeedSegment,
    Zone,
)
from generation.slots import ROLE_OWNER, SlotError, load_lexicons, parse_slots

DEFAULT_SEEDS = REPO_ROOT / "seeds"
DEFAULT_LEXICONS = REPO_ROOT / "generation" / "lexicons"

_SEGMENT_FILE_RE = re.compile(
    r"^(?P<meeting_type>[a-z_]+)__(?P<domain>[a-z_]+)__(?P<batch>b\d{2})$"
)
_HN_FILE_RE = re.compile(r"^(?P<category>[a-z_]+)__(?P<batch>b\d{2})$")
_END_PUNCT = (".", "?", "!", "…", '"', "'")
_INNER_BOUNDARY_RE = re.compile(r"[.!?]\s+[A-Z]")

#: Required form of each lexicon value (see generation/lexicons/README.md).
LEXICON_FORMS: dict[str, tuple[re.Pattern[str], str]] = {
    "PERSON": (re.compile(r"^[^\W\d_][^\W\d_]*$"), "one capitalised word"),
    "TEAM": (re.compile(r"^the [A-Za-z]"), "a noun phrase starting with 'the '"),
    "TOOL": (re.compile(r"^the [A-Za-z]"), "a noun phrase starting with 'the '"),
    "PROJECT": (re.compile(r"^[A-Z]"), "a capitalised name without article"),
    "PRODUCT": (re.compile(r"^[A-Z]"), "a capitalised name without article"),
    "CLIENT": (re.compile(r"^[A-Z]"), "a capitalised name without article"),
    "METRIC": (re.compile(r"^(?!(?:the|a|an) )[a-z0-9]"), "lowercase, without article"),
    "PERCENT": (re.compile(r"percent$"), "a spoken percentage ending in 'percent'"),
    "MONEY": (re.compile(r"[$€£\d]|dollars|euros|pounds"), "an amount with a currency"),
}
_SLOT_STRIP_RE = re.compile(r"\{[^{}]*\}")
_WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")


@dataclass
class Report:
    """Outcome of a validation run."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when there are no errors."""
        return not self.errors


@dataclass(frozen=True)
class LoadedSegment:
    """A validated segment and where it came from."""

    segment: SeedSegment
    path: Path
    line: int


def _iter_jsonl(path: Path, report: Report) -> Iterator[tuple[int, dict[str, Any]]]:
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            report.errors.append(f"{path}:{number}: invalid JSON: {exc.msg}")
            continue
        if not isinstance(obj, dict):
            report.errors.append(f"{path}:{number}: a line must be a JSON object")
            continue
        yield number, obj


def _format_validation_error(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or '<segment>'}: {err['msg']}"
        for err in exc.errors()
    )


def normalise(text: str) -> str:
    """Lowercase, drop punctuation and collapse whitespace (for duplicate checks)."""
    return " ".join(re.sub(r"[^\w\s{}|]", " ", text.lower()).split())


def word_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    """Set of word n-grams of normalised ``text``."""
    words = normalise(text).split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set[Any], b: set[Any]) -> float:
    """Jaccard similarity; 0 when both sets are empty."""
    return len(a & b) / len(a | b) if a or b else 0.0


def _segment_text(segment: SeedSegment) -> str:
    return " ".join(s.text for s in segment.sentences)


def _load_segment_file(path: Path, report: Report, *, hard_negative: bool) -> list[LoadedSegment]:
    """Validate one JSONL file against the schema and its path naming."""
    stem_re = _HN_FILE_RE if hard_negative else _SEGMENT_FILE_RE
    name = stem_re.match(path.stem)
    if name is None:
        expected = "<category>__bNN" if hard_negative else "<meeting_type>__<domain>__bNN"
        report.errors.append(f"{path}: file name must be {expected}.jsonl")
        return []
    loaded: list[LoadedSegment] = []
    for number, obj in _iter_jsonl(path, report):
        where = f"{path}:{number}"
        try:
            segment = SeedSegment.model_validate(obj)
        except ValidationError as exc:
            report.errors.append(f"{where}: {_format_validation_error(exc)}")
            continue
        if hard_negative:
            if not segment.hard_negative or segment.hn_category != name.group("category"):
                report.errors.append(
                    f"{where}: hard-negative file {path.name} needs hn_category "
                    f"{name.group('category')!r}"
                )
                continue
        else:
            if segment.hard_negative:
                report.errors.append(f"{where}: hard negatives belong in seeds/hard_negatives/")
                continue
            if (
                segment.primary_zone != path.parent.name
                or segment.domain != name.group("domain")
                or segment.cell_meeting_type != name.group("meeting_type")
            ):
                report.errors.append(
                    f"{where}: segment zone/meeting type/domain do not match its path"
                )
                continue
        loaded.append(LoadedSegment(segment, path, number))
    return loaded


def _check_sentences(item: LoadedSegment, cfg: GenerationConfig, report: Report) -> None:
    """Per-segment checks that depend on configurable thresholds or labelling rules."""
    seg = item.segment
    where = f"{item.path}:{item.line} ({seg.segment_id})"
    lo, hi = cfg.seeds.segment_sentences
    if not lo <= len(seg.sentences) <= hi:
        report.errors.append(f"{where}: {len(seg.sentences)} sentences, expected {lo}–{hi}")

    conf_lo, conf_hi = (
        cfg.seeds.hard_negative_confidence if seg.hard_negative else cfg.seeds.segment_confidence
    )
    if not conf_lo <= seg.confidence <= conf_hi:
        report.errors.append(f"{where}: confidence {seg.confidence} outside [{conf_lo}, {conf_hi}]")

    labels = Counter((s.zone, s.importance) for s in seg.sentences)
    if not seg.hard_negative:
        if seg.primary_zone == Zone.DECISION and not labels[(Zone.DECISION, 1)]:
            report.errors.append(f"{where}: decision segment has no important decision sentence")
        if seg.primary_zone == Zone.ACTION_ITEM and not labels[(Zone.ACTION_ITEM, 1)]:
            report.errors.append(
                f"{where}: action_item segment has no important action_item sentence"
            )
    else:
        important = sum(n for (_, imp), n in labels.items() if imp == 1)
        if seg.hn_category == HNCategory.REVERSED_DECISION:
            if labels[(Zone.DECISION, 1)] != 1:
                report.warnings.append(
                    f"{where}: reversed_decision should have exactly one important decision "
                    "(the final one)"
                )
        elif important:
            report.warnings.append(
                f"{where}: hard negative {seg.hn_category} has {important} importance-1 sentences"
            )

    for i, sentence in enumerate(seg.sentences):
        text = sentence.text
        if not text.rstrip().endswith(_END_PUNCT):
            report.warnings.append(f"{where} s{i}: no final punctuation: {text!r}")
        if _INNER_BOUNDARY_RE.search(text):
            report.warnings.append(f"{where} s{i}: looks like two sentences in one entry")
        if any(r.name == sentence.speaker for r in parse_slots(text)):
            report.warnings.append(f"{where} s{i}: {sentence.speaker} refers to themselves by name")
        if sentence.zone == Zone.ACTION_ITEM and sentence.importance == 1:
            refs = parse_slots(text)
            if not any(r.role == ROLE_OWNER for r in refs):
                report.warnings.append(f"{where} s{i}: action item has no {{…|owner}} slot")
            has_owner = any(r.role == ROLE_OWNER for r in refs)
            # A mentioned participant who is not the owner is fine; a bare date is not.
            unroled = [
                r.name
                for r in refs
                if r.role is None and ("DATE" in r.name or (r.is_speaker and not has_owner))
            ]
            if unroled:
                report.warnings.append(
                    f"{where} s{i}: action item has speaker/date slots without a role: {unroled}"
                )


def _check_families(
    items: list[LoadedSegment], variants_expected: int, cfg: GenerationConfig, report: Report
) -> dict[str, list[SeedSegment]]:
    """Group segments by family and check variant consistency."""
    families: dict[str, list[SeedSegment]] = defaultdict(list)
    for item in items:
        families[item.segment.family_id].append(item.segment)

    for family_id, members in sorted(families.items()):
        members.sort(key=lambda s: s.variant)
        variants = [s.variant for s in members]
        if variants != list(range(1, variants_expected + 1)):
            report.errors.append(
                f"{family_id}: variants {variants}, expected v1..v{variants_expected}"
            )
        first = members[0]
        for other in members[1:]:
            if (other.primary_zone, other.domain, other.hard_negative, other.hn_category) != (
                first.primary_zone,
                first.domain,
                first.hard_negative,
                first.hn_category,
            ):
                report.errors.append(
                    f"{family_id}: variants disagree on zone/domain/hard-negative fields"
                )
            if set(other.meeting_types) != set(first.meeting_types):
                report.warnings.append(f"{family_id}: variants list different meeting_types")
            important = {
                s.segment_id: Counter(x.zone for x in s.sentences if x.importance == 1)
                for s in (first, other)
            }
            if important[first.segment_id] != important[other.segment_id]:
                report.warnings.append(
                    f"{family_id}: variants have different important-sentence labels"
                )
        for a, b in combinations(members, 2):
            if normalise(_segment_text(a)) == normalise(_segment_text(b)):
                report.errors.append(
                    f"{family_id}: {a.segment_id} and {b.segment_id} are identical"
                )
                continue
            sim = jaccard(word_ngrams(_segment_text(a), 3), word_ngrams(_segment_text(b), 3))
            if sim > cfg.seeds.variant_similarity_max:
                report.warnings.append(
                    f"{family_id}: {a.segment_id}/{b.segment_id} too similar (3-gram Jaccard "
                    f"{sim:.2f}); variants should be real paraphrases"
                )
    return families


def _check_cross_family(items: list[LoadedSegment], report: Report) -> None:
    """Unique segment ids and no exact duplicate segment text across the bank."""
    ids = Counter(item.segment.segment_id for item in items)
    for segment_id, count in ids.items():
        if count > 1:
            report.errors.append(f"duplicate segment_id {segment_id} ({count}×)")
    seen: dict[str, str] = {}
    for item in items:
        key = normalise(_segment_text(item.segment))
        if key in seen and seen[key] != item.segment.segment_id:
            report.errors.append(f"{item.segment.segment_id} duplicates the text of {seen[key]}")
        seen.setdefault(key, item.segment.segment_id)


def _coverage(
    families: dict[str, list[SeedSegment]],
    coverage: Coverage,
    report: Report,
    *,
    strict: bool,
) -> None:
    """Compare normal-segment families with the per-cell targets."""
    cells: Counter[tuple[Zone, MeetingType, Domain]] = Counter()
    for members in families.values():
        seg = members[0]
        meeting_type = seg.cell_meeting_type
        if meeting_type is not None:
            cells[(seg.primary_zone, meeting_type, seg.domain)] += 1

    shortfalls: list[str] = []
    zone_stats: dict[str, dict[str, int]] = {}
    for zone in Zone:
        done = sum(n for (z, _, _), n in cells.items() if z == zone)
        zone_stats[zone.value] = {"families": done, "target": coverage.zone_total(zone)}
        for meeting_type in MeetingType:
            for domain in Domain:
                target = coverage.target(zone, meeting_type, domain)
                got = cells[(zone, meeting_type, domain)]
                if got < target:
                    shortfalls.append(
                        f"{zone.value}/{meeting_type.value}/{domain.value}: {got}/{target}"
                    )
                elif got > target:
                    report.warnings.append(
                        f"coverage: {zone.value}/{meeting_type.value}/{domain.value} has "
                        f"{got} families, target {target}"
                    )
    report.stats["families_by_zone"] = zone_stats
    report.stats["cells_short"] = len(shortfalls)

    if not strict:
        return
    report.errors.extend(f"coverage shortfall {s}" for s in shortfalls)
    for zone in Zone:
        for domain in Domain:
            n = sum(v for (z, _, d), v in cells.items() if z == zone and d == domain)
            if n < coverage.min_families_per_zone_domain:
                report.errors.append(f"coverage: {zone.value}/{domain.value} has {n} families")
        for meeting_type in MeetingType:
            n = sum(v for (z, t, _), v in cells.items() if z == zone and t == meeting_type)
            if n < coverage.min_families_per_zone_meeting_type:
                report.errors.append(
                    f"coverage: {zone.value}/{meeting_type.value} has {n} families"
                )


def _hn_coverage(
    families: dict[str, list[SeedSegment]], coverage: Coverage, report: Report, *, strict: bool
) -> None:
    counts = Counter(members[0].hn_category for members in families.values())
    stats = {}
    for category, target in coverage.hard_negatives.categories.items():
        stats[category.value] = {"families": counts[category], "target": target}
        if strict and counts[category] < target:
            report.errors.append(
                f"coverage shortfall hard_negatives/{category.value}: {counts[category]}/{target}"
            )
    report.stats["hard_negative_families"] = stats


def _discussion_rate(items: list[LoadedSegment], cfg: GenerationConfig, report: Report) -> None:
    sentences = [
        s
        for item in items
        if item.segment.primary_zone == Zone.DISCUSSION and not item.segment.hard_negative
        for s in item.segment.sentences
        if s.zone == Zone.DISCUSSION
    ]
    n_segments = sum(
        1
        for i in items
        if i.segment.primary_zone == Zone.DISCUSSION and not i.segment.hard_negative
    )
    if not sentences:
        return
    rate = sum(s.importance for s in sentences) / len(sentences)
    report.stats["discussion_importance_rate"] = round(rate, 3)
    target, tol = cfg.seeds.discussion_importance_target, cfg.seeds.discussion_importance_tolerance
    if n_segments >= cfg.seeds.discussion_rate_min_segments and abs(rate - target) > tol:
        report.warnings.append(
            f"discussion importance-1 rate {rate:.2f} is outside {target:.2f} ± {tol:.2f}"
        )


def check_fillers(path: Path, cfg: GenerationConfig, report: Report) -> list[Filler]:
    """Validate ``fillers.jsonl``: schema, unique ids and texts."""
    if not path.exists():
        report.warnings.append(f"{path}: no fillers yet")
        return []
    fillers: list[Filler] = []
    for number, obj in _iter_jsonl(path, report):
        try:
            fillers.append(Filler.model_validate(obj))
        except ValidationError as exc:
            report.errors.append(f"{path}:{number}: {_format_validation_error(exc)}")
    for value, count in Counter(f.filler_id for f in fillers).items():
        if count > 1:
            report.errors.append(f"{path}: duplicate filler_id {value}")
    for value, count in Counter(normalise(f.text) for f in fillers).items():
        if count > 1:
            report.errors.append(f"{path}: duplicate filler text {value!r}")
    if fillers and len(fillers) < cfg.seeds.fillers_target:
        report.warnings.append(
            f"{path}: {len(fillers)} fillers, target about {cfg.seeds.fillers_target}"
        )
    tags = Counter(f.tag.value if f.tag else "any" for f in fillers)
    report.stats["fillers"] = {"total": len(fillers), "target": cfg.seeds.fillers_target, **tags}
    return fillers


def check_lexicons(
    directory: Path,
    cfg: GenerationConfig,
    used_slots: set[str],
    report: Report,
    seed_vocabulary: frozenset[str] = frozenset(),
) -> None:
    """Check lexicon sizes, forms and duplicates, and that every used slot has a lexicon.

    ``seed_vocabulary`` holds the lowercased words of all seed and filler texts: a PERSON value
    that is also an ordinary word there ("will", "may") would be ambiguous after ASR lowercasing.
    """
    try:
        lexicons = load_lexicons(directory) if directory.exists() else {}
    except SlotError as exc:
        report.errors.append(str(exc))
        return
    if not lexicons:
        report.warnings.append(f"{directory}: no lexicons yet")
    for slot in sorted(used_slots - set(lexicons)):
        report.errors.append(f"slot {{{slot}}} is used in seeds but has no lexicon file")
    for slot, values in sorted(lexicons.items()):
        minimum = cfg.lexicons.min_values[slot]
        if len(values) < minimum:
            report.errors.append(f"lexicon {slot}: {len(values)} values, need ≥ {minimum}")
        dupes = [v for v, n in Counter(values).items() if n > 1]
        if dupes:
            report.errors.append(f"lexicon {slot}: duplicate values {dupes[:5]}")
        if slot in LEXICON_FORMS:
            pattern, form = LEXICON_FORMS[slot]
            if slot == "PERSON":
                bad = [v for v in values if not (pattern.match(v) and v[0].isupper())]
            else:
                bad = [v for v in values if not pattern.search(v)]
            if bad:
                report.errors.append(f"lexicon {slot}: values must be {form}: {bad[:5]}")
    owners: dict[str, list[str]] = defaultdict(list)
    for slot, values in lexicons.items():
        for value in set(values):
            owners[value.lower()].append(slot)
    for value, slots in sorted(owners.items()):
        if len(slots) > 1:
            report.errors.append(f"lexicon value {value!r} appears in {sorted(slots)}")
    ambiguous = sorted(v for v in lexicons.get("PERSON", ()) if v.lower() in seed_vocabulary)
    if ambiguous:
        report.warnings.append(f"PERSON values that are also words in seed text: {ambiguous}")
    report.stats["lexicons"] = {slot: len(values) for slot, values in sorted(lexicons.items())}


def validate(
    seeds_dir: Path = DEFAULT_SEEDS,
    lexicons_dir: Path = DEFAULT_LEXICONS,
    cfg: GenerationConfig | None = None,
    coverage: Coverage | None = None,
    *,
    strict_coverage: bool = False,
) -> Report:
    """Run every seed check and return the report."""
    cfg = cfg or load_config(DEFAULT_CONFIG)
    coverage = coverage or load_coverage(DEFAULT_COVERAGE)
    report = Report()

    segments: list[LoadedSegment] = []
    segments_dir = seeds_dir / "segments"
    for path in sorted(segments_dir.glob("*/*.jsonl")):
        if path.parent.name not in set(Zone):
            report.errors.append(f"{path}: folder must be one of {[z.value for z in Zone]}")
            continue
        segments.extend(_load_segment_file(path, report, hard_negative=False))
    hard_negatives: list[LoadedSegment] = []
    for path in sorted((seeds_dir / "hard_negatives").glob("*.jsonl")):
        hard_negatives.extend(_load_segment_file(path, report, hard_negative=True))

    every = segments + hard_negatives
    for item in every:
        _check_sentences(item, cfg, report)
    _check_cross_family(every, report)
    families = _check_families(segments, coverage.variants_per_family, cfg, report)
    hn_families = _check_families(
        hard_negatives, coverage.hard_negatives.variants_per_family, cfg, report
    )
    overlap = set(families) & set(hn_families)
    report.errors.extend(f"family {f} is both normal and hard negative" for f in sorted(overlap))

    _coverage(families, coverage, report, strict=strict_coverage)
    _hn_coverage(hn_families, coverage, report, strict=strict_coverage)
    _discussion_rate(segments, cfg, report)
    report.stats["segments"] = len(segments)
    report.stats["hard_negative_segments"] = len(hard_negatives)

    fillers = check_fillers(seeds_dir / "fillers" / "fillers.jsonl", cfg, report)
    texts = [s.text for item in every for s in item.segment.sentences] + [f.text for f in fillers]
    vocabulary = frozenset(
        w.lower() for t in texts for w in _WORD_RE.findall(_SLOT_STRIP_RE.sub(" ", t))
    )
    used = {
        ref.base
        for item in every
        for sentence in item.segment.sentences
        for ref in parse_slots(sentence.text)
        if not ref.is_speaker
    }
    check_lexicons(lexicons_dir, cfg, used, report, vocabulary)
    return report


def _print_report(report: Report, max_warnings: int) -> None:
    for key, value in report.stats.items():
        if isinstance(value, dict) and all(
            isinstance(v, dict) and set(v) == {"families", "target"} for v in value.values()
        ):
            value = ", ".join(f"{k} {v['families']}/{v['target']}" for k, v in value.items())
        print(f"{key}: {value}")
    if report.warnings:
        print(f"\n{len(report.warnings)} warning(s):")
        for warning in report.warnings[:max_warnings]:
            print(f"  WARN  {warning}")
        if len(report.warnings) > max_warnings:
            print(f"  … {len(report.warnings) - max_warnings} more (use --max-warnings)")
    if report.errors:
        print(f"\n{len(report.errors)} error(s):")
        for error in report.errors:
            print(f"  ERROR {error}")
    print("\nseed validation:", "OK" if report.ok else "FAILED")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds-dir", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--lexicons-dir", type=Path, default=DEFAULT_LEXICONS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument(
        "--strict-coverage",
        action="store_true",
        help="fail when a coverage target is not met (use at the end of S3/S4)",
    )
    parser.add_argument("--max-warnings", type=int, default=50)
    args = parser.parse_args(argv)
    report = validate(
        args.seeds_dir,
        args.lexicons_dir,
        load_config(args.config),
        load_coverage(args.coverage),
        strict_coverage=args.strict_coverage,
    )
    _print_report(report, args.max_warnings)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
