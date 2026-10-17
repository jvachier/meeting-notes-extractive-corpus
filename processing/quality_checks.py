"""Quality checks on the unified corpus (PLAN §7) → ``data/processed/report.md``.

    uv run python -m processing.quality_checks [--determinism]

Hard checks (exit code 1 on failure): schema, family leakage, test/train 8-gram overlap,
exact duplicate segments, and determinism when ``--determinism`` is given. Everything else
(distributions, near-duplicates, shortcut probes) is reported for a human to read.
"""

from __future__ import annotations

import argparse
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import pyarrow as pa

from generation.bank import SeedBank, load_bank
from generation.config import DEFAULT_CONFIG, GenerationConfig, load_config
from generation.output import PROCESSED_DIR, SENTENCE_SCHEMA, read_parquet
from generation.schema import SeedSegment
from processing.unify import UnifyError, validate_rows
from processing.validate_seeds import jaccard, normalise

_SLOT_RE = re.compile(r"\{[^{}]*\}")
_WORD_RE = re.compile(r"[a-z0-9']+")


@dataclass
class Section:
    """One report section."""

    title: str
    lines: list[str] = field(default_factory=list)
    passed: bool | None = None  # None = informational


@dataclass
class QualityReport:
    """All sections plus the overall verdict."""

    sections: list[Section] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no hard check failed."""
        return all(s.passed is not False for s in self.sections)

    def markdown(self) -> str:
        """Render the report."""
        verdict = "PASS" if self.ok else "FAIL"
        out = ["# Quality report: meeting-notes-extractive-v1", "", f"**Overall: {verdict}**", ""]
        for s in self.sections:
            mark = {True: " ✅", False: " ❌", None: ""}[s.passed]
            out += [f"## {s.title}{mark}", "", *s.lines, ""]
        return "\n".join(out)


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        cells = [f"{c:.3f}" if isinstance(c, float) else str(c) for c in row]
        out.append("| " + " | ".join(cells) + " |")
    return out


def template_ngrams(segment: SeedSegment, n: int) -> set[tuple[str, ...]]:
    """Word n-grams of a segment's template text with every slot removed."""
    grams: set[tuple[str, ...]] = set()
    for sentence in segment.sentences:
        words = normalise(_SLOT_RE.sub(" ", sentence.text)).split()
        grams.update(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    return grams


def check_schema(rows: Sequence[dict[str, Any]], table: pa.Table) -> Section:
    """§7.1: every row valid, column types exact."""
    section = Section("1. Schema")
    try:
        n = validate_rows(rows)
        section.lines.append(f"- {n} rows validated against the §2.2 row model.")
        section.passed = True
    except UnifyError as exc:
        section.lines.append(f"- {exc}")
        section.passed = False
    if table.schema != SENTENCE_SCHEMA:
        section.lines.append("- Column types differ from the §2.2 Arrow schema.")
        section.passed = False
    else:
        section.lines.append("- Column names and Arrow types match §2.2 exactly.")
    return section


def check_leakage(rows: Sequence[dict[str, Any]], bank: SeedBank, cfg: GenerationConfig) -> Section:
    """§7.2: no family in two splits; test/train template 8-gram overlap ≤ threshold."""
    section = Section("2. Leakage")
    splits_of: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        family = row["family_id"]
        if family and not family.startswith("fill-"):
            splits_of[family].add(row["split"])
    leaked = sorted(f for f, s in splits_of.items() if len(s) > 1)
    section.lines.append(
        f"- Families seen: {len(splits_of)}; in more than one split: {len(leaked)}."
    )
    if leaked:
        section.lines.append(f"  - e.g. {leaked[:10]}")

    by_id = {s.segment_id: s for s in bank.all_segments}
    used: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        seg = row["segment_id"]
        if seg and seg in by_id:
            used[row["split"]].add(seg)
    n = cfg.quality.ngram_n
    train = (
        set().union(*(template_ngrams(by_id[s], n) for s in used["train"]))
        if used["train"]
        else set()
    )
    test = (
        set().union(*(template_ngrams(by_id[s], n) for s in used["test"]))
        if used["test"]
        else set()
    )
    overlap = len(test & train) / len(test) if test else 0.0
    limit = cfg.quality.max_test_ngram_overlap
    section.lines.append(
        f"- Test {n}-grams (slots removed, fillers excluded): {len(test)}; also in train: "
        f"{len(test & train)} = **{overlap:.2%}** (limit {limit:.0%})."
    )
    ami_meetings = defaultdict(set)
    for row in rows:
        if row["source"] == "ami":
            ami_meetings[row["note_id"]].add(row["split"])
    ami_bad = [m for m, s in ami_meetings.items() if len(s) > 1]
    section.lines.append(f"- AMI meetings in more than one split: {len(ami_bad)}.")
    section.passed = not leaked and overlap <= limit and not ami_bad
    return section


def check_duplicates(bank: SeedBank, cfg: GenerationConfig) -> Section:
    """§7.3: no exact duplicate segments; near-duplicates across families are listed."""
    section = Section("3. Duplicates")
    segments = list(bank.all_segments)
    texts = {s.segment_id: normalise(" ".join(x.text for x in s.sentences)) for s in segments}
    exact = [k for k, v in Counter(texts.values()).items() if v > 1]
    section.lines.append(f"- Segments: {len(segments)}; exact duplicates: {len(exact)}.")
    n = cfg.quality.near_duplicate_ngram
    grams = {
        s.segment_id: {
            tuple(w[i : i + n])
            for w in [texts[s.segment_id].split()]
            for i in range(len(w) - n + 1)
        }
        for s in segments
    }
    family = {s.segment_id: s.family_id for s in segments}
    near = []
    for a, b in combinations(sorted(grams), 2):
        if family[a] != family[b]:
            sim = jaccard(grams[a], grams[b])
            if sim >= cfg.quality.near_duplicate_jaccard:
                near.append((a, b, sim))
    section.lines.append(
        f"- Near-duplicates across families (word-{n}-gram Jaccard ≥ "
        f"{cfg.quality.near_duplicate_jaccard}): {len(near)}."
    )
    section.lines += [f"  - {a} ~ {b} ({sim:.2f})" for a, b, sim in near[:30]]
    section.passed = not exact
    return section


def _rate(rows: Sequence[dict[str, Any]]) -> float:
    return sum(r["importance"] for r in rows) / len(rows) if rows else 0.0


def distributions(
    rows: Sequence[dict[str, Any]], notes: Sequence[dict[str, Any]], cfg: GenerationConfig
) -> list[Section]:
    """§7.4: label mix, lengths, reuse."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["source"], row["meeting_type"])].append(row)
    zones = ["context", "discussion", "decision", "action_item", "closing"]
    mix = Section("4a. Importance and zone mix by source and meeting type")
    mix.lines = _table(
        ["source", "meeting_type", "sentences", "importance=1", *zones],
        (
            (src, mt, len(g), _rate(g), *(sum(r["zone"] == z for r in g) / len(g) for z in zones))
            for (src, mt), g in sorted(groups.items())
        ),
    )
    by_split: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[(row["source"], row["split"])].append(row)
    split = Section("4b. By source and split")
    meetings = Counter((n["source"], n["split"]) for n in notes)
    split.lines = _table(
        ["source", "split", "meetings", "sentences", "importance=1"],
        ((s, sp, meetings[(s, sp)], len(g), _rate(g)) for (s, sp), g in sorted(by_split.items())),
    )
    lengths = Section("4c. Meeting length (words) vs §3.1 ranges")
    table_rows = []
    for mt, mt_cfg in cfg.meeting_types.items():
        for source in ("synthetic", "synthetic_hn"):
            words = [
                n["n_words"] for n in notes if n["meeting_type"] == mt and n["source"] == source
            ]
            if not words:
                continue
            lo, hi = mt_cfg.words
            inside = sum(lo <= w <= hi for w in words) / len(words)
            table_rows.append(
                (
                    mt.value,
                    source,
                    f"{lo}–{hi}",
                    min(words),
                    int(statistics.median(words)),
                    max(words),
                    inside,
                )
            )
    ami_words = [n["n_words"] for n in notes if n["source"] == "ami"]
    lengths.lines = _table(
        ["meeting_type", "source", "range", "min", "median", "max", "in range"], table_rows
    )
    if ami_words:
        lengths.lines.append(
            f"\nAMI meetings: min {min(ami_words)}, median {int(statistics.median(ami_words))}, "
            f"max {max(ami_words)} words (no target range)."
        )
    reuse = Section("4d. Segment reuse across the corpus")
    placements = {
        (r["note_id"], r["segment_id"])
        for r in rows
        if r["segment_id"] and not r["segment_id"].startswith("fill-")
    }
    per_segment = Counter(seg for _, seg in placements)
    counts = sorted(per_segment.values())
    if counts:
        bins = Counter(min(c // 5 * 5, 50) for c in counts)
        reuse.lines.append(
            f"- Segments used: {len(counts)}; placements: {sum(counts)}; per segment min "
            f"{counts[0]}, median {int(statistics.median(counts))}, max {counts[-1]} "
            f"(cap {cfg.composition.max_segment_reuse})."
        )
        reuse.lines += _table(
            ["uses", "segments"],
            ((f"{b}–{b + 4}" if b < 50 else "50+", n) for b, n in sorted(bins.items())),
        )
    return [mix, split, lengths, reuse]


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _rule(text: str, keywords: Sequence[str]) -> int:
    words = _words(text)
    for kw in keywords:
        if kw.startswith("'"):
            if any(w.endswith(kw) for w in words):
                return 1
        elif " " in kw:
            if kw in " ".join(words):
                return 1
        elif kw in words:
            return 1
    return 0


def _prf(
    rows: Sequence[dict[str, Any]], keywords: Sequence[str]
) -> tuple[float, float, float, int]:
    tp = fp = fn = 0
    for row in rows:
        pred = _rule(row["sentence_text"], keywords)
        gold = row["importance"]
        tp += pred and gold
        fp += pred and not gold
        fn += (not pred) and gold
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, len(rows)


def shortcut_probes(rows: Sequence[dict[str, Any]], cfg: GenerationConfig) -> Section:
    """§7.5: trigger-word importance rates and a keyword rule baseline."""
    section = Section("5. Shortcut probes (no ML)")
    q = cfg.quality
    train = [r for r in rows if r["source"] != "ami" and r["split"] == "train"]
    seen: Counter[str] = Counter()
    positive: Counter[str] = Counter()
    families: dict[str, set[str]] = defaultdict(set)
    for row in train:
        for w in set(_words(row["sentence_text"])):
            seen[w] += 1
            positive[w] += row["importance"]
            families[w].add(row["family_id"])
    base = _rate(train)
    ranked = sorted(
        (
            (positive[w] / seen[w], seen[w], w)
            for w in seen
            if seen[w] >= q.trigger_min_count and len(families[w]) >= q.trigger_min_families
        ),
        reverse=True,
    )[: q.top_trigger_words]
    section.lines.append(
        f"Top {q.top_trigger_words} words by P(importance=1 | word) in synthetic train "
        "(normal and hard-negative meetings) "
        f"(base rate {base:.3f}; words with ≥ {q.trigger_min_count} occurrences in ≥ "
        f"{q.trigger_min_families} seed families):"
    )
    section.lines.append("")
    section.lines.append(", ".join(f"`{w}` {p:.2f} ({n})" for p, n, w in ranked))
    section.lines.append("")
    kw = q.rule_keywords
    subsets = {
        "synthetic test": [r for r in rows if r["source"] == "synthetic" and r["split"] == "test"],
        "synthetic_hn test": [
            r for r in rows if r["source"] == "synthetic_hn" and r["split"] == "test"
        ],
        "AMI test": [r for r in rows if r["source"] == "ami" and r["split"] == "test"],
    }
    results = {name: _prf(subset, kw) for name, subset in subsets.items()}
    section.lines.append(f"Keyword rule baseline (importance=1 if any of {len(kw)} keywords):")
    section.lines.append("")
    section.lines += _table(
        ["subset", "precision", "recall", "F1", "sentences"], ((k, *v) for k, v in results.items())
    )
    hn_only = [
        r
        for r in rows
        if r["source"] == "synthetic_hn" and r["split"] == "test" and r["hard_negative"]
    ]
    fires = sum(_rule(r["sentence_text"], kw) for r in hn_only) / len(hn_only) if hn_only else 0.0
    section.lines.append("")
    section.lines.append(
        f"- On the {len(hn_only)} hard-negative sentences in test (all importance 0), the rule "
        f"fires on {fires:.1%}: every one of those is a false positive it cannot avoid."
    )
    gap = results["synthetic test"][2] - results["AMI test"][2]
    hn_p = results["synthetic_hn test"][0]
    syn_p = results["synthetic test"][0]
    section.lines.append("")
    section.lines.append(
        f"- F1 gap synthetic test − AMI test: {gap:+.3f} (warn above {q.max_rule_f1_gap})"
        + (" ⚠️ synthetic labels may be too easy for the rule." if gap > q.max_rule_f1_gap else ".")
    )
    section.lines.append(
        f"- Rule precision: synthetic test {syn_p:.3f} vs hard-negative meetings {hn_p:.3f}"
        + (
            " (hard negatives lower it, as intended)."
            if hn_p < syn_p
            else " ⚠️ hard negatives do not lower it."
        )
    )
    return section


def run_checks(
    sentences: pa.Table, notes: pa.Table, bank: SeedBank, cfg: GenerationConfig
) -> QualityReport:
    """Run §7.1–7.5 and return the report."""
    rows = sentences.to_pylist()
    note_rows = notes.to_pylist()
    report = QualityReport()
    report.sections.append(check_schema(rows, sentences))
    report.sections.append(check_leakage(rows, bank, cfg))
    report.sections.append(check_duplicates(bank, cfg))
    report.sections += distributions(rows, note_rows, cfg)
    report.sections.append(shortcut_probes(rows, cfg))
    return report


def determinism_section(hashes: Mapping[str, tuple[str, str]]) -> Section:
    """§7.6: one line per file with the hashes of two runs."""
    section = Section("6. Determinism (tiny config, two runs)")
    same = all(a == b for a, b in hashes.values())
    section.lines += [
        f"- `{name}`: {a[:16]} / {b[:16]} {'✓' if a == b else '✗'}"
        for name, (a, b) in hashes.items()
    ]
    section.passed = same
    return section


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sentences", type=Path, default=PROCESSED_DIR / "unified_corpus.parquet")
    parser.add_argument("--notes", type=Path, default=PROCESSED_DIR / "unified_notes.parquet")
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR / "report.md")
    parser.add_argument(
        "--determinism", action="store_true", help="also rebuild the tiny config twice"
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    report = run_checks(read_parquet(args.sentences), read_parquet(args.notes), load_bank(), cfg)
    if args.determinism:
        from processing.build import determinism_hashes  # local import: avoids a cycle

        report.sections.append(determinism_section(determinism_hashes()))
    args.out.write_text(report.markdown(), encoding="utf-8")
    print(f"wrote {args.out}: {'PASS' if report.ok else 'FAIL'}")
    for s in report.sections:
        if s.passed is not None:
            print(f"  {'PASS' if s.passed else 'FAIL'}  {s.title}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
