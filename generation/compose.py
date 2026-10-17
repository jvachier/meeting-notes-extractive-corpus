"""Compose full meetings from seed segments (PLAN §4.1–4.2).

A meeting is ``context{a,b} body closing{1}``. Body zones are drawn with the meeting type's
weights until the meeting reaches its word range. Labels are copied from the seeds; this
module only chooses, orders, maps speakers, fills slots and renders text.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from generation.config import GenerationConfig
from generation.noise import render_asr
from generation.schema import (
    Domain,
    Filler,
    FillerTag,
    MeetingType,
    SeedSegment,
    Source,
    Split,
    Zone,
)
from generation.slots import SlotFiller

BODY_ZONE_ORDER: tuple[Zone, ...] = (Zone.DISCUSSION, Zone.DECISION, Zone.ACTION_ITEM)


class CompositionError(RuntimeError):
    """A meeting cannot be composed (empty pool or word range unreachable)."""


@dataclass
class SegmentPool:
    """Segments of one split, indexed by zone, with corpus-wide usage counts.

    ``uses`` is shared between pools and generators so ``max_segment_reuse`` holds across
    the whole corpus.
    """

    segments: Sequence[SeedSegment]
    hard_negatives: Sequence[SeedSegment]
    uses: Counter[str] = field(default_factory=Counter)

    def __post_init__(self) -> None:
        self.by_zone: dict[Zone, list[SeedSegment]] = {zone: [] for zone in Zone}
        for segment in self.segments:
            self.by_zone[segment.primary_zone].append(segment)


@dataclass(frozen=True)
class MeetingSpec:
    """What to compose."""

    note_id: str
    meeting_type: MeetingType
    domain: Domain
    split: Split
    source: Source
    n_hard_negatives: int = 0


def _capitalise_first(text: str) -> str:
    if not text:
        return text
    head = text[0].upper()
    return head + text[1:] if len(head) == 1 else text


def _choose(
    candidates: Sequence[SeedSegment],
    spec: MeetingSpec,
    *,
    n_participants: int,
    used_families: set[str],
    pool: SegmentPool,
    cfg: GenerationConfig,
    rng: random.Random,
) -> SeedSegment:
    """Pick one segment, weighting meeting-type and domain matches and penalising reuse."""
    cap = cfg.composition.max_segment_reuse
    usable = [
        s
        for s in candidates
        if s.n_speakers <= n_participants
        and s.family_id not in used_families
        and pool.uses[s.segment_id] < cap
    ]
    if not usable:
        raise CompositionError(f"{spec.note_id}: no usable segment left")
    comp = cfg.composition
    cross = rng.random() < comp.p_cross_domain

    def weight(s: SeedSegment) -> float:
        w = comp.type_weight if spec.meeting_type in s.meeting_types else 1.0
        if (s.domain == spec.domain) != cross:  # same domain, or another one on a cross draw
            w *= comp.domain_weight
        return w * (1 - pool.uses[s.segment_id] / cap) ** comp.reuse_penalty_power

    return rng.choices(usable, weights=[weight(s) for s in usable], k=1)[0]


def _plan(
    spec: MeetingSpec,
    pool: SegmentPool,
    n_participants: int,
    cfg: GenerationConfig,
    rng: random.Random,
) -> list[SeedSegment]:
    """Choose context, body and closing segments for one attempt."""
    mt_cfg = cfg.meeting_types[spec.meeting_type]
    used: set[str] = set()

    def take(candidates: Sequence[SeedSegment]) -> SeedSegment:
        segment = _choose(
            candidates,
            spec,
            n_participants=n_participants,
            used_families=used,
            pool=pool,
            cfg=cfg,
            rng=rng,
        )
        used.add(segment.family_id)
        return segment

    context = [
        take(pool.by_zone[Zone.CONTEXT]) for _ in range(rng.randint(*mt_cfg.context_segments))
    ]
    closing = [
        take(pool.by_zone[Zone.CLOSING])
        for _ in range(rng.randint(*cfg.composition.closing_segments))
    ]
    body = [take(pool.hard_negatives) for _ in range(spec.n_hard_negatives)]

    def words(segments: Sequence[SeedSegment]) -> int:
        return sum(len(s.text.split()) for seg in segments for s in seg.sentences)

    target = rng.randint(*mt_cfg.words)
    zones = list(BODY_ZONE_ORDER)
    weights = [mt_cfg.zone_weights[z] for z in zones]
    while not body or words(context + body + closing) < target:
        zone = rng.choices(zones, weights=weights, k=1)[0]
        # If the drawn zone has nothing usable left, fall back to the other body zones.
        fallback = [zone, *sorted((z for z in zones if z != zone), key=zones.index)]
        for candidate_zone in fallback:
            try:
                body.append(take(pool.by_zone[candidate_zone]))
                break
            except CompositionError:
                continue
        else:
            raise CompositionError(f"{spec.note_id}: every body zone is exhausted")
    rng.shuffle(body)
    return context + body + closing


def _speaker_map(
    segment: SeedSegment, participants: Sequence[str], *, chair_first: bool, rng: random.Random
) -> dict[str, str]:
    """Map S1..Sn to distinct participants; the chair is S1 in context and closing."""
    if chair_first:
        others = rng.sample(list(participants[1:]), segment.n_speakers - 1)
        names = [participants[0], *others]
    else:
        names = rng.sample(list(participants), segment.n_speakers)
    return {f"S{i}": name for i, name in enumerate(names, start=1)}


def _filler_speaker(participants: Sequence[str], avoid: str | None, rng: random.Random) -> str:
    choices = [p for p in participants if p != avoid] or list(participants)
    return rng.choice(choices)


def _render(
    spec: MeetingSpec,
    plan: Sequence[SeedSegment],
    participants: Sequence[str],
    fillers: Mapping[FillerTag | None, Sequence[Filler]],
    lexicons: Mapping[str, Sequence[str]],
    cfg: GenerationConfig,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Turn a plan into output rows (without ``sentence_id``)."""
    comp = cfg.composition
    slots = SlotFiller(
        lexicons, rng, reserved=participants, segment_scoped=comp.segment_scoped_slots
    )
    rows: list[dict[str, Any]] = []

    def base(speaker: str) -> dict[str, Any]:
        return {
            "note_id": spec.note_id,
            "source": spec.source.value,
            "meeting_type": spec.meeting_type.value,
            "domain": spec.domain.value,
            "split": spec.split.value,
            "speaker": speaker,
        }

    def add_filler(tag: FillerTag | None, avoid: str | None) -> None:
        options = fillers.get(tag) or ()
        if not options:
            return
        filler = rng.choice(list(options))
        text = filler.text
        rows.append(
            base(_filler_speaker(participants, avoid, rng))
            | {
                "sentence_text": text,
                "sentence_text_asr": render_asr(text, rng, cfg.noise),
                "importance": 0,
                "confidence": 1.0,
                "zone": Zone.DISCUSSION.value,
                "segment_id": filler.filler_id,
                "family_id": filler.filler_id,
                "hard_negative": False,
                "token_count": len(text.split()),
                "spans": [],
            }
        )

    if rng.random() < comp.p_opening_filler:
        add_filler(FillerTag.OPENING, None)
    for index, segment in enumerate(plan):
        slots.new_segment()
        chair_first = segment.primary_zone in (Zone.CONTEXT, Zone.CLOSING)
        mapping = _speaker_map(segment, participants, chair_first=chair_first, rng=rng)
        for i, sentence in enumerate(segment.sentences):
            filled = slots.fill(sentence.text, mapping)
            text = _capitalise_first(filled.text)
            rows.append(
                base(mapping[sentence.speaker])
                | {
                    "sentence_text": text,
                    "sentence_text_asr": render_asr(text, rng, cfg.noise),
                    "importance": sentence.importance,
                    "confidence": segment.confidence,
                    "zone": sentence.zone.value,
                    "segment_id": segment.segment_id,
                    "family_id": segment.family_id,
                    "hard_negative": segment.hard_negative,
                    "token_count": len(text.split()),
                    "spans": filled.spans,
                }
            )
            nxt = segment.sentences[i + 1] if i + 1 < len(segment.sentences) else None
            if (
                nxt is not None
                and sentence.importance == 0
                and nxt.importance == 0
                and rng.random() < comp.p_interrupt
            ):
                add_filler(None, mapping[sentence.speaker])
        if index < len(plan) - 1 and rng.random() < comp.p_filler:
            add_filler(None, rows[-1]["speaker"])
    if rng.random() < comp.p_closing_filler:
        add_filler(FillerTag.CLOSING_CHATTER, rows[-1]["speaker"] if rows else None)
    return rows


def compose_meeting(
    spec: MeetingSpec,
    pool: SegmentPool,
    fillers: Mapping[FillerTag | None, Sequence[Filler]],
    lexicons: Mapping[str, Sequence[str]],
    cfg: GenerationConfig,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Compose one meeting and return its sentence rows.

    Tries up to ``max_attempts`` plans until the rendered meeting fits the word range of its
    type; only the accepted plan counts toward segment reuse.
    """
    mt_cfg = cfg.meeting_types[spec.meeting_type]
    lo, hi = mt_cfg.words
    names = lexicons["PERSON"]
    for _ in range(cfg.composition.max_attempts):
        n_participants = rng.randint(*mt_cfg.participants)
        participants = rng.sample(list(names), n_participants)
        plan = _plan(spec, pool, n_participants, cfg, rng)
        rows = _render(spec, plan, participants, fillers, lexicons, cfg, rng)
        n_words = sum(r["token_count"] for r in rows)
        if lo <= n_words <= hi:
            for segment in plan:
                pool.uses[segment.segment_id] += 1
            for i, row in enumerate(rows):
                row["sentence_id"] = i
            return rows
    raise CompositionError(
        f"{spec.note_id}: no plan fits {lo}-{hi} words in {cfg.composition.max_attempts} attempts"
    )
