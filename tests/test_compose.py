"""Tests for generation.compose and the generator helpers, on a fixture bank."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from generation.bank import SeedBank
from generation.compose import MeetingSpec, SegmentPool, compose_meeting
from generation.generate_synthetic_notes import allocate, build_pools, generate, meeting_specs
from generation.output import SENTENCE_SCHEMA, to_table, write_parquet
from generation.schema import Domain, FillerTag, MeetingType, SentenceRow, Source, Split
from tests.bank_fixture import FILLERS, LEXICONS, make_hard_negatives, make_segments, small_config


@pytest.fixture
def bank() -> SeedBank:
    return SeedBank(tuple(make_segments()), tuple(make_hard_negatives()), FILLERS, LEXICONS)


@pytest.fixture
def scfg(cfg):
    return small_config(cfg)


def _fillers(bank):
    return {tag: bank.fillers_with(tag) for tag in (None, *FillerTag)}


def _meeting(bank, scfg, seed=0, split=Split.TRAIN, n_hn=0, meeting_type=MeetingType.STANDUP):
    pools = build_pools(bank, scfg, scfg.seed)
    spec = MeetingSpec("syn-00001", meeting_type, Domain.SALES, split, Source.SYNTHETIC, n_hn)
    return compose_meeting(
        spec, pools[split], _fillers(bank), bank.lexicons, scfg, random.Random(seed)
    ), pools


def test_meeting_respects_word_range_and_row_schema(bank, scfg):
    for seed in range(10):
        rows, _ = _meeting(bank, scfg, seed)
        lo, hi = scfg.meeting_types[MeetingType.STANDUP].words
        assert lo <= sum(r["token_count"] for r in rows) <= hi
        assert [r["sentence_id"] for r in rows] == list(range(len(rows)))
        for row in rows:
            SentenceRow.model_validate(row)


def test_meeting_uses_only_its_split(bank, scfg):
    pools = build_pools(bank, scfg, scfg.seed)
    for split in Split:
        allowed = {s.segment_id for s in pools[split].segments}
        rows, _ = _meeting(bank, scfg, 1, split)
        used = {r["segment_id"] for r in rows if not r["segment_id"].startswith("fill-")}
        assert used <= allowed


def test_no_segment_or_family_repeats_within_a_meeting(bank, scfg):
    for seed in range(10):
        rows, _ = _meeting(bank, scfg, seed)
        placed = []
        for r in rows:
            if r["segment_id"].startswith("fill-"):
                continue
            if not placed or placed[-1][0] != r["segment_id"]:
                placed.append((r["segment_id"], r["family_id"]))
        assert len({p[0] for p in placed}) == len(placed)
        assert len({p[1] for p in placed}) == len(placed)


def test_speakers_are_consistent_within_segments(bank, scfg):
    rows, _ = _meeting(bank, scfg, 4)
    by_segment: dict[str, dict[str, str]] = {}
    segments = {s.segment_id: s for s in bank.all_segments}
    i = 0
    while i < len(rows):
        seg_id = rows[i]["segment_id"]
        if seg_id.startswith("fill-"):
            i += 1
            continue
        seg = segments[seg_id]
        mapping: dict[str, str] = {}
        for sentence in seg.sentences:
            while rows[i]["segment_id"].startswith("fill-"):
                i += 1
            assert rows[i]["segment_id"] == seg_id
            mapping.setdefault(sentence.speaker, rows[i]["speaker"])
            assert mapping[sentence.speaker] == rows[i]["speaker"]
            i += 1
        assert len(set(mapping.values())) == len(mapping)  # distinct people
        by_segment[seg_id] = mapping
    assert by_segment


def test_labels_are_copied_from_seeds(bank, scfg):
    segments = {s.segment_id: s for s in bank.all_segments}
    rows, _ = _meeting(bank, scfg, 2)
    seen: Counter[str] = Counter()
    for row in rows:
        if row["segment_id"].startswith("fill-"):
            assert (row["importance"], row["zone"]) == (0, "discussion")
            continue
        sentence = segments[row["segment_id"]].sentences[seen[row["segment_id"]]]
        seen[row["segment_id"]] += 1
        assert (row["importance"], row["zone"]) == (sentence.importance, sentence.zone.value)


def test_spans_point_at_the_owner_and_deadline(bank, scfg):
    for seed in range(10):
        rows, _ = _meeting(bank, scfg, seed)
        for row in rows:
            for span in row["spans"]:
                piece = row["sentence_text"][span["start"] : span["end"]]
                if span["role"] == "owner":
                    assert piece.startswith("Name")
                else:
                    assert piece in LEXICONS["DATE_REL"]


def test_hard_negative_meetings_contain_enough_hard_negatives(bank, scfg):
    rows, _ = _meeting(bank, scfg, 5, n_hn=3)
    hn_segments = {r["segment_id"] for r in rows if r["hard_negative"]}
    assert len(hn_segments) >= 3


def test_reuse_cap_is_respected(bank, cfg):
    raw = small_config(cfg).model_dump(mode="json")
    raw["composition"]["max_segment_reuse"] = 3
    from generation.config import GenerationConfig

    capped = GenerationConfig.model_validate(raw)
    pools = build_pools(bank, capped, capped.seed)
    specs = [
        s
        for s in meeting_specs("syn", Source.SYNTHETIC, 10, capped, random.Random(0))
        if s.split == Split.TRAIN
    ][:12]
    generate(specs, bank, pools, capped, capped.seed)
    assert max(pools[Split.TRAIN].uses.values()) <= 3


def test_allocate_sums_and_follows_ratios(cfg):
    counts = allocate(500, cfg.splits.ratios)
    assert counts == {Split.TRAIN: 400, Split.VALIDATION: 50, Split.TEST: 50}
    assert sum(allocate(7, cfg.splits.ratios).values()) == 7


def test_generation_is_byte_identical_for_a_seed(bank, scfg, tmp_path):
    def build(path):
        pools = build_pools(bank, scfg, scfg.seed)
        specs = meeting_specs("syn", Source.SYNTHETIC, 10, scfg, random.Random(scfg.seed))
        write_parquet(
            to_table(generate(specs, bank, pools, scfg, scfg.seed), SENTENCE_SCHEMA), path
        )
        return path.read_bytes()

    assert build(tmp_path / "a.parquet") == build(tmp_path / "b.parquet")


def test_different_seed_changes_output(bank, scfg):
    def rows(seed):
        pools = build_pools(bank, scfg, seed)
        specs = meeting_specs("syn", Source.SYNTHETIC, 10, scfg, random.Random(seed))
        return [r["sentence_text"] for r in generate(specs[:5], bank, pools, scfg, seed)]

    assert rows(1) != rows(2)


def test_pool_indexes_by_zone(bank):
    pool = SegmentPool(bank.segments, bank.hard_negatives)
    assert sum(len(v) for v in pool.by_zone.values()) == len(bank.segments)
