"""Tests for processing.assign_splits."""

from __future__ import annotations

from collections import Counter, defaultdict

from processing.assign_splits import assign_family_splits, stratum_of
from tests.bank_fixture import make_hard_negatives, make_segments


def test_every_family_gets_exactly_one_split(cfg):
    segments = make_segments(30) + make_hard_negatives(20)
    splits = assign_family_splits(segments, cfg.splits.ratios, 7)
    assert set(splits) == {s.family_id for s in segments}
    # variants of one family share the split by construction (keyed by family_id)
    assert len(splits) == len({s.family_id for s in segments})


def test_global_ratios_are_exact_and_strata_within_tolerance(cfg):
    segments = make_segments(30)
    splits = assign_family_splits(segments, cfg.splits.ratios, 7)
    counts = Counter(splits.values())
    n = len(splits)
    for split, ratio in cfg.splits.ratios.items():
        assert abs(counts[split] - ratio * n) <= 1
    strata: dict[str, list[str]] = defaultdict(list)
    for s in segments:
        if s.family_id not in strata[stratum_of(s)]:
            strata[stratum_of(s)].append(s.family_id)
    for families in strata.values():
        share = Counter(splits[f] for f in families)
        for split, ratio in cfg.splits.ratios.items():
            assert abs(share[split] / len(families) - ratio) <= cfg.splits.stratify_tolerance


def test_hard_negatives_are_split_separately(cfg):
    hn = make_hard_negatives(20)
    splits = assign_family_splits(make_segments(10) + hn, cfg.splits.ratios, 1)
    hn_counts = Counter(splits[f] for f in {s.family_id for s in hn})
    assert hn_counts == {"train": 16, "validation": 2, "test": 2}


def test_assignment_is_deterministic(cfg):
    segments = make_segments(10)
    assert assign_family_splits(segments, cfg.splits.ratios, 3) == assign_family_splits(
        segments, cfg.splits.ratios, 3
    )
