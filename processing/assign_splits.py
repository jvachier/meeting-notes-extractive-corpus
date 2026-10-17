"""Assign seed families to train / validation / test (PLAN §4.4).

Splits are per **family**, never per meeting. Normal families are stratified by
(primary_zone, domain), hard-negative families by category. Within the normal set and
within the hard-negative set the global split shares match the configured ratios exactly
(up to one family), and each stratum stays as close to them as its size allows.

This runs *before* composition: a meeting is composed only from the families of one split.

    uv run python -m processing.assign_splits [--config ...] [--seed N]
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

from generation.bank import load_bank
from generation.config import DEFAULT_CONFIG, load_config
from generation.output import FAMILY_SPLIT_SCHEMA, PROCESSED_DIR, to_table, write_parquet
from generation.schema import SeedSegment, Split

SPLIT_ORDER: tuple[Split, ...] = (Split.TRAIN, Split.VALIDATION, Split.TEST)


def stratum_of(segment: SeedSegment) -> str:
    """Stratum key: ``hn/<category>`` or ``<zone>/<domain>``."""
    if segment.hn_category is not None:
        return f"hn/{segment.hn_category.value}"
    return f"{segment.primary_zone.value}/{segment.domain.value}"


def _assign(
    strata: Mapping[str, list[str]], ratios: Mapping[Split, float], rng: random.Random
) -> dict[str, Split]:
    """Walk strata in sorted order, families shuffled within each, and give each family the
    split with the largest running deficit ``ratio * (k + 1) - count``."""
    counts = dict.fromkeys(SPLIT_ORDER, 0)
    assigned: dict[str, Split] = {}
    k = 0
    for stratum in sorted(strata):
        families = sorted(strata[stratum])
        rng.shuffle(families)
        for family in families:
            split = max(
                SPLIT_ORDER, key=lambda s: (ratios[s] * (k + 1) - counts[s], -SPLIT_ORDER.index(s))
            )
            assigned[family] = split
            counts[split] += 1
            k += 1
    return assigned


def assign_family_splits(
    segments: Iterable[SeedSegment], ratios: Mapping[Split, float], seed: int
) -> dict[str, Split]:
    """Return ``family_id → split`` for every family in ``segments``."""
    normal: dict[str, set[str]] = defaultdict(set)
    hard: dict[str, set[str]] = defaultdict(set)
    for segment in segments:
        target = hard if segment.hard_negative else normal
        target[stratum_of(segment)].add(segment.family_id)
    rng = random.Random(seed)
    out = _assign({k: sorted(v) for k, v in normal.items()}, ratios, rng)
    out.update(_assign({k: sorted(v) for k, v in hard.items()}, ratios, rng))
    return out


def family_split_rows(segments: Iterable[SeedSegment], splits: Mapping[str, Split]) -> list[dict]:
    """One row per family for ``family_splits.parquet``."""
    rows = {}
    for segment in segments:
        rows[segment.family_id] = {
            "family_id": segment.family_id,
            "split": splits[segment.family_id].value,
            "stratum": stratum_of(segment),
            "hard_negative": segment.hard_negative,
        }
    return [rows[k] for k in sorted(rows)]


def main(argv: list[str] | None = None) -> int:
    """CLI: write ``data/processed/family_splits.parquet``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR / "family_splits.parquet")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    bank = load_bank()
    seed = cfg.seed if args.seed is None else args.seed
    splits = assign_family_splits(bank.all_segments, cfg.splits.ratios, seed)
    rows = family_split_rows(bank.all_segments, splits)
    write_parquet(to_table(rows, FAMILY_SPLIT_SCHEMA), args.out)
    counts = {s.value: sum(r["split"] == s.value for r in rows) for s in SPLIT_ORDER}
    print(f"wrote {args.out}: {len(rows)} families {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
