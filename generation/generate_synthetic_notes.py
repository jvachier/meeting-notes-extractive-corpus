"""Generate the normal synthetic meetings (PLAN §4.3): ``synthetic_notes.parquet``.

    uv run python -m generation.generate_synthetic_notes [--config ...] [--seed N]

Meetings per type come from ``volumes.synthetic_notes_per_type``; split counts follow the
split ratios; domains are spread evenly. Each meeting uses only segments of its split.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa

from generation.bank import SeedBank, load_bank
from generation.compose import MeetingSpec, SegmentPool, compose_meeting
from generation.config import DEFAULT_CONFIG, GenerationConfig, load_config
from generation.output import SENTENCE_SCHEMA, SYNTHETIC_DIR, to_table, write_parquet
from generation.schema import FillerTag, MeetingType, Source, Split
from processing.assign_splits import SPLIT_ORDER, assign_family_splits


def allocate(n: int, ratios: Mapping[Split, float]) -> dict[Split, int]:
    """Split ``n`` items by ``ratios`` with the largest-remainder method (sums to ``n``)."""
    raw = {s: n * ratios[s] for s in SPLIT_ORDER}
    counts = {s: int(raw[s]) for s in SPLIT_ORDER}
    order = sorted(SPLIT_ORDER, key=lambda s: (-(raw[s] - counts[s]), SPLIT_ORDER.index(s)))
    for s in order[: n - sum(counts.values())]:
        counts[s] += 1
    return counts


def meeting_specs(
    prefix: str,
    source: Source,
    per_type: int,
    cfg: GenerationConfig,
    rng: random.Random,
    n_hard_negatives: Callable[[random.Random], int] = lambda _rng: 0,
) -> list[MeetingSpec]:
    """Specs for ``per_type`` meetings of every type, balanced over splits and domains."""
    specs: list[MeetingSpec] = []
    number = 0
    for meeting_type in MeetingType:
        counts = allocate(per_type, cfg.splits.ratios)
        splits = [s for s in SPLIT_ORDER for _ in range(counts[s])]
        rng.shuffle(splits)
        domains = list(cfg.domains)
        rng.shuffle(domains)
        for i, split in enumerate(splits):
            number += 1
            specs.append(
                MeetingSpec(
                    note_id=f"{prefix}-{number:05d}",
                    meeting_type=meeting_type,
                    domain=domains[i % len(domains)],
                    split=split,
                    source=source,
                    n_hard_negatives=n_hard_negatives(rng),
                )
            )
    return specs


def build_pools(
    bank: SeedBank, cfg: GenerationConfig, seed: int, uses: Counter[str] | None = None
) -> dict[Split, SegmentPool]:
    """One pool per split; all pools share one usage counter."""
    family_split = assign_family_splits(bank.all_segments, cfg.splits.ratios, seed)
    shared: Counter[str] = uses if uses is not None else Counter()
    return {
        split: SegmentPool(
            segments=[s for s in bank.segments if family_split[s.family_id] == split],
            hard_negatives=[s for s in bank.hard_negatives if family_split[s.family_id] == split],
            uses=shared,
        )
        for split in SPLIT_ORDER
    }


def generate(
    specs: Sequence[MeetingSpec],
    bank: SeedBank,
    pools: Mapping[Split, SegmentPool],
    cfg: GenerationConfig,
    seed: int,
) -> list[dict[str, Any]]:
    """Compose every meeting; each gets its own generator seeded from ``seed`` and its id."""
    fillers = {tag: bank.fillers_with(tag) for tag in (None, *FillerTag)}
    rows: list[dict[str, Any]] = []
    for spec in specs:
        rng = random.Random(f"{seed}:{spec.note_id}")
        rows.extend(compose_meeting(spec, pools[spec.split], fillers, bank.lexicons, cfg, rng))
    return rows


def segment_uses(table: pa.Table) -> Counter[str]:
    """Placements per segment in an existing sentences table (fillers excluded)."""
    pairs = {
        (n, s)
        for n, s in zip(
            table.column("note_id").to_pylist(), table.column("segment_id").to_pylist(), strict=True
        )
        if s is not None and not s.startswith("fill-")
    }
    return Counter(s for _, s in pairs)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=Path, default=SYNTHETIC_DIR / "synthetic_notes.parquet")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    seed = cfg.seed if args.seed is None else args.seed
    bank = load_bank()
    pools = build_pools(bank, cfg, seed)
    specs = meeting_specs(
        "syn", Source.SYNTHETIC, cfg.volumes.synthetic_notes_per_type, cfg, random.Random(seed)
    )
    rows = generate(specs, bank, pools, cfg, seed)
    write_parquet(to_table(rows, SENTENCE_SCHEMA), args.out)
    print(f"wrote {args.out}: {len(specs)} meetings, {len(rows)} sentences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
