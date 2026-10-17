"""Generate meetings that contain hard negatives (PLAN §4.3): ``hard_negatives.parquet``.

    uv run python -m generation.generate_hard_negatives [--config ...] [--seed N]

Each meeting holds ``volumes.hard_negative_segments`` hard-negative segments mixed with
normal ones. If ``synthetic_notes.parquet`` exists, its segment usage is loaded first so
``max_segment_reuse`` holds across both files.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from generation.bank import load_bank
from generation.config import DEFAULT_CONFIG, load_config
from generation.generate_synthetic_notes import build_pools, generate, meeting_specs, segment_uses
from generation.output import SENTENCE_SCHEMA, SYNTHETIC_DIR, read_parquet, to_table, write_parquet
from generation.schema import Source


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--synthetic", type=Path, default=SYNTHETIC_DIR / "synthetic_notes.parquet")
    parser.add_argument("--out", type=Path, default=SYNTHETIC_DIR / "hard_negatives.parquet")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    seed = cfg.seed if args.seed is None else args.seed
    bank = load_bank()
    uses = segment_uses(read_parquet(args.synthetic)) if args.synthetic.exists() else Counter()
    pools = build_pools(bank, cfg, seed, uses)
    lo, hi = cfg.volumes.hard_negative_segments
    specs = meeting_specs(
        "hn",
        Source.SYNTHETIC_HN,
        cfg.volumes.hard_negative_notes_per_type,
        cfg,
        random.Random(f"{seed}:hard_negatives"),
        n_hard_negatives=lambda rng: rng.randint(lo, hi),
    )
    rows = generate(specs, bank, pools, cfg, seed)
    write_parquet(to_table(rows, SENTENCE_SCHEMA), args.out)
    print(f"wrote {args.out}: {len(specs)} meetings, {len(rows)} sentences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
