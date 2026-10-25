"""Generate the v1.1 addendum meetings: ``addendum.parquet`` (PLAN §9c).

    uv run python -m generation.generate_addendum [--config ...] [--seed N]

The addendum seeds (``seeds/addendum/segments``) teach what v1.0 lacked: tasks volunteered in
the first person that nobody restates, terse decisions answering a proposal, and informal
conversations (``personal`` domain). They are composed into **new** meetings (``add-…``,
``addp-…``) with the v1.0 composer, so the v1.0 meetings stay byte-identical:

- the addendum families get their own 80/10/10 split; v1.0 families keep theirs;
- decision and action-item segments come only from the addendum, so every addendum meeting
  shows the new patterns; context, discussion and closing segments come from both banks;
- business meetings follow the v1.0 meeting types and domains, capped at
  ``addendum.max_words`` (the small validation / test pools must be able to fill them), and
  never use personal segments;
- personal meetings are short quick syncs (``addendum.personal_words``) made only of personal
  segments, all in train (too few personal families for validation / test meetings).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from generation.bank import DEFAULT_SEEDS, SeedBank, load_bank
from generation.compose import MeetingSpec, SegmentPool
from generation.config import DEFAULT_CONFIG, GenerationConfig, load_config
from generation.generate_synthetic_notes import generate, meeting_specs
from generation.output import SENTENCE_SCHEMA, SYNTHETIC_DIR, to_table, write_parquet
from generation.schema import Domain, MeetingType, SeedSegment, Source, Split, Zone
from processing.assign_splits import SPLIT_ORDER, assign_family_splits

ADDENDUM_DIR = DEFAULT_SEEDS / "addendum" / "segments"
ADDENDUM_ONLY = (Zone.DECISION, Zone.ACTION_ITEM)


def load_addendum(directory: Path = ADDENDUM_DIR) -> tuple[SeedSegment, ...]:
    """The addendum segments, validated against the seed schema (sorted file order)."""
    return tuple(
        SeedSegment.model_validate(json.loads(line))
        for path in sorted(directory.glob("*/*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def addendum_pools(
    bank: SeedBank, addendum: tuple[SeedSegment, ...], cfg: GenerationConfig, seed: int
) -> tuple[dict[Split, SegmentPool], SegmentPool]:
    """Business pools per split (addendum business segments, all zones + v1.0 context /
    discussion / closing) and one personal pool (every personal segment, train only)."""
    business = [s for s in addendum if s.domain != Domain.PERSONAL]
    personal = [s for s in addendum if s.domain == Domain.PERSONAL]
    base_split = assign_family_splits(bank.all_segments, cfg.splits.ratios, seed)
    add_split = assign_family_splits(business, cfg.splits.ratios, seed + cfg.addendum.seed_offset)
    uses: Counter[str] = Counter()
    pools = {}
    for split in SPLIT_ORDER:
        base = [
            s
            for s in bank.segments
            if base_split[s.family_id] == split and s.primary_zone not in ADDENDUM_ONLY
        ]
        new = [s for s in business if add_split[s.family_id] == split]
        pools[split] = SegmentPool(segments=[*base, *new], hard_negatives=[], uses=uses)
    return pools, SegmentPool(segments=personal, hard_negatives=[], uses=uses)


def personal_specs(n: int) -> list[MeetingSpec]:
    """``n`` personal quick syncs, all in train: the 33 personal families are too few to
    compose validation / test meetings from their own split (tested qualitatively instead)."""
    return [
        MeetingSpec(
            note_id=f"addp-{i:05d}",
            meeting_type=MeetingType.QUICK_SYNC,
            domain=Domain.PERSONAL,
            split=Split.TRAIN,
            source=Source.SYNTHETIC,
        )
        for i in range(1, n + 1)
    ]


def capped(cfg: GenerationConfig, low: int, high: int) -> GenerationConfig:
    """``cfg`` with every meeting type's word range clipped to ``[low, high]``."""
    return cfg.model_copy(
        update={
            "meeting_types": {
                mt: m.model_copy(update={"words": (min(m.words[0], low), min(m.words[1], high))})
                for mt, m in cfg.meeting_types.items()
            }
        }
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--addendum", type=Path, default=ADDENDUM_DIR)
    parser.add_argument("--out", type=Path, default=SYNTHETIC_DIR / "addendum.parquet")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    seed = cfg.seed if args.seed is None else args.seed
    add = cfg.addendum
    add_cfg = cfg.model_copy(
        update={
            "composition": cfg.composition.model_copy(update={"p_cross_domain": add.p_cross_domain})
        }
    )
    bank, addendum = load_bank(), load_addendum(args.addendum)
    pools, personal_pool = addendum_pools(bank, addendum, cfg, seed)
    rng = random.Random(seed + add.seed_offset)
    business = meeting_specs("add", Source.SYNTHETIC, add.business_per_type, cfg, rng)
    personal = personal_specs(add.personal_meetings)
    stream = seed + add.seed_offset
    rows = generate(
        business, bank, pools, capped(add_cfg, add.max_words // 2, add.max_words), stream
    )
    rows += generate(
        personal,
        bank,
        {split: personal_pool for split in SPLIT_ORDER},
        capped(add_cfg, *add.personal_words),
        stream,
    )
    specs = business + personal
    write_parquet(to_table(rows, SENTENCE_SCHEMA), args.out)
    print(f"wrote {args.out}: {len(specs)} meetings, {len(rows)} sentences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
