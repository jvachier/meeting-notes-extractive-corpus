"""Build the v1.1 release from the v1.0 intermediate files (PLAN §9c).

    uv run python -m processing.build_v1_1 [--root data]

1. The v1.0 synthetic and hard-negative meetings are relabelled in place from the current
   seeds (:mod:`processing.relabel_from_seeds`); only the edits listed in
   ``seeds/addendum/relabelled.json`` may change a label, and the text stays byte-identical.
2. The addendum meetings (``generation.generate_addendum``) are added to the synthetic layer.
3. AMI is unchanged. Then unify, quality report and release export, as in ``processing.build``,
   into ``<root>/release/v1.1``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa

from generation import generate_addendum
from generation.bank import DEFAULT_SEEDS, load_bank
from generation.config import DEFAULT_CONFIG
from generation.output import DATA_DIR, read_parquet, write_parquet
from processing import export_parquet, quality_checks, unify
from processing.relabel_from_seeds import relabel_table

EDITS = DEFAULT_SEEDS / "addendum" / "relabelled.json"


def edited_positions(path: Path = EDITS) -> set[tuple[str, int]]:
    """``(segment_id, index)`` of every seed sentence whose labels were edited for v1.1."""
    return {(e["segment_id"], e["index"]) for e in json.loads(path.read_text())["edits"]}


def build(root: Path, config: Path = DEFAULT_CONFIG) -> dict[str, Path]:
    """Write the v1.1 intermediate files and release under ``root``; return the main paths."""
    v10 = {
        "synthetic": root / "synthetic" / "synthetic_notes.parquet",
        "hard_negatives": root / "synthetic" / "hard_negatives.parquet",
        "ami": root / "processed" / "ami_labelled.parquet",
        "ami_notes": root / "processed" / "ami_notes.parquet",
    }
    out = root / "processed" / "v1.1"
    paths = {
        "synthetic": out / "synthetic_notes.parquet",
        "hard_negatives": out / "hard_negatives.parquet",
        "addendum": root / "synthetic" / "addendum.parquet",
        "sentences": out / "unified_corpus.parquet",
        "notes": out / "unified_notes.parquet",
        "report": out / "report.md",
        "release": root / "release" / "v1.1",
    }
    bank, changed = load_bank(), edited_positions()
    counts = {}
    for name in ("synthetic", "hard_negatives"):
        table, counts[name] = relabel_table(read_parquet(v10[name]), bank.all_segments, changed)
        write_parquet(table, paths[name])
    print(f"relabelled rows: {counts}")
    generate_addendum.main(["--config", str(config), "--out", str(paths["addendum"])])
    synthetic = pa.concat_tables(
        [read_parquet(paths["synthetic"]), read_parquet(paths["addendum"])]
    )
    write_parquet(synthetic, paths["synthetic"])
    unify.main(
        [
            "--synthetic", str(paths["synthetic"]),
            "--hard-negatives", str(paths["hard_negatives"]),
            "--ami", str(v10["ami"]),
            "--ami-notes", str(v10["ami_notes"]),
            "--out", str(paths["sentences"]),
            "--notes-out", str(paths["notes"]),
        ]
    )  # fmt: skip
    quality_checks.main(
        [
            "--config", str(config),
            "--sentences", str(paths["sentences"]),
            "--notes", str(paths["notes"]),
            "--out", str(paths["report"]),
        ]
    )  # fmt: skip
    export_parquet.main(
        [
            "--config", str(config),
            "--sentences", str(paths["sentences"]),
            "--notes", str(paths["notes"]),
            "--out", str(paths["release"]),
        ]
    )  # fmt: skip
    return paths


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DATA_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    build(args.root, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
