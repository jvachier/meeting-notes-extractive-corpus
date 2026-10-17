"""Run the whole pipeline in order, from seeds to the local release folder.

    uv run python -m processing.build                 # full config
    uv run python -m processing.build --config configs/generation_tiny.yaml --root /tmp/x

Steps: family splits → synthetic meetings → hard-negative meetings → AMI (if the archive is
present) → unify → quality report → release folder. Publishing is not part of the build.
"""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path

from generation import generate_hard_negatives, generate_synthetic_notes, relabel_ami
from generation.config import DEFAULT_CONFIG, load_config
from generation.output import DATA_DIR, NOTE_SCHEMA, SENTENCE_SCHEMA, to_table, write_parquet
from processing import assign_splits, export_parquet, quality_checks, unify

TINY_CONFIG = DEFAULT_CONFIG.parent / "generation_tiny.yaml"


def run_pipeline(
    config: Path, root: Path, seed: int | None = None, *, report: bool = True
) -> dict[str, Path]:
    """Build everything under ``root`` (a ``data/``-like folder); return the main paths."""
    cfg = load_config(config)
    seed_args = [] if seed is None else ["--seed", str(seed)]
    common = ["--config", str(config), *seed_args]
    paths = {
        "splits": root / "processed" / "family_splits.parquet",
        "synthetic": root / "synthetic" / "synthetic_notes.parquet",
        "hard_negatives": root / "synthetic" / "hard_negatives.parquet",
        "ami": root / "processed" / "ami_labelled.parquet",
        "ami_notes": root / "processed" / "ami_notes.parquet",
        "sentences": root / "processed" / "unified_corpus.parquet",
        "notes": root / "processed" / "unified_notes.parquet",
        "report": root / "processed" / "report.md",
        "release": root / "release" / "v1",
    }
    assign_splits.main([*common, "--out", str(paths["splits"])])
    generate_synthetic_notes.main([*common, "--out", str(paths["synthetic"])])
    generate_hard_negatives.main(
        [*common, "--synthetic", str(paths["synthetic"]), "--out", str(paths["hard_negatives"])]
    )
    archive = DATA_DIR / "raw" / "ami" / f"ami_public_manual_{cfg.ami.version}.zip"
    if archive.exists():
        relabel_ami.main(
            [
                *common,
                "--archive",
                str(archive),
                "--out",
                str(paths["ami"]),
                "--notes-out",
                str(paths["ami_notes"]),
            ]
        )
    else:  # no archive (e.g. CI): empty AMI layer
        print(f"AMI archive not found at {archive}; building without the AMI layer")
        write_parquet(to_table([], SENTENCE_SCHEMA), paths["ami"])
        write_parquet(to_table([], NOTE_SCHEMA), paths["ami_notes"])
    unify.main(
        [
            "--synthetic", str(paths["synthetic"]),
            "--hard-negatives", str(paths["hard_negatives"]),
            "--ami", str(paths["ami"]),
            "--ami-notes", str(paths["ami_notes"]),
            "--out", str(paths["sentences"]),
            "--notes-out", str(paths["notes"]),
        ]
    )  # fmt: skip
    if report:
        quality_checks.main(
            [
                "--config",
                str(config),
                "--sentences",
                str(paths["sentences"]),
                "--notes",
                str(paths["notes"]),
                "--out",
                str(paths["report"]),
            ]
        )
    export_parquet.main(
        [
            "--config",
            str(config),
            "--sentences",
            str(paths["sentences"]),
            "--notes",
            str(paths["notes"]),
            "--out",
            str(paths["release"]),
        ]
    )
    return paths


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def determinism_hashes(config: Path = TINY_CONFIG) -> dict[str, tuple[str, str]]:
    """Build ``config`` twice in temporary folders; return per-file hash pairs."""
    names = ["splits", "synthetic", "hard_negatives", "ami", "sentences", "notes"]
    runs = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as tmp:
            paths = run_pipeline(config, Path(tmp), report=False)
            digests = {n: _digest(paths[n]) for n in names}
            release = sorted((paths["release"] / "data").rglob("*.parquet"))
            digests.update({str(p.relative_to(paths["release"])): _digest(p) for p in release})
            runs.append(digests)
    return {name: (runs[0][name], runs[1][name]) for name in runs[0]}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--root", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)
    run_pipeline(args.config, args.root, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
