"""Publish the release folder to the Hugging Face Hub and Kaggle (PLAN §6, phase G6).

    uv run python -m processing.publish --dry-run --hf --kaggle   # show what would happen
    HF_TOKEN=... uv run --extra publish python -m processing.publish --hf
    uv run --extra publish python -m processing.publish --kaggle   # needs ~/.kaggle/kaggle.json

Nothing is published without an explicit ``--hf`` or ``--kaggle``. The release must have been
built from a clean, committed tree (its ``stats.json`` commit equals ``HEAD``), so the dataset
always points at the code that produced it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from generation.config import REPO_ROOT
from generation.output import RELEASE_DIR
from processing.export_parquet import HF_ID

TAG = "v1.0"
REQUIRED = ("README.md", "stats.json", "dataset-metadata.json")


class PublishError(RuntimeError):
    """The release is not in a publishable state."""


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def check_release(release: Path, *, require_clean_commit: bool = True) -> dict:
    """Check the folder is complete and built from the current clean commit; return stats."""
    missing = [name for name in REQUIRED if not (release / name).exists()]
    parquet = sorted((release / "data").rglob("*.parquet")) if (release / "data").exists() else []
    if missing or not parquet:
        raise PublishError(f"incomplete release at {release}: missing {missing or 'Parquet files'}")
    stats = json.loads((release / "stats.json").read_text(encoding="utf-8"))
    if require_clean_commit:
        if _git("status", "--porcelain"):
            raise PublishError(
                "working tree has uncommitted changes: commit, rebuild, then publish"
            )
        head = _git("rev-parse", "HEAD")
        if stats["provenance"]["git_commit"] != head:
            raise PublishError(
                f"release was built from {stats['provenance']['git_commit'][:12]}, HEAD is "
                f"{head[:12]}: rebuild with `uv run python -m processing.build`"
            )
    return stats


def publish_hf(release: Path, repo_id: str, tag: str) -> None:
    """Upload the folder as a dataset repo and tag the revision consumers pin."""
    from huggingface_hub import HfApi  # optional dependency (extra "publish")

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(release),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Release {tag}",
    )
    api.create_tag(repo_id, tag=tag, repo_type="dataset", exist_ok=True)


def kaggle_command(release: Path, *, new: bool, tag: str) -> list[str]:
    """The Kaggle CLI call: ``create`` for the first upload, ``version`` afterwards."""
    if new:
        return ["kaggle", "datasets", "create", "-p", str(release), "--dir-mode", "zip"]
    return [
        "kaggle",
        "datasets",
        "version",
        "-p",
        str(release),
        "-m",
        f"Release {tag}",
        "--dir-mode",
        "zip",
    ]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--release", type=Path, default=RELEASE_DIR / "v1")
    parser.add_argument("--hf", action="store_true", help="publish to the Hugging Face Hub")
    parser.add_argument("--kaggle", action="store_true", help="publish to Kaggle")
    parser.add_argument("--kaggle-new", action="store_true", help="first Kaggle upload (create)")
    parser.add_argument("--repo-id", default=HF_ID)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not (args.hf or args.kaggle):
        parser.error("choose --hf and/or --kaggle")
    try:
        stats = check_release(args.release, require_clean_commit=not args.dry_run)
    except PublishError as exc:
        print(f"refusing to publish: {exc}")
        return 1
    commit = stats["provenance"]["git_commit"][:12]
    if args.hf:
        print(
            f"Hugging Face: upload {args.release} to {args.repo_id}, "
            f"tag {args.tag} (commit {commit})"
        )
        if not args.dry_run:
            publish_hf(args.release, args.repo_id, args.tag)
    if args.kaggle:
        cmd = kaggle_command(args.release, new=args.kaggle_new, tag=args.tag)
        print("Kaggle: " + " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)
    print("dry run, nothing published" if args.dry_run else "published")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
