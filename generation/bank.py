"""Load the committed seed bank: segments, hard negatives, fillers and lexicons.

Loading re-validates every object against ``generation.schema``. Run
``processing.validate_seeds`` for the full set of checks; this loader only refuses
malformed seeds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from generation.config import REPO_ROOT
from generation.schema import Filler, FillerTag, SeedSegment
from generation.slots import load_lexicons

DEFAULT_SEEDS = REPO_ROOT / "seeds"
DEFAULT_LEXICONS = REPO_ROOT / "generation" / "lexicons"


@dataclass(frozen=True)
class SeedBank:
    """All seed material needed to compose meetings."""

    segments: tuple[SeedSegment, ...]
    hard_negatives: tuple[SeedSegment, ...]
    fillers: tuple[Filler, ...]
    lexicons: dict[str, list[str]]

    def fillers_with(self, tag: FillerTag | None) -> list[Filler]:
        """Fillers carrying ``tag`` (None = untagged, usable anywhere)."""
        return [f for f in self.fillers if f.tag == tag]

    @property
    def all_segments(self) -> tuple[SeedSegment, ...]:
        """Normal and hard-negative segments together."""
        return self.segments + self.hard_negatives


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_bank(seeds_dir: Path = DEFAULT_SEEDS, lexicons_dir: Path = DEFAULT_LEXICONS) -> SeedBank:
    """Load and validate the seed bank. Files are read in sorted order (determinism)."""
    segments = tuple(
        SeedSegment.model_validate(obj)
        for path in sorted((seeds_dir / "segments").glob("*/*.jsonl"))
        for obj in _read_jsonl(path)
    )
    hard_negatives = tuple(
        SeedSegment.model_validate(obj)
        for path in sorted((seeds_dir / "hard_negatives").glob("*.jsonl"))
        for obj in _read_jsonl(path)
    )
    fillers_path = seeds_dir / "fillers" / "fillers.jsonl"
    fillers = (
        tuple(Filler.model_validate(obj) for obj in _read_jsonl(fillers_path))
        if fillers_path.exists()
        else ()
    )
    return SeedBank(segments, hard_negatives, fillers, load_lexicons(lexicons_dir))
