"""Load and validate ``configs/generation*.yaml`` and ``seeds/coverage.yaml``.

A config may name a parent with ``extends: <file>`` (relative to its own folder); its keys are
deep-merged over the parent's.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from generation.schema import Domain, HNCategory, MeetingType, Split, Zone
from generation.slots import SLOT_NAMES

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "generation.yaml"
DEFAULT_COVERAGE = REPO_ROOT / "seeds" / "coverage.yaml"

BODY_ZONES: frozenset[Zone] = frozenset({Zone.DISCUSSION, Zone.DECISION, Zone.ACTION_ITEM})


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _check_range(value: tuple[float, float]) -> tuple[float, float]:
    if value[0] > value[1]:
        raise ValueError(f"range low > high: {value}")
    return value


def _check_probability(value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"probability out of [0, 1]: {value}")
    return value


class MeetingTypeConfig(_Strict):
    """Length, participants and grammar weights for one meeting type."""

    words: tuple[int, int]
    participants: tuple[int, int]
    context_segments: tuple[int, int]
    zone_weights: dict[Zone, float]

    _ranges = field_validator("words", "participants", "context_segments")(_check_range)

    @field_validator("zone_weights")
    @classmethod
    def _weights(cls, value: dict[Zone, float]) -> dict[Zone, float]:
        if set(value) != BODY_ZONES:
            raise ValueError("zone_weights must cover exactly discussion, decision, action_item")
        if any(w < 0 for w in value.values()) or not math.isclose(sum(value.values()), 1.0):
            raise ValueError("zone_weights must be non-negative and sum to 1")
        return value


class CompositionConfig(_Strict):
    """Composition probabilities and caps (PLAN §4.2)."""

    closing_segments: tuple[int, int]
    p_cross_domain: float
    p_filler: float
    p_interrupt: float
    p_opening_filler: float
    p_closing_filler: float
    max_segment_reuse: int = Field(ge=1)
    max_attempts: int = Field(ge=1)
    type_weight: float = Field(ge=1.0)
    domain_weight: float = Field(ge=1.0)
    reuse_penalty_power: float = Field(ge=0.0)
    segment_scoped_slots: list[str] = Field(default_factory=list)

    @field_validator("segment_scoped_slots")
    @classmethod
    def _scoped(cls, value: list[str]) -> list[str]:
        unknown = set(value) - SLOT_NAMES
        if unknown:
            raise ValueError(f"unknown slots in segment_scoped_slots: {sorted(unknown)}")
        return value

    _range = field_validator("closing_segments")(_check_range)
    _probs = field_validator(
        "p_cross_domain", "p_filler", "p_interrupt", "p_opening_filler", "p_closing_filler"
    )(_check_probability)


class NoiseConfig(_Strict):
    """ASR-style noise (PLAN §4.2 step 6)."""

    p_disfl: float
    disfluency_tokens: list[str] = Field(min_length=1)
    p_repetition: float

    _probs = field_validator("p_disfl", "p_repetition")(_check_probability)


class VolumesConfig(_Strict):
    """Meeting counts (PLAN §4.3)."""

    synthetic_notes_per_type: int = Field(ge=1)
    hard_negative_notes_per_type: int = Field(ge=1)
    hard_negative_segments: tuple[int, int]

    @field_validator("hard_negative_segments")
    @classmethod
    def _hn_range(cls, value: tuple[int, int]) -> tuple[int, int]:
        if value[0] < 1:
            raise ValueError("hard-negative meetings need at least one hard-negative segment")
        return _check_range(value)


class SplitsConfig(_Strict):
    """Split ratios (PLAN §4.4)."""

    ratios: dict[Split, float]
    stratify_tolerance: float = Field(ge=0.0, le=1.0)

    @field_validator("ratios")
    @classmethod
    def _ratios(cls, value: dict[Split, float]) -> dict[Split, float]:
        if set(value) != set(Split) or not math.isclose(sum(value.values()), 1.0):
            raise ValueError("ratios must cover train/validation/test and sum to 1")
        return value


class SeedsConfig(_Strict):
    """Thresholds used by ``processing.validate_seeds``."""

    segment_sentences: tuple[int, int]
    segment_confidence: tuple[float, float]
    hard_negative_confidence: tuple[float, float]
    discussion_importance_target: float
    discussion_importance_tolerance: float
    discussion_rate_min_segments: int = Field(ge=1)
    variant_similarity_max: float
    fillers_target: int = Field(ge=1)

    _ranges = field_validator(
        "segment_sentences", "segment_confidence", "hard_negative_confidence"
    )(_check_range)
    _probs = field_validator(
        "discussion_importance_target", "discussion_importance_tolerance", "variant_similarity_max"
    )(_check_probability)


class LexiconsConfig(_Strict):
    """Minimum lexicon sizes (PLAN §3.2)."""

    min_values: dict[str, int]

    @field_validator("min_values")
    @classmethod
    def _slots(cls, value: dict[str, int]) -> dict[str, int]:
        if set(value) != SLOT_NAMES:
            raise ValueError(f"min_values must list exactly the slots {sorted(SLOT_NAMES)}")
        return value


class QualityConfig(_Strict):
    """Thresholds for ``processing.quality_checks`` (PLAN §7)."""

    ngram_n: int = Field(ge=1)
    max_test_ngram_overlap: float
    near_duplicate_ngram: int = Field(ge=1)
    near_duplicate_jaccard: float
    top_trigger_words: int = Field(ge=1)
    trigger_min_count: int = Field(ge=1)
    trigger_min_families: int = Field(ge=1)
    rule_keywords: list[str] = Field(min_length=1)
    max_rule_f1_gap: float

    _probs = field_validator("max_test_ngram_overlap", "near_duplicate_jaccard")(_check_probability)


class AmiConfidence(_Strict):
    """Fixed confidences per AMI annotation status (PLAN §5)."""

    extracted_linked: float
    extracted_other: float
    not_extracted: float


class AmiConfig(_Strict):
    """AMI download and labelling settings (PLAN §5)."""

    version: str
    url: str
    confidence: AmiConfidence
    partition: dict[Split, list[str]]

    @field_validator("partition")
    @classmethod
    def _partition(cls, value: dict[Split, list[str]]) -> dict[Split, list[str]]:
        if Split.TRAIN in value:
            raise ValueError("partition lists validation and test series; the rest is train")
        series = [s for names in value.values() for s in names]
        if len(series) != len(set(series)):
            raise ValueError("a meeting series appears in two splits")
        return value

    def split_of(self, meeting_id: str) -> Split:
        """Split of an AMI meeting id such as ``ES2004c``."""
        for split, series in self.partition.items():
            if any(meeting_id.startswith(s) for s in series):
                return split
        return Split.TRAIN


class AddendumConfig(_Strict):
    """v1.1 addendum meetings (PLAN §9c, generation/generate_addendum.py)."""

    business_per_type: int = Field(ge=0)
    personal_meetings: int = Field(ge=0)
    p_cross_domain: float = Field(ge=0.0, le=1.0)
    seed_offset: int = Field(ge=1)
    max_words: int = Field(ge=50)
    personal_words: tuple[int, int]


class GenerationConfig(_Strict):
    """Root of ``configs/generation.yaml``."""

    seed: int
    meeting_types: dict[MeetingType, MeetingTypeConfig]
    domains: list[Domain]
    composition: CompositionConfig
    noise: NoiseConfig
    volumes: VolumesConfig
    splits: SplitsConfig
    seeds: SeedsConfig
    lexicons: LexiconsConfig
    quality: QualityConfig
    addendum: AddendumConfig | None = None
    ami: AmiConfig

    @model_validator(mode="after")
    def _complete(self) -> GenerationConfig:
        if set(self.meeting_types) != set(MeetingType):
            raise ValueError("meeting_types must configure all five meeting types")
        business = sorted(d for d in Domain if d != Domain.PERSONAL)
        if sorted(self.domains) != business or len(self.domains) != len(business):
            raise ValueError("domains must list the eight business domains once")
        return self


class HardNegativeCoverage(_Strict):
    """Target families per hard-negative category (PLAN §3.5)."""

    variants_per_family: int = Field(ge=1)
    categories: dict[HNCategory, int]

    @field_validator("categories")
    @classmethod
    def _all(cls, value: dict[HNCategory, int]) -> dict[HNCategory, int]:
        if set(value) != set(HNCategory):
            raise ValueError("categories must list all hard-negative categories")
        return value


class Coverage(_Strict):
    """Root of ``seeds/coverage.yaml``: target families per (zone, meeting type, domain)."""

    variants_per_family: int = Field(ge=1)
    min_families_per_zone_domain: int = Field(ge=0)
    min_families_per_zone_meeting_type: int = Field(ge=0)
    segments: dict[Zone, dict[MeetingType, dict[Domain, int]]]
    hard_negatives: HardNegativeCoverage

    @model_validator(mode="after")
    def _complete(self) -> Coverage:
        for zone in Zone:
            if set(self.segments.get(zone, {})) != set(MeetingType):
                raise ValueError(f"segments.{zone.value} must list all meeting types")
            for meeting_type, cells in self.segments[zone].items():
                if not set(cells) <= set(Domain):
                    raise ValueError(
                        f"segments.{zone.value}.{meeting_type.value} lists unknown domains"
                    )
        return self

    def target(self, zone: Zone, meeting_type: MeetingType, domain: Domain) -> int:
        """Target number of families for one cell (0 for a domain the file does not list)."""
        return self.segments[zone][meeting_type].get(domain, 0)

    def zone_total(self, zone: Zone) -> int:
        """Target number of families for a zone."""
        return sum(sum(cells.values()) for cells in self.segments[zone].values())


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path, seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"circular 'extends' at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = data.pop("extends", None)
    if parent is None:
        return data
    return _deep_merge(_load_yaml(path.parent / parent, seen | {path}), data)


def load_config(path: Path = DEFAULT_CONFIG) -> GenerationConfig:
    """Load a generation config, resolving ``extends``."""
    return GenerationConfig.model_validate(_load_yaml(path))


def load_coverage(path: Path = DEFAULT_COVERAGE) -> Coverage:
    """Load the seed coverage targets."""
    return Coverage.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
