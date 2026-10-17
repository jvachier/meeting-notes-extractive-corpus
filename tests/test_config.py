"""Tests for generation.config and the committed config / coverage files."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from generation.config import REPO_ROOT, GenerationConfig, load_config
from generation.schema import HNCategory, MeetingType, Zone


def test_real_config_matches_plan(cfg):
    assert cfg.meeting_types[MeetingType.QUICK_SYNC].words == (150, 300)
    assert cfg.meeting_types[MeetingType.PROJECT_REVIEW].participants == (4, 7)
    assert cfg.volumes.synthetic_notes_per_type * len(MeetingType) == 1250
    assert cfg.volumes.hard_negative_notes_per_type * len(MeetingType) == 250
    assert cfg.seeds.segment_sentences == (2, 12)
    assert cfg.seeds.hard_negative_confidence == (0.6, 0.8)


def test_tiny_config_extends_the_full_one(cfg):
    tiny = load_config(REPO_ROOT / "configs" / "generation_tiny.yaml")
    assert tiny.seed == cfg.seed
    assert tiny.volumes.synthetic_notes_per_type < cfg.volumes.synthetic_notes_per_type
    assert tiny.meeting_types == cfg.meeting_types
    assert tiny.composition.p_filler == cfg.composition.p_filler


def test_coverage_totals_match_plan(coverage):
    totals = {zone: coverage.zone_total(zone) for zone in Zone}
    assert totals == {
        Zone.CONTEXT: 60,
        Zone.DISCUSSION: 240,
        Zone.DECISION: 120,
        Zone.ACTION_ITEM: 120,
        Zone.CLOSING: 40,
    }
    assert coverage.variants_per_family == 2
    hn = coverage.hard_negatives
    assert sum(hn.categories.values()) == 150 and hn.variants_per_family == 2
    assert hn.categories[HNCategory.QUESTION_NOT_COMMITMENT] == 15


def test_coverage_targets_meet_the_minimum_rules(coverage):
    for zone in Zone:
        for meeting_type, cells in coverage.segments[zone].items():
            assert sum(cells.values()) >= coverage.min_families_per_zone_meeting_type, meeting_type
            assert min(cells.values()) >= coverage.min_families_per_zone_domain


def _raw_config() -> dict:
    return yaml.safe_load((REPO_ROOT / "configs" / "generation.yaml").read_text())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["meeting_types"]["standup"].update(words=[450, 300]),
        lambda c: c["meeting_types"]["standup"]["zone_weights"].update(decision=0.5),
        lambda c: c["meeting_types"].pop("retrospective"),
        lambda c: c["composition"].update(p_filler=1.5),
        lambda c: c["splits"]["ratios"].update(test=0.2),
        lambda c: c["domains"].pop(),
        lambda c: c["lexicons"]["min_values"].pop("PERSON"),
        lambda c: c.update(unexpected=1),
    ],
)
def test_invalid_configs_raise(mutate):
    raw = _raw_config()
    mutate(raw)
    with pytest.raises(ValidationError):
        GenerationConfig.model_validate(raw)


def test_circular_extends_raises(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ValueError, match="circular"):
        load_config(tmp_path / "a.yaml")
