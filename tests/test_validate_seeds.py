"""Tests for processing.validate_seeds on small temporary seed trees."""

from __future__ import annotations

from pathlib import Path

import pytest

from generation.config import GenerationConfig
from processing import validate_seeds as vs
from tests.conftest import (
    ACTION_SEGMENT,
    DECISION_SEGMENT,
    DECISION_VARIANT_2,
    HN_SEGMENT,
    segment,
    write_jsonl,
)

DECISION_FILE = Path("segments/decision/project_review__engineering__b01.jsonl")
ACTION_FILE = Path("segments/action_item/quick_sync__sales__b01.jsonl")
HN_FILE = Path("hard_negatives/tentative_idea__b01.jsonl")


@pytest.fixture
def small_cfg(cfg) -> GenerationConfig:
    """Real config with lexicon minimums lowered to 1 so fixtures stay small."""
    raw = cfg.model_dump(mode="json")
    raw["lexicons"]["min_values"] = dict.fromkeys(raw["lexicons"]["min_values"], 1)
    return GenerationConfig.model_validate(raw)


def _action_v2() -> dict:
    return segment(
        ACTION_SEGMENT,
        segment_id="act-quick_sync-sales-001-v2",
        sentences=[
            {
                "speaker": "S2",
                "text": "I can take the quote for {CLIENT}.",
                "importance": 0,
                "zone": "discussion",
            },
            {"speaker": "S1", "text": "Great, thanks.", "importance": 0, "zone": "discussion"},
            {
                "speaker": "S2",
                "text": "{S2|owner} gets it out {DATE_REL|deadline}.",
                "importance": 1,
                "zone": "action_item",
            },
        ],
    )


def _hn_v2() -> dict:
    return segment(
        HN_SEGMENT,
        segment_id="hn-tentative_idea-001-v2",
        sentences=[
            {
                "speaker": "S2",
                "text": "At some point caching might be worth a look.",
                "importance": 0,
                "zone": "discussion",
            },
            {
                "speaker": "S1",
                "text": "Possibly, not now though.",
                "importance": 0,
                "zone": "discussion",
            },
        ],
    )


@pytest.fixture
def tree(tmp_path: Path) -> tuple[Path, Path]:
    """A valid seeds tree and lexicon folder."""
    seeds, lexicons = tmp_path / "seeds", tmp_path / "lexicons"
    write_jsonl(seeds / DECISION_FILE, [DECISION_SEGMENT, DECISION_VARIANT_2])
    write_jsonl(seeds / ACTION_FILE, [ACTION_SEGMENT, _action_v2()])
    write_jsonl(seeds / HN_FILE, [HN_SEGMENT, _hn_v2()])
    write_jsonl(
        seeds / "fillers/fillers.jsonl",
        [
            {"filler_id": "fill-001", "text": "Yeah."},
            {"filler_id": "fill-002", "text": "Can you hear me?", "tag": "opening"},
        ],
    )
    lexicons.mkdir()
    for slot, values in {
        "project": "Bluefin",
        "date_rel": "next Tuesday",
        "client": "Norvale",
    }.items():
        (lexicons / f"{slot}.txt").write_text(values + "\n", encoding="utf-8")
    return seeds, lexicons


def _run(tree, small_cfg, coverage, **kwargs) -> vs.Report:
    seeds, lexicons = tree
    return vs.validate(seeds, lexicons, small_cfg, coverage, **kwargs)


def test_valid_tree_passes(tree, small_cfg, coverage):
    report = _run(tree, small_cfg, coverage)
    assert report.ok, report.errors
    assert report.stats["segments"] == 4
    assert report.stats["hard_negative_segments"] == 2
    assert report.stats["families_by_zone"]["decision"]["families"] == 1
    assert report.stats["hard_negative_families"]["tentative_idea"]["families"] == 1
    assert report.stats["fillers"]["total"] == 2


def test_empty_tree_passes_with_warnings(tmp_path, small_cfg, coverage):
    report = vs.validate(tmp_path, tmp_path / "lex", small_cfg, coverage)
    assert report.ok
    assert any("no fillers" in w for w in report.warnings)
    assert any("no lexicons" in w for w in report.warnings)


def test_strict_coverage_fails_on_shortfall(tree, small_cfg, coverage):
    report = _run(tree, small_cfg, coverage, strict_coverage=True)
    assert any(e.startswith("coverage shortfall") for e in report.errors)


def _errors_after(tree, small_cfg, coverage, rel: Path, rows: list[dict]) -> list[str]:
    write_jsonl(tree[0] / rel, rows)
    return _run(tree, small_cfg, coverage).errors


def test_schema_error_is_reported_with_location(tree, small_cfg, coverage):
    bad = segment(DECISION_SEGMENT, n_speakers=5)
    errors = _errors_after(tree, small_cfg, coverage, DECISION_FILE, [bad, DECISION_VARIANT_2])
    assert any("project_review__engineering__b01.jsonl:1" in e for e in errors)


def test_invalid_json_line(tree, small_cfg, coverage):
    path = tree[0] / DECISION_FILE
    path.write_text(path.read_text() + "{not json\n")
    assert any("invalid JSON" in e for e in _run(tree, small_cfg, coverage).errors)


def test_segment_in_wrong_file(tree, small_cfg, coverage):
    rel = Path("segments/decision/standup__engineering__b01.jsonl")
    write_jsonl(tree[0] / rel, [segment(DECISION_SEGMENT)])
    (tree[0] / DECISION_FILE).unlink()
    errors = _run(tree, small_cfg, coverage).errors
    assert any("do not match its path" in e for e in errors)


def test_bad_file_name(tree, small_cfg, coverage):
    (tree[0] / DECISION_FILE).rename(tree[0] / "segments/decision/batch1.jsonl")
    assert any("file name must be" in e for e in _run(tree, small_cfg, coverage).errors)


def test_missing_variant(tree, small_cfg, coverage):
    errors = _errors_after(tree, small_cfg, coverage, DECISION_FILE, [DECISION_SEGMENT])
    assert any("expected v1..v2" in e for e in errors)


def test_identical_variants(tree, small_cfg, coverage):
    twin = segment(DECISION_SEGMENT, segment_id="dec-project_review-eng-003-v2")
    errors = _errors_after(tree, small_cfg, coverage, DECISION_FILE, [DECISION_SEGMENT, twin])
    assert any("identical" in e for e in errors)


def test_duplicate_segment_text_across_families(tree, small_cfg, coverage):
    other = segment(
        DECISION_SEGMENT,
        family_id="dec-project_review-eng-004",
        segment_id="dec-project_review-eng-004-v1",
    )
    other_v2 = segment(
        DECISION_VARIANT_2, family_id=other["family_id"], segment_id="dec-project_review-eng-004-v2"
    )
    rows = [DECISION_SEGMENT, DECISION_VARIANT_2, other, other_v2]
    errors = _errors_after(tree, small_cfg, coverage, DECISION_FILE, rows)
    assert any("duplicates the text" in e for e in errors)


def test_too_many_sentences(tree, small_cfg, coverage):
    long = segment(DECISION_SEGMENT)
    filler = {"speaker": "S2", "text": "Right.", "importance": 0, "zone": "discussion"}
    long["sentences"] = long["sentences"] + [filler] * 10
    errors = _errors_after(tree, small_cfg, coverage, DECISION_FILE, [long, DECISION_VARIANT_2])
    assert any("14 sentences" in e for e in errors)


def test_decision_segment_without_important_decision(tree, small_cfg, coverage):
    weak = segment(DECISION_SEGMENT)
    weak["sentences"][2]["importance"] = 0
    errors = _errors_after(tree, small_cfg, coverage, DECISION_FILE, [weak, DECISION_VARIANT_2])
    assert any("no important decision" in e for e in errors)


def test_hard_negative_confidence_band(tree, small_cfg, coverage):
    errors = _errors_after(
        tree, small_cfg, coverage, HN_FILE, [segment(HN_SEGMENT, confidence=1.0), _hn_v2()]
    )
    assert any("confidence 1.0 outside [0.6, 0.8]" in e for e in errors)


def test_hard_negative_category_must_match_file(tree, small_cfg, coverage):
    rel = Path("hard_negatives/hypothetical__b01.jsonl")
    (tree[0] / HN_FILE).rename(tree[0] / rel)
    assert any("needs hn_category" in e for e in _run(tree, small_cfg, coverage).errors)


def test_hard_negative_outside_its_folder(tree, small_cfg, coverage):
    rel = Path("segments/discussion/standup__engineering__b01.jsonl")
    write_jsonl(tree[0] / rel, [HN_SEGMENT])
    assert any(
        "belong in seeds/hard_negatives" in e for e in _run(tree, small_cfg, coverage).errors
    )


def test_used_slot_without_lexicon(tree, small_cfg, coverage):
    (tree[1] / "client.txt").unlink()
    assert any("{CLIENT} is used" in e for e in _run(tree, small_cfg, coverage).errors)


def test_lexicon_minimum_and_duplicates(tree, cfg, coverage):
    (tree[1] / "project.txt").write_text("Bluefin\nBluefin\n")
    errors = vs.validate(tree[0], tree[1], cfg, coverage).errors
    assert any("lexicon PROJECT: 2 values, need ≥ 60" in e for e in errors)
    assert any("lexicon PROJECT: duplicate values" in e for e in errors)


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("team.txt", "platform team\n", "TEAM: values must be"),
        ("metric.txt", "the churn\n", "METRIC: values must be"),
        ("person.txt", "amara\n", "PERSON: values must be"),
        ("person.txt", "Mary Jane\n", "PERSON: values must be"),
        ("percent.txt", "12%\n", "PERCENT: values must be"),
        ("product.txt", "Bluefin\n", "appears in ['PRODUCT', 'PROJECT']"),
    ],
)
def test_lexicon_forms_and_cross_lexicon_duplicates(
    tree, small_cfg, coverage, filename, content, message
):
    (tree[1] / filename).write_text(content, encoding="utf-8")
    assert any(message in e for e in _run(tree, small_cfg, coverage).errors)


def test_person_name_that_is_a_seed_word_warns(tree, small_cfg, coverage):
    (tree[1] / "person.txt").write_text("Amara\nWorks\n", encoding="utf-8")
    report = _run(tree, small_cfg, coverage)
    assert any("also words in seed text: ['Works']" in w for w in report.warnings)


def test_duplicate_fillers(tree, small_cfg, coverage):
    rows = [{"filler_id": "fill-001", "text": "Yeah."}, {"filler_id": "fill-001", "text": "yeah"}]
    errors = _errors_after(tree, small_cfg, coverage, Path("fillers/fillers.jsonl"), rows)
    assert any("duplicate filler_id" in e for e in errors)
    assert any("duplicate filler text" in e for e in errors)


def test_action_item_without_owner_warns(tree, small_cfg, coverage):
    rows = [segment(ACTION_SEGMENT), _action_v2()]
    rows[0]["sentences"][1]["text"] = "I will send it over by {DATE_REL|deadline}."
    write_jsonl(tree[0] / ACTION_FILE, rows)
    report = _run(tree, small_cfg, coverage)
    assert report.ok
    assert any("no {…|owner} slot" in w for w in report.warnings)


def test_main_exit_codes(tree, small_cfg, coverage, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(vs, "load_config", lambda _path: small_cfg)
    args = ["--seeds-dir", str(tree[0]), "--lexicons-dir", str(tree[1])]
    assert vs.main(args) == 0
    assert vs.main([*args, "--strict-coverage"]) == 1
    assert "seed validation: FAILED" in capsys.readouterr().out


def test_repository_seeds_are_valid():
    """The committed seeds must always pass (coverage is not enforced here)."""
    assert vs.validate().ok
