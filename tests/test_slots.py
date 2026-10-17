"""Tests for generation.slots."""

from __future__ import annotations

import random

import pytest

from generation.slots import (
    SlotError,
    SlotFiller,
    UnknownSlotError,
    load_lexicons,
    parse_slots,
)

LEXICONS = {
    "PROJECT": ["Bluefin", "Kestrel", "Marigold"],
    "DATE_REL": ["next Tuesday", "end of sprint"],
    "PERSON": ["Amara", "Bo", "Chidi", "Dana"],
}


def test_parse_slots_finds_names_roles_and_offsets():
    text = "{S2|owner} will ship {PROJECT} by {DATE_REL|deadline}."
    refs = parse_slots(text)
    assert [(r.name, r.role) for r in refs] == [
        ("S2", "owner"),
        ("PROJECT", None),
        ("DATE_REL", "deadline"),
    ]
    assert text[refs[1].start : refs[1].end] == "{PROJECT}"
    assert refs[0].is_speaker and refs[0].speaker_index == 2
    assert not refs[1].is_speaker and refs[1].speaker_index is None


def test_plain_text_has_no_slots():
    assert parse_slots("Works for me.") == []


@pytest.mark.parametrize(
    "text",
    [
        "We use {FOO}.",
        "{project} is late.",
        "{S0} said so.",
        "{PROJECT|owner} owns it.",
        "{S1|boss} will do it.",
        "{PERSON|deadline} again.",
    ],
)
def test_unknown_slot_or_role_raises(text):
    with pytest.raises(UnknownSlotError):
        parse_slots(text)


@pytest.mark.parametrize("text", ["A { stray brace.", "Closing } only.", "{{PROJECT}}"])
def test_stray_braces_raise(text):
    with pytest.raises(SlotError):
        parse_slots(text)


def test_owner_role_allowed_on_person_team_and_speakers():
    for text in ["{PERSON|owner} does it.", "{TEAM|owner} does it.", "{S3|owner} does it."]:
        assert parse_slots(text)[0].role == "owner"


def test_fill_is_consistent_within_a_meeting():
    filler = SlotFiller(LEXICONS, random.Random(0))
    first = filler.fill("{PROJECT} slips to {DATE_REL}.", {})
    second = filler.fill("Back to {PROJECT}, still {DATE_REL}?", {})
    project, date = filler.values["PROJECT"], filler.values["DATE_REL"]
    assert first.text == f"{project} slips to {date}."
    assert second.text == f"Back to {project}, still {date}?"


def test_fill_maps_speakers_and_reports_role_spans():
    filler = SlotFiller(LEXICONS, random.Random(1))
    out = filler.fill("{S2|owner} sends it by {DATE_REL|deadline}.", {"S1": "Zoe", "S2": "Yusuf"})
    assert out.text.startswith("Yusuf sends it by ")
    owner, deadline = out.spans
    assert out.text[owner["start"] : owner["end"]] == "Yusuf" and owner["role"] == "owner"
    assert out.text[deadline["start"] : deadline["end"]] == filler.values["DATE_REL"]
    assert deadline["role"] == "deadline"


def test_fill_is_deterministic_for_a_seed():
    texts = ["{PROJECT} and {PERSON} by {DATE_REL}."]
    a = [SlotFiller(LEXICONS, random.Random(7)).fill(t, {}).text for t in texts]
    b = [SlotFiller(LEXICONS, random.Random(7)).fill(t, {}).text for t in texts]
    assert a == b


def test_person_never_takes_a_reserved_participant_name():
    for seed in range(20):
        filler = SlotFiller(LEXICONS, random.Random(seed), reserved=["Amara", "Bo", "Chidi"])
        assert filler.value("PERSON") == "Dana"


def test_unmapped_speaker_and_missing_lexicon_raise():
    filler = SlotFiller(LEXICONS, random.Random(0))
    with pytest.raises(SlotError):
        filler.fill("{S3} agrees.", {"S1": "Zoe"})
    with pytest.raises(SlotError):
        filler.fill("{MONEY} is the budget.", {})
    with pytest.raises(UnknownSlotError):
        filler.value("NOT_A_SLOT")


def test_load_lexicons(tmp_path):
    (tmp_path / "project.txt").write_text("# invented\nBluefin\n\n  Kestrel  \n", encoding="utf-8")
    assert load_lexicons(tmp_path) == {"PROJECT": ["Bluefin", "Kestrel"]}
    (tmp_path / "planet.txt").write_text("Mars\n", encoding="utf-8")
    with pytest.raises(UnknownSlotError):
        load_lexicons(tmp_path)


def test_load_lexicons_rejects_braces(tmp_path):
    (tmp_path / "team.txt").write_text("the {bad} team\n", encoding="utf-8")
    with pytest.raises(SlotError):
        load_lexicons(tmp_path)


def test_indexed_slots_parse_and_get_distinct_values():
    refs = parse_slots("{PROJECT} or {PROJECT_2}, {DATE_REL_2|deadline}.")
    assert [r.base for r in refs] == ["PROJECT", "PROJECT", "DATE_REL"]
    for seed in range(20):
        filler = SlotFiller(LEXICONS, random.Random(seed))
        out = filler.fill("{PROJECT_2} beats {PROJECT}; ask {PERSON} and {PERSON_2}.", {})
        values = filler.values
        assert values["PROJECT"] != values["PROJECT_2"]
        assert values["PERSON"] != values["PERSON_2"]
        assert out.text.startswith(values["PROJECT_2"])


@pytest.mark.parametrize("text", ["{PROJECT_1}.", "{PROJECT_10}.", "{FOO_2}.", "{S2_2}."])
def test_bad_indexed_slots_raise(text):
    with pytest.raises(UnknownSlotError):
        parse_slots(text)


def test_segment_scoped_slots_are_redrawn_per_segment_but_entities_are_not():
    lex = {"PROJECT": ["Bluefin", "Kestrel"], "PERCENT": [f"{i} percent" for i in range(50)]}
    filler = SlotFiller(lex, random.Random(3), segment_scoped=["PERCENT"])
    first = filler.fill("{PROJECT} is at {PERCENT}; yes, {PERCENT}.", {}).text
    project = filler.values["PROJECT"]
    percents = set()
    for _ in range(10):
        filler.new_segment()
        filler.fill("{PERCENT} for {PROJECT}.", {})
        percents.add(filler.values["PERCENT"])
        assert filler.values["PROJECT"] == project
    assert len(percents) > 1
    assert first.count(first.split(" is at ")[1].split(";")[0]) == 2  # same value within a segment
