"""Tests for generation.schema: seed models and output row models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from generation.schema import (
    Filler,
    MeetingType,
    NoteRow,
    SeedSegment,
    SentenceRow,
    Zone,
)
from tests.conftest import ACTION_SEGMENT, DECISION_SEGMENT, HN_SEGMENT, segment


def test_plan_example_segment_is_valid():
    seg = SeedSegment.model_validate(DECISION_SEGMENT)
    assert seg.variant == 1
    assert seg.cell_meeting_type == MeetingType.PROJECT_REVIEW
    assert seg.sentences[3].zone == Zone.DECISION and seg.sentences[3].importance == 0


def test_participant_may_be_mentioned_without_speaking():
    seg = segment(DECISION_SEGMENT, n_speakers=4)
    seg["sentences"][0]["text"] = "So {S4} says we either ship {PROJECT} now or wait."
    assert SeedSegment.model_validate(seg).n_speakers == 4


def test_action_and_hard_negative_segments_are_valid():
    assert SeedSegment.model_validate(ACTION_SEGMENT).primary_zone == Zone.ACTION_ITEM
    hn = SeedSegment.model_validate(HN_SEGMENT)
    assert hn.hard_negative and hn.cell_meeting_type is None


def _with_sentence(base, index, **changes):
    out = segment(base)
    out["sentences"][index].update(changes)
    return out


@pytest.mark.parametrize(
    "bad",
    [
        segment(DECISION_SEGMENT, segment_id="dec-project_review-eng-003"),  # no -v<n>
        segment(DECISION_SEGMENT, segment_id="dec-project_review-eng-004-v1"),  # wrong family
        segment(
            DECISION_SEGMENT,
            family_id="dec-project_review-eng-3",
            segment_id="dec-project_review-eng-3-v1",
        ),  # bad number
        segment(DECISION_SEGMENT, domain="sales"),  # abbr mismatch
        segment(DECISION_SEGMENT, primary_zone="action_item"),  # prefix mismatch
        segment(DECISION_SEGMENT, meeting_types=["standup"]),  # id type not listed
        segment(DECISION_SEGMENT, meeting_types=[]),
        segment(DECISION_SEGMENT, meeting_types=["project_review", "project_review"]),
        segment(DECISION_SEGMENT, meeting_types=["project_review", "board_meeting"]),
        segment(DECISION_SEGMENT, hard_negative=True),  # no category
        segment(DECISION_SEGMENT, hn_category="hypothetical"),  # category but not HN
        segment(DECISION_SEGMENT, n_speakers=4),  # S4 never speaks
        segment(DECISION_SEGMENT, n_speakers=2),  # S3 speaks
        segment(DECISION_SEGMENT, confidence=0.0),
        segment(DECISION_SEGMENT, confidence=1.2),
        segment(DECISION_SEGMENT, extra_field=1),
        _with_sentence(DECISION_SEGMENT, 0, speaker="A"),
        _with_sentence(DECISION_SEGMENT, 0, importance=2),
        _with_sentence(DECISION_SEGMENT, 0, zone="agenda"),
        _with_sentence(DECISION_SEGMENT, 0, text=" leading space."),
        _with_sentence(DECISION_SEGMENT, 0, text="Two\nlines."),
        _with_sentence(DECISION_SEGMENT, 0, text="Ask {S5} about it."),  # > n_speakers
        _with_sentence(DECISION_SEGMENT, 0, text="Ask {BOSS} about it."),  # unknown slot
        segment(
            DECISION_SEGMENT,
            primary_zone="closing",
            family_id="clo-project_review-eng-003",
            segment_id="clo-project_review-eng-003-v1",
        ),  # primary zone absent
        _with_sentence(DECISION_SEGMENT, 0, text="{S2|owner} will check it."),  # role off action
        _with_sentence(ACTION_SEGMENT, 1, importance=0),  # role on importance-0 sentence
        segment(HN_SEGMENT, family_id="hn-hypothetical-001", segment_id="hn-hypothetical-001-v1"),
    ],
)
def test_invalid_segments_raise(bad):
    with pytest.raises(ValidationError):
        SeedSegment.model_validate(bad)


def test_filler_model():
    filler = Filler.model_validate({"filler_id": "fill-001", "text": "Sorry, go ahead."})
    assert filler.importance == 0 and filler.zone == Zone.DISCUSSION and filler.tag is None
    assert (
        Filler.model_validate(
            {"filler_id": "fill-002", "text": "Can you hear me?", "tag": "opening"}
        ).tag
        == "opening"
    )
    for bad in [
        {"filler_id": "f1", "text": "Yeah."},
        {"filler_id": "fill-003", "text": "Yeah.", "importance": 1},
        {"filler_id": "fill-003", "text": "Yeah.", "zone": "decision"},
        {"filler_id": "fill-003", "text": "Hi {S1}."},
        {"filler_id": "fill-003", "text": "Yeah.", "tag": "middle"},
    ]:
        with pytest.raises(ValidationError):
            Filler.model_validate(bad)


SYNTHETIC_ROW = {
    "note_id": "syn-000001",
    "source": "synthetic",
    "meeting_type": "quick_sync",
    "domain": "sales",
    "split": "train",
    "sentence_id": 3,
    "speaker": "Yusuf",
    "sentence_text": "Yusuf will send it by next Tuesday.",
    "sentence_text_asr": "yusuf will send it by next tuesday",
    "importance": 1,
    "confidence": 1.0,
    "zone": "action_item",
    "segment_id": "act-quick_sync-sales-001-v1",
    "family_id": "act-quick_sync-sales-001",
    "hard_negative": False,
    "token_count": 7,
    "spans": [
        {"start": 0, "end": 5, "role": "owner"},
        {"start": 22, "end": 34, "role": "deadline"},
    ],
}

AMI_ROW = {
    **SYNTHETIC_ROW,
    "note_id": "ES2002a",
    "source": "ami",
    "meeting_type": "ami_design",
    "domain": "product_design",
    "speaker": "A",
    "zone": "decision",
    "confidence": 0.9,
    "segment_id": None,
    "family_id": None,
    "spans": [],
}


def test_output_rows_valid():
    SentenceRow.model_validate(SYNTHETIC_ROW)
    SentenceRow.model_validate(AMI_ROW)


@pytest.mark.parametrize(
    "bad",
    [
        {**SYNTHETIC_ROW, "token_count": 6},
        {**SYNTHETIC_ROW, "split": "dev"},
        {**SYNTHETIC_ROW, "meeting_type": "ami_design"},
        {**SYNTHETIC_ROW, "spans": [{"start": 30, "end": 99, "role": "owner"}]},
        {**SYNTHETIC_ROW, "spans": [{"start": 5, "end": 5, "role": "owner"}]},
        {**SYNTHETIC_ROW, "zone": "decision"},  # spans on a non-action sentence
        {**AMI_ROW, "zone": "context"},
        {**AMI_ROW, "segment_id": "x-v1"},
        {**AMI_ROW, "spans": [{"start": 0, "end": 5, "role": "owner"}], "zone": "action_item"},
        {**AMI_ROW, "meeting_type": "quick_sync"},
    ],
)
def test_invalid_output_rows_raise(bad):
    with pytest.raises(ValidationError):
        SentenceRow.model_validate(bad)


def test_note_row():
    row = NoteRow.model_validate(
        {
            "note_id": "ES2002a",
            "source": "ami",
            "meeting_type": "ami_design",
            "split": "test",
            "n_sentences": 400,
            "n_words": 5000,
            "n_participants": 4,
            "summary_decisions": "They chose a rubber case.",
        }
    )
    assert row.summary_abstract is None
    with pytest.raises(ValidationError):
        NoteRow.model_validate({**row.model_dump(), "n_sentences": 0})
