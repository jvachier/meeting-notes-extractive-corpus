"""v1.1 (PLAN §9c): in-place relabelling, the addendum seeds and the addendum pools."""

import json
import re

import pytest

from generation.bank import DEFAULT_SEEDS, load_bank
from generation.config import DEFAULT_CONFIG, load_config
from generation.generate_addendum import addendum_pools, load_addendum
from generation.schema import Domain, MeetingType, SeedSegment, Zone
from processing.build_v1_1 import edited_positions
from processing.relabel_from_seeds import RelabelError, relabel_rows

FIRST_PERSON = re.compile(
    r"\b(i|we)(?:'ll| will| can| am going to|'m going to)\b|leave it with me|consider it done", re.I
)


def _segment(labels):
    return SeedSegment.model_validate(
        {
            "segment_id": "act-quick_sync-eng-901-v1",
            "family_id": "act-quick_sync-eng-901",
            "primary_zone": "action_item",
            "meeting_types": ["quick_sync"],
            "domain": "engineering",
            "hard_negative": False,
            "hn_category": None,
            "confidence": 1.0,
            "n_speakers": 2,
            "sentences": [
                {"speaker": s, "text": t, "importance": i, "zone": z}
                for s, (t, i, z) in zip(("S1", "S2", "S1"), labels, strict=True)
            ],
        }
    )


def _rows(labels):
    sid = "act-quick_sync-eng-901-v1"
    rows = [
        {"note_id": "m", "sentence_id": k, "segment_id": sid, "importance": i, "zone": z}
        for k, (_, i, z) in enumerate(labels)
    ]
    filler = {
        "note_id": "m",
        "sentence_id": 9,
        "segment_id": "fill-001",
        "importance": 0,
        "zone": "discussion",
    }
    return [rows[0], filler, *rows[1:]]


OLD = [
    ("It broke.", 0, "discussion"),
    ("I'll fix it.", 0, "action_item"),
    ("Thanks.", 0, "action_item"),
]
NEW = [OLD[0], ("I'll fix it.", 1, "action_item"), OLD[2]]


def test_relabel_rows_changes_only_edited_seed_sentences():
    rows, n = relabel_rows(_rows(OLD), [_segment(NEW)], {("act-quick_sync-eng-901-v1", 1)})
    assert n == 1
    assert [r["importance"] for r in sorted(rows, key=lambda r: r["sentence_id"])] == [0, 1, 0, 0]


def test_relabel_rows_refuses_unlisted_or_misaligned_changes():
    with pytest.raises(RelabelError, match="not edited"):
        relabel_rows(_rows(OLD), [_segment(NEW)], set())
    with pytest.raises(RelabelError, match="does not match"):
        relabel_rows(_rows(OLD)[:-1], [_segment(NEW)], {("act-quick_sync-eng-901-v1", 1)})


def test_edit_list_matches_the_seeds():
    bank = {s.segment_id: s for s in load_bank().all_segments}
    edits = json.loads((DEFAULT_SEEDS / "addendum" / "relabelled.json").read_text())["edits"]
    assert len(edits) == len(edited_positions()) == 18
    for e in edits:
        sentence = bank[e["segment_id"]].sentences[e["index"]]
        assert sentence.text == e["text"] and sentence.importance == e["to"]["importance"] == 1


def test_addendum_seeds_teach_first_person_tasks():
    addendum = load_addendum()
    assert len(addendum) >= 97
    tasks = [
        s
        for seg in addendum
        if seg.domain != Domain.PERSONAL and seg.primary_zone == Zone.ACTION_ITEM
        for s in seg.sentences
        if s.importance == 1 and s.zone == Zone.ACTION_ITEM
    ]
    assert (
        len(tasks) >= 40
        and sum(bool(FIRST_PERSON.search(s.text)) for s in tasks) / len(tasks) > 0.9
    )
    # not every "I'll ..." is a task, so the model cannot learn a keyword
    assert any(
        FIRST_PERSON.search(s.text) and s.importance == 0 for seg in addendum for s in seg.sentences
    )
    personal = [s for s in addendum if s.domain == Domain.PERSONAL]
    assert {s.primary_zone for s in personal} == set(Zone)
    assert all(s.meeting_types == [MeetingType.QUICK_SYNC] for s in personal)
    v10_families = {s.family_id for s in load_bank().all_segments}
    assert not {s.family_id for s in addendum} & v10_families


def test_addendum_pools_keep_business_and_personal_apart():
    cfg = load_config(DEFAULT_CONFIG)
    pools, personal = addendum_pools(load_bank(), load_addendum(), cfg, cfg.seed)
    for pool in pools.values():
        assert all(s.domain != Domain.PERSONAL for s in pool.segments)
        new = {s.family_id for s in pool.segments if s.family_id.split("-")[-1] > "100"}
        assert all(
            s.primary_zone not in (Zone.DECISION, Zone.ACTION_ITEM) or s.family_id in new
            for s in pool.segments
        )
    assert personal.segments and all(s.domain == Domain.PERSONAL for s in personal.segments)
