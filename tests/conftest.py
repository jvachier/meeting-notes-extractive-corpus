"""Shared fixtures: hand-made seed segments and a temporary seeds tree."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from generation.config import DEFAULT_CONFIG, DEFAULT_COVERAGE, load_config, load_coverage

DECISION_SEGMENT: dict[str, Any] = {
    "segment_id": "dec-project_review-eng-003-v1",
    "family_id": "dec-project_review-eng-003",
    "primary_zone": "decision",
    "meeting_types": ["project_review", "stakeholder_update"],
    "domain": "engineering",
    "hard_negative": False,
    "hn_category": None,
    "confidence": 1.0,
    "n_speakers": 3,
    "sentences": [
        {
            "speaker": "S1",
            "text": "So for {PROJECT} we either ship on {DATE_REL} or wait for the load tests.",
            "importance": 0,
            "zone": "discussion",
        },
        {
            "speaker": "S2",
            "text": "The load tests are the only thing blocking us, honestly.",
            "importance": 0,
            "zone": "discussion",
        },
        {
            "speaker": "S1",
            "text": "Okay, decision: we hold the release until the load tests pass.",
            "importance": 1,
            "zone": "decision",
        },
        {"speaker": "S3", "text": "Works for me.", "importance": 0, "zone": "decision"},
    ],
}

DECISION_VARIANT_2: dict[str, Any] = {
    **DECISION_SEGMENT,
    "segment_id": "dec-project_review-eng-003-v2",
    "sentences": [
        {
            "speaker": "S2",
            "text": "Honestly the only blocker for {PROJECT} right now is load testing.",
            "importance": 0,
            "zone": "discussion",
        },
        {
            "speaker": "S1",
            "text": "Then let's not ship {DATE_REL}; we wait until those tests are green.",
            "importance": 1,
            "zone": "decision",
        },
        {"speaker": "S3", "text": "Agreed.", "importance": 0, "zone": "decision"},
    ],
}

ACTION_SEGMENT: dict[str, Any] = {
    "segment_id": "act-quick_sync-sales-001-v1",
    "family_id": "act-quick_sync-sales-001",
    "primary_zone": "action_item",
    "meeting_types": ["quick_sync"],
    "domain": "sales",
    "hard_negative": False,
    "hn_category": None,
    "confidence": 1.0,
    "n_speakers": 2,
    "sentences": [
        {
            "speaker": "S1",
            "text": "Who can get the revised quote to {CLIENT}?",
            "importance": 0,
            "zone": "discussion",
        },
        {
            "speaker": "S2",
            "text": "{S2|owner} will send it by {DATE_REL|deadline}, I mean me.",
            "importance": 1,
            "zone": "action_item",
        },
    ],
}

HN_SEGMENT: dict[str, Any] = {
    "segment_id": "hn-tentative_idea-001-v1",
    "family_id": "hn-tentative_idea-001",
    "primary_zone": "discussion",
    "meeting_types": ["standup", "retrospective"],
    "domain": "engineering",
    "hard_negative": True,
    "hn_category": "tentative_idea",
    "confidence": 0.7,
    "n_speakers": 2,
    "sentences": [
        {
            "speaker": "S1",
            "text": "Maybe someone should look into caching at some point.",
            "importance": 0,
            "zone": "discussion",
        },
        {
            "speaker": "S2",
            "text": "Yeah, could be worth it.",
            "importance": 0,
            "zone": "discussion",
        },
    ],
}


def segment(base: dict[str, Any], **changes: Any) -> dict[str, Any]:
    """Deep copy of ``base`` with top-level ``changes`` applied."""
    out = copy.deepcopy(base)
    out.update(changes)
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write ``rows`` as JSON lines, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def cfg():
    """The real generation config."""
    return load_config(DEFAULT_CONFIG)


@pytest.fixture(scope="session")
def coverage():
    """The real coverage targets."""
    return load_coverage(DEFAULT_COVERAGE)
