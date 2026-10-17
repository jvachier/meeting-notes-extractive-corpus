"""The seeding briefs and their examples must stay valid against the schema."""

from __future__ import annotations

import json

import pytest

from generation.config import REPO_ROOT
from generation.schema import MeetingType, SeedSentence

PROMPTS = REPO_ROOT / "generation" / "prompts"


@pytest.mark.parametrize("meeting_type", list(MeetingType))
def test_every_meeting_type_has_a_brief_with_valid_examples(meeting_type):
    lines = (PROMPTS / f"{meeting_type.value}.txt").read_text(encoding="utf-8").splitlines()
    examples = [json.loads(line) for line in lines if line.startswith('{"speaker"')]
    assert examples, "each brief should show an example segment"
    for example in examples:
        SeedSentence.model_validate(example)
