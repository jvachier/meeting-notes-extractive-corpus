"""Pydantic models for seeds and output rows (PLAN §2).

Seed models validate what Claude Code writes in ``seeds/``. Output models describe one row of
the final ``sentences`` and ``notes`` Parquet files. Checks that depend on configurable ranges
(sentence counts, confidence bands) live in ``processing.validate_seeds``, not here.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from generation.slots import ROLE_DEADLINE, ROLE_OWNER, SPEAKER_RE, parse_slots


class Zone(StrEnum):
    """Sentence zone (PLAN §0)."""

    CONTEXT = "context"
    DISCUSSION = "discussion"
    DECISION = "decision"
    ACTION_ITEM = "action_item"
    CLOSING = "closing"


class MeetingType(StrEnum):
    """Synthetic meeting types (PLAN §3.1)."""

    QUICK_SYNC = "quick_sync"
    STANDUP = "standup"
    PROJECT_REVIEW = "project_review"
    STAKEHOLDER_UPDATE = "stakeholder_update"
    RETROSPECTIVE = "retrospective"


class Domain(StrEnum):
    """Synthetic domains (PLAN §3.3)."""

    ENGINEERING = "engineering"
    PRODUCT = "product"
    SALES = "sales"
    MARKETING = "marketing"
    CUSTOMER_SUPPORT = "customer_support"
    HR_PEOPLE = "hr_people"
    FINANCE_OPS = "finance_ops"
    RESEARCH = "research"


class HNCategory(StrEnum):
    """Hard-negative categories (PLAN §3.5)."""

    REJECTED_PROPOSAL = "rejected_proposal"
    TENTATIVE_IDEA = "tentative_idea"
    OWNERLESS_TASK = "ownerless_task"
    REVERSED_DECISION = "reversed_decision"
    REPORTED_ELSEWHERE = "reported_elsewhere"
    HYPOTHETICAL = "hypothetical"
    QUESTION_NOT_COMMITMENT = "question_not_commitment"


class Source(StrEnum):
    """Origin of an output row (PLAN §2.2)."""

    SYNTHETIC = "synthetic"
    SYNTHETIC_HN = "synthetic_hn"
    AMI = "ami"


class Split(StrEnum):
    """Dataset split."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class SpanRole(StrEnum):
    """Role of a filled slot span in an action-item sentence (PLAN §2.1)."""

    OWNER = ROLE_OWNER
    DEADLINE = ROLE_DEADLINE


class FillerTag(StrEnum):
    """Placement restriction for a filler (PLAN §3.6)."""

    OPENING = "opening"
    CLOSING_CHATTER = "closing_chatter"


AMI_MEETING_TYPE = "ami_design"
AMI_DOMAIN = "product_design"
AMI_ZONES: frozenset[Zone] = frozenset({Zone.DISCUSSION, Zone.DECISION, Zone.ACTION_ITEM})

#: Family-id prefix per zone, e.g. ``dec-project_review-eng-003``.
ZONE_PREFIX: dict[Zone, str] = {
    Zone.CONTEXT: "ctx",
    Zone.DISCUSSION: "disc",
    Zone.DECISION: "dec",
    Zone.ACTION_ITEM: "act",
    Zone.CLOSING: "clo",
}

#: Domain abbreviation used in family ids.
DOMAIN_ABBR: dict[Domain, str] = {
    Domain.ENGINEERING: "eng",
    Domain.PRODUCT: "prod",
    Domain.SALES: "sales",
    Domain.MARKETING: "mkt",
    Domain.CUSTOMER_SUPPORT: "sup",
    Domain.HR_PEOPLE: "hr",
    Domain.FINANCE_OPS: "fin",
    Domain.RESEARCH: "res",
}

_VARIANT_RE = re.compile(r"^(?P<family>.+)-v(?P<n>[1-9]\d*)$")
_FILLER_ID_RE = re.compile(r"^fill-\d{3,}$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _check_clean_text(text: str) -> str:
    if text != text.strip():
        raise ValueError("text has leading or trailing whitespace")
    if "\n" in text or "\t" in text:
        raise ValueError("text contains a newline or tab")
    return text


def family_id_pattern(
    primary_zone: Zone,
    domain: Domain,
    hn_category: HNCategory | None,
) -> re.Pattern[str]:
    """Regex a family id must match, given the segment's zone, domain and HN category.

    Normal families: ``<zone_prefix>-<meeting_type>-<domain_abbr>-<nnn>``.
    Hard-negative families: ``hn-<category>-<nnn>``.
    """
    if hn_category is not None:
        return re.compile(rf"^hn-{hn_category.value}-(?P<num>\d{{3}})$")
    types = "|".join(t.value for t in MeetingType)
    return re.compile(
        rf"^{ZONE_PREFIX[primary_zone]}-(?P<meeting_type>{types})-{DOMAIN_ABBR[domain]}"
        rf"-(?P<num>\d{{3}})$"
    )


class SeedSentence(_Strict):
    """One sentence of a seed segment, with its own labels."""

    speaker: str
    text: str = Field(min_length=1)
    importance: Literal[0, 1]
    zone: Zone

    @field_validator("speaker")
    @classmethod
    def _speaker_id(cls, value: str) -> str:
        if not SPEAKER_RE.fullmatch(value):
            raise ValueError(f"speaker must be S1, S2, …; got {value!r}")
        return value

    @field_validator("text")
    @classmethod
    def _text(cls, value: str) -> str:
        _check_clean_text(value)
        parse_slots(value)  # SlotError is a ValueError → reported as a validation error
        return value


class SeedSegment(_Strict):
    """A short labelled exchange with one main purpose (PLAN §2.1)."""

    segment_id: str
    family_id: str
    primary_zone: Zone
    meeting_types: list[MeetingType] = Field(min_length=1)
    domain: Domain
    hard_negative: bool
    hn_category: HNCategory | None
    confidence: float = Field(gt=0.0, le=1.0)
    n_speakers: int = Field(ge=1)
    sentences: list[SeedSentence] = Field(min_length=1)

    @property
    def variant(self) -> int:
        """Variant number ``n`` from the ``-v<n>`` suffix of ``segment_id``."""
        match = _VARIANT_RE.match(self.segment_id)
        assert match is not None  # guaranteed by validation
        return int(match.group("n"))

    @property
    def cell_meeting_type(self) -> MeetingType | None:
        """Meeting type named in the family id (the coverage cell), None for hard negatives."""
        match = family_id_pattern(self.primary_zone, self.domain, self.hn_category).match(
            self.family_id
        )
        if match is None or self.hn_category is not None:
            return None
        return MeetingType(match.group("meeting_type"))

    @model_validator(mode="after")
    def _consistency(self) -> SeedSegment:
        variant = _VARIANT_RE.match(self.segment_id)
        if variant is None or variant.group("family") != self.family_id:
            raise ValueError(f"segment_id must be '<family_id>-v<n>'; got {self.segment_id!r}")

        if self.hard_negative != (self.hn_category is not None):
            raise ValueError("hn_category must be set if and only if hard_negative is true")

        match = family_id_pattern(self.primary_zone, self.domain, self.hn_category).match(
            self.family_id
        )
        if match is None:
            raise ValueError(
                f"family_id {self.family_id!r} does not match zone/domain/category naming"
            )
        if self.hn_category is None and match.group("meeting_type") not in self.meeting_types:
            raise ValueError("the meeting type in family_id must be listed in meeting_types")

        if len(set(self.meeting_types)) != len(self.meeting_types):
            raise ValueError("meeting_types has duplicates")

        # Every participant S1..Sn speaks or is addressed/mentioned as {Sk}; nobody else exists.
        speakers = {s.speaker for s in self.sentences}
        mentioned = {
            ref.name for s in self.sentences for ref in parse_slots(s.text) if ref.is_speaker
        }
        expected = {f"S{i}" for i in range(1, self.n_speakers + 1)}
        if not speakers <= expected or speakers | mentioned != expected:
            raise ValueError(
                f"speakers {sorted(speakers)} and mentions {sorted(mentioned)} must cover "
                f"exactly S1..S{self.n_speakers}"
            )

        for sentence in self.sentences:
            for ref in parse_slots(sentence.text):
                index = ref.speaker_index
                if index is not None and index > self.n_speakers:
                    raise ValueError(f"{{{ref.name}}} exceeds n_speakers={self.n_speakers}")
                if ref.role is not None and not (
                    sentence.zone == Zone.ACTION_ITEM and sentence.importance == 1
                ):
                    raise ValueError(
                        "role slots are only allowed in action_item sentences with importance 1"
                    )

        if self.primary_zone not in {s.zone for s in self.sentences}:
            raise ValueError(f"no sentence has the primary zone {self.primary_zone.value!r}")
        return self


class Filler(_Strict):
    """A one-sentence backchannel or small-talk line (PLAN §3.6). Never carries a slot."""

    filler_id: str
    text: str = Field(min_length=1)
    importance: Literal[0] = 0
    zone: Literal[Zone.DISCUSSION] = Zone.DISCUSSION
    tag: FillerTag | None = None

    @field_validator("filler_id")
    @classmethod
    def _filler_id(cls, value: str) -> str:
        if not _FILLER_ID_RE.fullmatch(value):
            raise ValueError(f"filler_id must look like 'fill-001'; got {value!r}")
        return value

    @field_validator("text")
    @classmethod
    def _text(cls, value: str) -> str:
        _check_clean_text(value)
        if parse_slots(value):
            raise ValueError("fillers must not contain slots")
        return value


class Span(_Strict):
    """Character span of a filled role slot in ``sentence_text``."""

    start: int = Field(ge=0)
    end: int
    role: SpanRole

    @model_validator(mode="after")
    def _order(self) -> Span:
        if self.end <= self.start:
            raise ValueError("span end must be greater than start")
        return self


class SentenceRow(_Strict):
    """One row of the final ``sentences`` Parquet (PLAN §2.2)."""

    note_id: str
    source: Source
    meeting_type: str
    domain: str
    split: Split
    sentence_id: int = Field(ge=0)
    speaker: str = Field(min_length=1)
    sentence_text: str = Field(min_length=1)
    sentence_text_asr: str
    importance: Literal[0, 1]
    confidence: float = Field(gt=0.0, le=1.0)
    zone: Zone
    segment_id: str | None
    family_id: str | None
    hard_negative: bool
    token_count: int = Field(ge=0)
    spans: list[Span] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistency(self) -> SentenceRow:
        if self.token_count != len(self.sentence_text.split()):
            raise ValueError("token_count must be the whitespace word count of sentence_text")
        for span in self.spans:
            if span.end > len(self.sentence_text):
                raise ValueError("span exceeds sentence_text")
        if self.source == Source.AMI:
            if self.meeting_type != AMI_MEETING_TYPE or self.domain != AMI_DOMAIN:
                raise ValueError("AMI rows use meeting_type ami_design, domain product_design")
            if self.segment_id is not None or self.family_id is not None:
                raise ValueError("AMI rows have no segment_id / family_id")
            if self.spans or self.hard_negative:
                raise ValueError("AMI rows have no spans and are never hard negatives")
            if self.zone not in AMI_ZONES:
                raise ValueError("AMI rows use discussion, decision or action_item zones only")
        else:
            if self.meeting_type not in set(MeetingType) or self.domain not in set(Domain):
                raise ValueError("synthetic rows need a synthetic meeting_type and domain")
            if self.spans and self.zone != Zone.ACTION_ITEM:
                raise ValueError("spans only occur on action_item sentences")
        return self


class NoteRow(_Strict):
    """One row of the ``notes`` Parquet: one meeting (PLAN §2.2)."""

    note_id: str
    source: Source
    meeting_type: str
    split: Split
    n_sentences: int = Field(ge=1)
    n_words: int = Field(ge=1)
    n_participants: int = Field(ge=1)
    summary_abstract: str | None = None
    summary_decisions: str | None = None
    summary_actions: str | None = None
    summary_problems: str | None = None
