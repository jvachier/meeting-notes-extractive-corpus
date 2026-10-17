"""Slot parsing and filling (PLAN §2.1, §3.2).

Seed text contains placeholders in braces:

- lexicon slots such as ``{PROJECT}`` or ``{DATE_REL}``, filled from ``generation/lexicons``;
- speaker references ``{S1}``, ``{S2}``…, filled with the name of the meeting participant the
  segment-local speaker is mapped to;
- an indexed lexicon slot such as ``{PROJECT_2}`` (index 2–9) for a second, different value
  of the same slot within one meeting;
- an optional role after a pipe, ``{S2|owner}`` or ``{DATE_REL|deadline}``, allowed only on
  action-item sentences. Filled role slots are reported as character spans.

Within one meeting each lexicon slot is drawn once and reused (``SlotFiller`` caches values),
and ``{PROJECT}``, ``{PROJECT_2}``… always get distinct values.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

SLOT_NAMES: frozenset[str] = frozenset(
    {
        "PERSON",
        "TEAM",
        "PROJECT",
        "PRODUCT",
        "CLIENT",
        "TOOL",
        "DATE_REL",
        "DATE_ABS",
        "NUMBER",
        "PERCENT",
        "MONEY",
        "METRIC",
    }
)

ROLE_OWNER = "owner"
ROLE_DEADLINE = "deadline"

#: Which slots may carry which role. Speaker references (``S<n>``) may be owners too.
ROLE_SLOTS: Mapping[str, frozenset[str]] = {
    ROLE_OWNER: frozenset({"PERSON", "TEAM"}),
    ROLE_DEADLINE: frozenset({"DATE_REL", "DATE_ABS"}),
}

SPEAKER_RE = re.compile(r"S[1-9]\d*")
_SLOT_RE = re.compile(r"\{([^{}|]*)(?:\|([^{}]*))?\}")
_INDEX_RE = re.compile(r"^(?P<base>[A-Z_]+)_(?P<index>[2-9])$")


def base_slot(name: str) -> str | None:
    """Lexicon slot behind ``name`` (``PROJECT_2`` → ``PROJECT``), or None if unknown."""
    if name in SLOT_NAMES:
        return name
    match = _INDEX_RE.match(name)
    if match is not None and match.group("base") in SLOT_NAMES:
        return match.group("base")
    return None


class SlotError(ValueError):
    """Malformed slot syntax or a slot that cannot be filled."""


class UnknownSlotError(SlotError):
    """A slot name or role that is not defined in §3.2 / §2.1."""


@dataclass(frozen=True)
class SlotRef:
    """One placeholder found in a text."""

    name: str
    role: str | None
    start: int
    end: int

    @property
    def is_speaker(self) -> bool:
        """True for speaker references such as ``{S2}``."""
        return SPEAKER_RE.fullmatch(self.name) is not None

    @property
    def base(self) -> str:
        """Lexicon slot name without index (``PROJECT_2`` → ``PROJECT``); speakers unchanged."""
        return self.name if self.is_speaker else base_slot(self.name) or self.name

    @property
    def speaker_index(self) -> int | None:
        """The ``n`` of ``{Sn}``, or None for lexicon slots."""
        return int(self.name[1:]) if self.is_speaker else None


@dataclass(frozen=True)
class FilledText:
    """Rendered text plus the character spans of filled role slots."""

    text: str
    spans: list[dict[str, int | str]] = field(default_factory=list)


def parse_slots(text: str) -> list[SlotRef]:
    """Return the slots in ``text`` in order of appearance.

    Raises:
        UnknownSlotError: unknown slot name, unknown role, or a role on a slot that cannot
            carry it.
        SlotError: stray or nested braces.
    """
    refs: list[SlotRef] = []
    for match in _SLOT_RE.finditer(text):
        name, role = match.group(1), match.group(2)
        is_speaker = SPEAKER_RE.fullmatch(name) is not None
        base = name if is_speaker else base_slot(name)
        if base is None:
            raise UnknownSlotError(f"unknown slot {{{name}}} in: {text!r}")
        if role is not None:
            if role not in ROLE_SLOTS:
                raise UnknownSlotError(f"unknown role {role!r} in: {text!r}")
            allowed = ROLE_SLOTS[role]
            if not (base in allowed or (is_speaker and role == ROLE_OWNER)):
                raise UnknownSlotError(f"slot {{{name}}} cannot carry role {role!r}: {text!r}")
        refs.append(SlotRef(name=name, role=role, start=match.start(), end=match.end()))
    leftover = _SLOT_RE.sub("", text)
    if "{" in leftover or "}" in leftover:
        raise SlotError(f"stray brace in: {text!r}")
    return refs


def load_lexicons(directory: Path) -> dict[str, list[str]]:
    """Load ``<slot_lower>.txt`` files: one value per line, ``#`` comments and blanks ignored.

    Raises:
        UnknownSlotError: a file whose name is not a known slot.
        SlotError: a value containing braces.
    """
    lexicons: dict[str, list[str]] = {}
    for path in sorted(directory.glob("*.txt")):
        name = path.stem.upper()
        if name not in SLOT_NAMES:
            raise UnknownSlotError(f"lexicon file for unknown slot: {path.name}")
        values = []
        for line in path.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            if "{" in value or "}" in value:
                raise SlotError(f"{path.name}: value contains a brace: {value!r}")
            values.append(value)
        lexicons[name] = values
    return lexicons


class SlotFiller:
    """Fills slots for one meeting, drawing each lexicon slot once and reusing it.

    Args:
        lexicons: slot name → candidate values.
        rng: the meeting's random generator (determinism).
        reserved: values never drawn for ``PERSON`` (the meeting's participants), so a
            mentioned third person never shares a participant's name.
        segment_scoped: slots whose values last for one segment only (see ``new_segment``);
            all other slots keep one value for the whole meeting.
    """

    def __init__(
        self,
        lexicons: Mapping[str, Sequence[str]],
        rng: random.Random,
        reserved: Iterable[str] = (),
        segment_scoped: Iterable[str] = (),
    ) -> None:
        self._lexicons = lexicons
        self._rng = rng
        self._reserved = frozenset(reserved)
        self._scoped = frozenset(segment_scoped)
        self._values: dict[str, str] = {}

    def new_segment(self) -> None:
        """Forget the values of segment-scoped slots (call before each new segment)."""
        for name in [k for k in self._values if base_slot(k) in self._scoped]:
            del self._values[name]

    @property
    def values(self) -> dict[str, str]:
        """Values drawn so far in this meeting (copy)."""
        return dict(self._values)

    def value(self, name: str) -> str:
        """Return the meeting's value for lexicon slot ``name``, drawing it on first use.

        Indexed slots of one base (``PROJECT``, ``PROJECT_2``) never share a value.
        """
        base = base_slot(name)
        if base is None:
            raise UnknownSlotError(f"unknown slot {{{name}}}")
        if name not in self._values:
            taken = self._reserved | {v for k, v in self._values.items() if base_slot(k) == base}
            candidates = [v for v in self._lexicons.get(base, ()) if v not in taken]
            if not candidates:
                raise SlotError(f"no lexicon values available for {{{name}}}")
            self._values[name] = self._rng.choice(candidates)
        return self._values[name]

    def fill(self, text: str, speaker_map: Mapping[str, str]) -> FilledText:
        """Render ``text``: lexicon slots from the meeting cache, ``{Sn}`` from ``speaker_map``.

        Returns the rendered text and ``{start, end, role}`` spans of role slots, as offsets
        into the rendered text.
        """
        parts: list[str] = []
        spans: list[dict[str, int | str]] = []
        cursor = 0
        length = 0
        for ref in parse_slots(text):
            literal = text[cursor : ref.start]
            parts.append(literal)
            length += len(literal)
            if ref.is_speaker:
                if ref.name not in speaker_map:
                    raise SlotError(f"speaker {{{ref.name}}} is not mapped: {text!r}")
                filled = speaker_map[ref.name]
            else:
                filled = self.value(ref.name)
            if ref.role is not None:
                spans.append({"start": length, "end": length + len(filled), "role": ref.role})
            parts.append(filled)
            length += len(filled)
            cursor = ref.end
        parts.append(text[cursor:])
        return FilledText(text="".join(parts), spans=spans)
