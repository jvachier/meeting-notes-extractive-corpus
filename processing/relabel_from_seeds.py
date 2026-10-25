"""Apply the current seed labels to already composed meetings, without changing any text.

Composition draws interrupt fillers only between two importance-0 sentences, from the
meeting's random stream, so regenerating after a relabel would shift every later sentence of
the meeting. v1.1 therefore keeps the v1.0 meetings byte-identical and relabels them in place:
each synthetic row records its seed ``segment_id``; within a meeting the rows of one segment
(fillers excluded) are its sentences in order, so row ``k`` takes the ``importance`` and
``zone`` of seed sentence ``k``.

Every row whose labels do not change must already match its seed; only the seed sentences
that were relabelled may differ. Any other mismatch means the alignment is wrong and raises.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import pyarrow as pa

from generation.output import SENTENCE_SCHEMA, to_table
from generation.schema import SeedSegment


class RelabelError(RuntimeError):
    """A composed meeting does not line up with its seed segments."""


def relabel_rows(
    rows: list[dict[str, Any]],
    segments: Iterable[SeedSegment],
    changed: set[tuple[str, int]],
) -> tuple[list[dict[str, Any]], int]:
    """Relabel ``rows`` from ``segments``; ``changed`` lists the ``(segment_id, index)`` seed
    sentences whose labels were edited. Returns the rows and the number of rows changed."""
    by_id = {s.segment_id: s for s in segments}
    placements: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: (r["note_id"], r["sentence_id"])):
        sid = row["segment_id"]
        if sid is not None and not sid.startswith("fill-"):
            placements[(row["note_id"], sid)].append(row)
    n_changed = 0
    for (note_id, sid), placed in placements.items():
        seed = by_id.get(sid)
        if seed is None or len(seed.sentences) != len(placed):
            raise RelabelError(f"{note_id}: segment {sid} does not match its seed")
        for k, (row, sentence) in enumerate(zip(placed, seed.sentences, strict=True)):
            new = (sentence.importance, sentence.zone.value)
            if (row["importance"], row["zone"]) == new:
                continue
            if (sid, k) not in changed:
                raise RelabelError(
                    f"{note_id}: {sid}[{k}] differs from its seed but was not edited"
                )
            row["importance"], row["zone"] = new
            n_changed += 1
    return rows, n_changed


def relabel_table(
    table: pa.Table, segments: Iterable[SeedSegment], changed: set[tuple[str, int]]
) -> tuple[pa.Table, int]:
    """:func:`relabel_rows` on a sentences table."""
    rows, n = relabel_rows(table.to_pylist(), list(segments), changed)
    return to_table(rows, SENTENCE_SCHEMA), n
