"""Merge the synthetic, hard-negative and AMI layers into one corpus (PLAN §2.2).

    uv run python -m processing.unify

Every sentence row is validated against ``generation.schema.SentenceRow``. The notes table
gets one row per meeting; AMI notes keep their human abstractive summaries.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
from pydantic import ValidationError

from generation.output import (
    NOTE_SCHEMA,
    PROCESSED_DIR,
    SENTENCE_SCHEMA,
    SYNTHETIC_DIR,
    read_parquet,
    to_table,
    write_parquet,
)
from generation.schema import NoteRow, SentenceRow

SOURCE_ORDER = {"synthetic": 0, "synthetic_hn": 1, "ami": 2}


class UnifyError(ValueError):
    """The layers cannot be merged into a valid corpus."""


def validate_rows(rows: Iterable[dict[str, Any]], max_errors: int = 20) -> int:
    """Validate rows against ``SentenceRow`` and sentence ids per meeting; return the count."""
    errors: list[str] = []
    expected_id: dict[str, int] = defaultdict(int)
    n = 0
    for row in rows:
        n += 1
        try:
            SentenceRow.model_validate(row)
        except ValidationError as exc:
            errors.append(
                f"{row.get('note_id')}#{row.get('sentence_id')}: {exc.errors()[0]['msg']}"
            )
        if row["sentence_id"] != expected_id[row["note_id"]]:
            errors.append(f"{row['note_id']}: sentence_id {row['sentence_id']} out of order")
        expected_id[row["note_id"]] = row["sentence_id"] + 1
        if len(errors) >= max_errors:
            break
    if errors:
        raise UnifyError("invalid rows:\n  " + "\n  ".join(errors))
    return n


def notes_from_sentences(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One notes row per synthetic meeting, derived from its sentences."""
    meetings: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        meetings[row["note_id"]].append(row)
    notes = []
    for note_id, sentences in meetings.items():
        first = sentences[0]
        notes.append(
            {
                "note_id": note_id,
                "source": first["source"],
                "meeting_type": first["meeting_type"],
                "split": first["split"],
                "n_sentences": len(sentences),
                "n_words": sum(s["token_count"] for s in sentences),
                "n_participants": len({s["speaker"] for s in sentences}),
            }
        )
    return notes


def unify(
    synthetic: pa.Table, hard_negatives: pa.Table, ami: pa.Table, ami_notes: pa.Table
) -> tuple[pa.Table, pa.Table]:
    """Return (sentences, notes) tables for the whole corpus, sorted by source and note id."""
    sentences = pa.concat_tables(
        [
            t.select(SENTENCE_SCHEMA.names).cast(SENTENCE_SCHEMA)
            for t in (synthetic, hard_negatives, ami)
        ]
    )
    rows = sentences.to_pylist()
    rows.sort(key=lambda r: (SOURCE_ORDER[r["source"]], r["note_id"], r["sentence_id"]))
    validate_rows(rows)
    note_ids = [r["note_id"] for r in rows]
    if len({(r["source"], r["note_id"]) for r in rows}) != len(set(note_ids)):
        raise UnifyError("a note_id is used by two sources")

    notes = notes_from_sentences([r for r in rows if r["source"] != "ami"])
    notes += ami_notes.to_pylist()
    notes.sort(key=lambda n: (SOURCE_ORDER[n["source"]], n["note_id"]))
    for note in notes:
        NoteRow.model_validate(note)
    return to_table(rows, SENTENCE_SCHEMA), to_table(notes, NOTE_SCHEMA)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--synthetic", type=Path, default=SYNTHETIC_DIR / "synthetic_notes.parquet")
    parser.add_argument(
        "--hard-negatives", type=Path, default=SYNTHETIC_DIR / "hard_negatives.parquet"
    )
    parser.add_argument("--ami", type=Path, default=PROCESSED_DIR / "ami_labelled.parquet")
    parser.add_argument("--ami-notes", type=Path, default=PROCESSED_DIR / "ami_notes.parquet")
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR / "unified_corpus.parquet")
    parser.add_argument("--notes-out", type=Path, default=PROCESSED_DIR / "unified_notes.parquet")
    args = parser.parse_args(argv)
    sentences, notes = unify(
        read_parquet(args.synthetic),
        read_parquet(args.hard_negatives),
        read_parquet(args.ami),
        read_parquet(args.ami_notes),
    )
    write_parquet(sentences, args.out)
    write_parquet(notes, args.notes_out)
    print(f"wrote {args.out}: {sentences.num_rows} sentences, {notes.num_rows} meetings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
