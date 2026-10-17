"""Arrow schemas for the output files (PLAN §2.2) and deterministic Parquet writing."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from generation.config import REPO_ROOT

DATA_DIR = REPO_ROOT / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
RELEASE_DIR = DATA_DIR / "release"

SPAN_TYPE = pa.struct([("start", pa.int32()), ("end", pa.int32()), ("role", pa.string())])

SENTENCE_SCHEMA = pa.schema(
    [
        ("note_id", pa.string()),
        ("source", pa.string()),
        ("meeting_type", pa.string()),
        ("domain", pa.string()),
        ("split", pa.string()),
        ("sentence_id", pa.int32()),
        ("speaker", pa.string()),
        ("sentence_text", pa.string()),
        ("sentence_text_asr", pa.string()),
        ("importance", pa.int8()),
        ("confidence", pa.float32()),
        ("zone", pa.string()),
        ("segment_id", pa.string()),
        ("family_id", pa.string()),
        ("hard_negative", pa.bool_()),
        ("token_count", pa.int32()),
        ("spans", pa.list_(SPAN_TYPE)),
    ]
)

NOTE_SCHEMA = pa.schema(
    [
        ("note_id", pa.string()),
        ("source", pa.string()),
        ("meeting_type", pa.string()),
        ("split", pa.string()),
        ("n_sentences", pa.int32()),
        ("n_words", pa.int32()),
        ("n_participants", pa.int32()),
        ("summary_abstract", pa.string()),
        ("summary_decisions", pa.string()),
        ("summary_actions", pa.string()),
        ("summary_problems", pa.string()),
    ]
)

FAMILY_SPLIT_SCHEMA = pa.schema(
    [
        ("family_id", pa.string()),
        ("split", pa.string()),
        ("stratum", pa.string()),
        ("hard_negative", pa.bool_()),
    ]
)


def to_table(rows: Iterable[Mapping[str, Any]], schema: pa.Schema) -> pa.Table:
    """Build a table with exactly ``schema`` (missing optional fields become null)."""
    rows = list(rows)
    columns = {name: [row.get(name) for row in rows] for name in schema.names}
    return pa.table(columns, schema=schema)


def write_parquet(table: pa.Table, path: Path) -> Path:
    """Write ``table`` with fixed settings so equal tables give byte-identical files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        store_schema=True,
    )
    return path


def read_parquet(path: Path) -> pa.Table:
    """Read a Parquet file written by this pipeline."""
    return pq.read_table(path)
