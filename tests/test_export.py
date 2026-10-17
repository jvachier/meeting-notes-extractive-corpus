"""Unify and export: column types match §2.2, release layout and card are complete."""

from __future__ import annotations

import json
import random

import pyarrow.parquet as pq
import pytest
import yaml

from generation.bank import SeedBank
from generation.generate_synthetic_notes import build_pools, generate, meeting_specs
from generation.output import NOTE_SCHEMA, SENTENCE_SCHEMA, to_table
from generation.schema import Source
from processing.export_parquet import export
from processing.quality_checks import run_checks
from processing.unify import UnifyError, unify, validate_rows
from tests.bank_fixture import FILLERS, LEXICONS, make_hard_negatives, make_segments, small_config


@pytest.fixture
def corpus(cfg):
    scfg = small_config(cfg)
    bank = SeedBank(tuple(make_segments()), tuple(make_hard_negatives()), FILLERS, LEXICONS)
    pools = build_pools(bank, scfg, scfg.seed)
    syn = generate(
        meeting_specs("syn", Source.SYNTHETIC, 10, scfg, random.Random(0)),
        bank,
        pools,
        scfg,
        scfg.seed,
    )
    hn = generate(
        meeting_specs("hn", Source.SYNTHETIC_HN, 4, scfg, random.Random(1), lambda r: 3),
        bank,
        pools,
        scfg,
        scfg.seed,
    )
    sentences, notes = unify(
        to_table(syn, SENTENCE_SCHEMA),
        to_table(hn, SENTENCE_SCHEMA),
        to_table([], SENTENCE_SCHEMA),
        to_table([], NOTE_SCHEMA),
    )
    return scfg, bank, sentences, notes


def test_unified_tables_have_exact_schema(corpus):
    _, _, sentences, notes = corpus
    assert sentences.schema == SENTENCE_SCHEMA
    assert notes.schema == NOTE_SCHEMA
    assert notes.num_rows == 5 * 10 + 5 * 4


def test_validate_rows_rejects_bad_order(corpus):
    rows = corpus[2].to_pylist()[:5]
    rows[1], rows[2] = rows[2], rows[1]
    with pytest.raises(UnifyError):
        validate_rows(rows)


def test_export_layout_types_and_card(corpus, tmp_path):
    scfg, _, sentences, notes = corpus
    stats = export(sentences, notes, scfg, tmp_path)
    for config in ("synthetic", "ami"):
        for table, schema in (("sentences", SENTENCE_SCHEMA), ("notes", NOTE_SCHEMA)):
            for split in ("train", "validation", "test"):
                path = tmp_path / "data" / config / table / f"{split}.parquet"
                assert pq.read_schema(path).remove_metadata() == schema.remove_metadata()
    total = sum(f["rows"] for k, f in stats["files"].items() if "/sentences/" in k)
    assert total == sentences.num_rows
    card = (tmp_path / "README.md").read_text()
    front = yaml.safe_load(card.split("---")[1])
    assert front["license"] == "cc-by-4.0"
    assert {c["config_name"] for c in front["configs"]} == {
        "synthetic",
        "synthetic_notes",
        "ami",
        "ami_notes",
    }
    meta = json.loads((tmp_path / "dataset-metadata.json").read_text())
    assert meta["licenses"] == [{"name": "CC-BY-4.0"}]


def test_quality_checks_pass_on_fixture(corpus):
    scfg, bank, sentences, notes = corpus
    report = run_checks(sentences, notes, bank, scfg)
    titles = {s.title: s.passed for s in report.sections}
    assert titles["1. Schema"] is True
    assert titles["3. Duplicates"] is True
    assert "## 5. Shortcut probes (no ML)" in report.markdown()
