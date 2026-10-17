"""AMI conversion on a tiny hand-made NXT archive (no download)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from generation.relabel_ami import (
    AmiArchive,
    AmiFormatError,
    convert_meeting,
    join_words,
    parse_href,
)
from generation.schema import SentenceRow

NS = 'xmlns:nite="http://nite.sourceforge.net/"'
M = "ES2004a"  # a test-split meeting in the standard partition


def _words(speaker: str, words: list[tuple[str, float | None, bool]]) -> str:
    items = []
    for i, (text, start, punc) in enumerate(words):
        attrs = f' starttime="{start}"' if start is not None else ""
        attrs += ' punc="true"' if punc else ""
        items.append(f'<w nite:id="{M}.{speaker}.words{i}"{attrs}>{text}</w>')
        if i == 1 and speaker == "A":
            items.append(f'<vocalsound nite:id="{M}.{speaker}.words{i}x" type="laugh"/>')
    return f'<?xml version="1.0"?><nite:root {NS}>{"".join(items)}</nite:root>'


def _das(speaker: str, ranges: list[tuple[int, int]]) -> str:
    acts = "".join(
        f'<dact nite:id="{M}.{speaker}.dialog-act.{k}"><nite:child href="{M}.{speaker}.words.xml#id({M}.{speaker}.words{a})..id({M}.{speaker}.words{b})"/></dact>'
        for k, (a, b) in enumerate(ranges)
    )
    return f'<?xml version="1.0"?><nite:root {NS}>{acts}</nite:root>'


def _archive(tmp_path: Path) -> Path:
    files = {
        # A: "we take the rubber case ." (t=1) | "i will draw it ." (t=5)
        f"words/{M}.A.words.xml": _words(
            "A",
            [
                ("we", 1.0, False),
                ("take", 1.2, False),
                ("the", 1.3, False),
                ("rubber", 1.4, False),
                ("case", 1.5, False),
                (".", 1.5, True),
                ("I", 5.0, False),
                ("will", 5.1, False),
                ("draw", 5.2, False),
                ("it", 5.3, False),
                (".", 5.3, True),
            ],
        ),
        f"dialogueActs/{M}.A.dialog-act.xml": _das("A", [(0, 5), (6, 10)]),
        # B: "okay" (t=2) | "battery life is a problem ." (t=3) | "um" with no time (t=None) | "nice ." (t=6)
        f"words/{M}.B.words.xml": _words(
            "B",
            [
                ("okay", 2.0, False),
                ("battery", 3.0, False),
                ("life", 3.1, False),
                ("is", 3.2, False),
                ("a", 3.3, False),
                ("problem", 3.4, False),
                (".", 3.4, True),
                ("nice", 6.0, False),
                (".", 6.0, True),
            ],
        ),
        f"dialogueActs/{M}.B.dialog-act.xml": _das("B", [(0, 0), (1, 6), (7, 8)]),
        f"extractive/{M}.extsumm.xml": (
            f'<?xml version="1.0"?><nite:root {NS}><extsumm nite:id="{M}.extsumm.1">'
            f'<nite:child href="{M}.A.dialog-act.xml#id({M}.A.dialog-act.0)..id({M}.A.dialog-act.1)"/>'
            f'<nite:child href="{M}.B.dialog-act.xml#id({M}.B.dialog-act.1)"/>'
            f'<nite:child href="{M}.B.dialog-act.xml#id({M}.B.dialog-act.2)"/>'
            "</extsumm></nite:root>"
        ),
        f"abstractive/{M}.abssumm.xml": (
            f'<?xml version="1.0"?><nite:root {NS}>'
            f'<abstract nite:id="a"><sentence nite:id="{M}.s.1">They met.</sentence></abstract>'
            f'<actions nite:id="b"><sentence nite:id="{M}.s.2">A will draw it.</sentence></actions>'
            f'<decisions nite:id="c"><sentence nite:id="{M}.s.3">Rubber case.</sentence></decisions>'
            f'<problems nite:id="d"><sentence nite:id="{M}.s.4">Battery life.</sentence></problems>'
            "</nite:root>"
        ),
        f"extractive/{M}.summlink.xml": (
            f'<?xml version="1.0"?><nite:root {NS}>'
            + "".join(
                f'<summlink nite:id="l{i}"><nite:pointer role="extractive" href="{M}.{da.split(".")[1]}.dialog-act.xml#id({da})"/>'
                f'<nite:pointer role="abstractive" href="{M}.abssumm.xml#id({M}.s.{s})"/></summlink>'
                for i, (da, s) in enumerate(
                    [
                        (f"{M}.A.dialog-act.0", 1),  # abstract ...
                        (f"{M}.A.dialog-act.0", 3),  # ... and decisions → decisions wins
                        (f"{M}.A.dialog-act.1", 2),  # actions
                        (f"{M}.B.dialog-act.1", 4),  # problems
                    ]
                )
            )
            + "</nite:root>"
        ),
    }
    path = tmp_path / "ami.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return path


def test_conversion_labels_and_order(tmp_path, cfg):
    archive = AmiArchive(_archive(tmp_path))
    assert archive.meetings_with_extractive_summary() == [M]
    rows, note, stats = convert_meeting(archive, M, cfg)
    got = [
        (r["speaker"], r["sentence_text"], r["importance"], r["zone"], r["confidence"])
        for r in rows
    ]
    assert got == [
        ("A", "We take the rubber case.", 1, "decision", 0.9),
        ("B", "Okay", 0, "discussion", 0.8),
        ("B", "Battery life is a problem.", 1, "discussion", 0.9),
        ("A", "I will draw it.", 1, "action_item", 0.9),
        ("B", "Nice.", 1, "discussion", 0.8),  # extracted, not linked
    ]
    assert all(r["split"] == "test" for r in rows)
    assert [r["sentence_id"] for r in rows] == list(range(5))
    for row in rows:
        SentenceRow.model_validate(row)
        assert row["sentence_text_asr"] == row["sentence_text_asr"].lower()
    assert note["summary_decisions"] == "Rubber case."
    assert note["n_participants"] == 2 and note["n_sentences"] == 5
    assert stats["das"] == 5


def test_join_words_attaches_punctuation():
    from generation.relabel_ami import Token

    tokens = [
        Token("1", True, "hi", False, 0.0),
        Token("2", True, ",", True, 0.0),
        Token("3", True, "there", False, 0.0),
        Token("4", False, "", False, None),
    ]
    assert join_words(tokens) == "Hi, there"


def test_parse_href():
    assert parse_href("x.xml#id(a)..id(b)") == ("x.xml", "a", "b")
    assert parse_href("x.xml#id(a)") == ("x.xml", "a", None)
    with pytest.raises(AmiFormatError):
        parse_href("nonsense")


def test_split_of_uses_partition(cfg):
    assert cfg.ami.split_of("ES2004c") == "test"
    assert cfg.ami.split_of("IS1008a") == "validation"
    assert cfg.ami.split_of("ES2002a") == "train"
