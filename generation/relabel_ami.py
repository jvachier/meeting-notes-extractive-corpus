"""Convert AMI manual annotations into sentence-labelled rows (PLAN §5).

    uv run python -m generation.relabel_ami --download   # fetch the archive once
    uv run python -m generation.relabel_ami              # convert

Sentence unit: AMI dialogue acts (DAs), in time order, text rebuilt from the cased and
punctuated word tokens. Labels come only from AMI's human annotation:

- extracted DA linked to a *decisions* / *actions* / *problems* summary sentence
  → importance 1, zone decision / action_item / discussion;
- extracted DA linked to the abstract only, or not linked → importance 1, discussion;
- DA not extracted → importance 0, discussion.

A DA linked to several sections takes the first of decisions, actions, problems, abstract.
The archive is read in place (never extracted) and parsed without entities or network.
"""

from __future__ import annotations

import argparse
import random
import re
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lxml import etree

from generation.config import DEFAULT_CONFIG, AmiConfig, GenerationConfig, load_config
from generation.noise import render_asr
from generation.output import (
    NOTE_SCHEMA,
    PROCESSED_DIR,
    RAW_DIR,
    SENTENCE_SCHEMA,
    to_table,
    write_parquet,
)
from generation.schema import AMI_DOMAIN, AMI_MEETING_TYPE, Source, Zone

NITE = "{http://nite.sourceforge.net/}"
SECTIONS: tuple[str, ...] = ("decisions", "actions", "problems", "abstract")
SECTION_ZONE: dict[str, Zone] = {
    "decisions": Zone.DECISION,
    "actions": Zone.ACTION_ITEM,
    "problems": Zone.DISCUSSION,
    "abstract": Zone.DISCUSSION,
}
_HREF_RE = re.compile(r"^(?P<file>[^#]+)#id\((?P<a>[^)]+)\)(?:\.\.id\((?P<b>[^)]+)\))?$")
_PARSER = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False, huge_tree=False)
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.?!;:])")


class AmiFormatError(ValueError):
    """The archive does not look like AMI NXT annotations."""


@dataclass(frozen=True)
class Token:
    """One element of a words file, in document order."""

    id: str
    is_word: bool
    text: str
    punc: bool
    start: float | None


@dataclass(frozen=True)
class DialogueAct:
    """One DA with its text and start time."""

    id: str
    speaker: str
    text: str
    start: float | None
    order: int


class AmiArchive:
    """Read-only access to the AMI annotation zip."""

    def __init__(self, path: Path) -> None:
        self._zip = zipfile.ZipFile(path)
        self._names = set(self._zip.namelist())

    def has(self, name: str) -> bool:
        """True if ``name`` is in the archive."""
        return name in self._names

    def xml(self, name: str) -> etree._Element:
        """Parse one member safely."""
        return etree.fromstring(self._zip.read(name), _PARSER)

    def meetings_with_extractive_summary(self) -> list[str]:
        """Meeting ids that have ``extractive/<id>.extsumm.xml``, sorted."""
        return sorted(
            n[len("extractive/") : -len(".extsumm.xml")]
            for n in self._names
            if n.startswith("extractive/") and n.endswith(".extsumm.xml")
        )

    def speakers(self, meeting: str) -> list[str]:
        """Speaker letters with a dialogue-act file for ``meeting``."""
        pattern = re.compile(rf"^dialogueActs/{re.escape(meeting)}\.([A-Z])\.dialog-act\.xml$")
        return sorted(m.group(1) for n in self._names if (m := pattern.match(n)))


def parse_href(href: str) -> tuple[str, str, str | None]:
    """Split ``file#id(a)..id(b)`` into (file, a, b or None)."""
    match = _HREF_RE.match(href)
    if match is None:
        raise AmiFormatError(f"unexpected href {href!r}")
    return match.group("file"), match.group("a"), match.group("b")


def _expand(ids: list[str], index: dict[str, int], a: str, b: str | None) -> list[str]:
    if a not in index or (b is not None and b not in index):
        raise AmiFormatError(f"href points to a missing id: {a}..{b}")
    lo = index[a]
    hi = index[b] if b is not None else lo
    return ids[lo : hi + 1]


def read_tokens(root: etree._Element) -> list[Token]:
    """All elements of a words file, in document order."""
    tokens = []
    for el in root:
        if not isinstance(el.tag, str):
            continue
        start = el.get("starttime")
        tokens.append(
            Token(
                id=el.get(f"{NITE}id", ""),
                is_word=etree.QName(el).localname == "w",
                text=(el.text or "").strip(),
                punc=el.get("punc") == "true",
                start=float(start) if start not in (None, "") else None,
            )
        )
    return tokens


def join_words(tokens: list[Token]) -> str:
    """Rebuild a sentence: words joined by spaces, punctuation attached, first letter upper."""
    text = " ".join(t.text for t in tokens if t.is_word and t.text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text).strip()
    return text[:1].upper() + text[1:] if text else text


def read_dialogue_acts(archive: AmiArchive, meeting: str) -> list[DialogueAct]:
    """DAs of every speaker, sorted by start time (then speaker, then file order)."""
    acts: list[DialogueAct] = []
    for speaker in archive.speakers(meeting):
        words_name = f"words/{meeting}.{speaker}.words.xml"
        if not archive.has(words_name):
            raise AmiFormatError(f"missing {words_name}")
        tokens = read_tokens(archive.xml(words_name))
        ids = [t.id for t in tokens]
        index = {t.id: i for i, t in enumerate(tokens)}
        last_start: float | None = None
        da_root = archive.xml(f"dialogueActs/{meeting}.{speaker}.dialog-act.xml")
        for order, dact in enumerate(da_root.iter("dact")):
            child = dact.find(f"{NITE}child")
            if child is None:
                continue
            _, a, b = parse_href(child.get("href", ""))
            span = [tokens[index[i]] for i in _expand(ids, index, a, b)]
            starts = [t.start for t in span if t.is_word and t.start is not None]
            start = min(starts) if starts else last_start
            last_start = start if start is not None else last_start
            acts.append(
                DialogueAct(
                    id=dact.get(f"{NITE}id", ""),
                    speaker=speaker,
                    text=join_words(span),
                    start=start,
                    order=order,
                )
            )
    return sorted(
        acts, key=lambda d: (d.start if d.start is not None else float("inf"), d.speaker, d.order)
    )


def read_extracted(archive: AmiArchive, meeting: str) -> set[str]:
    """Ids of DAs in the extractive summary (ranges expanded within each speaker file)."""
    root = archive.xml(f"extractive/{meeting}.extsumm.xml")
    da_ids: dict[str, list[str]] = {}
    extracted: set[str] = set()
    for child in root.iter(f"{NITE}child"):
        file, a, b = parse_href(child.get("href", ""))
        if file not in da_ids:
            da_root = archive.xml(f"dialogueActs/{file}")
            da_ids[file] = [d.get(f"{NITE}id", "") for d in da_root.iter("dact")]
        ids = da_ids[file]
        extracted.update(_expand(ids, {x: i for i, x in enumerate(ids)}, a, b))
    return extracted


def read_abstractive(archive: AmiArchive, meeting: str) -> tuple[dict[str, str], dict[str, str]]:
    """Return (sentence id → section, section → joined summary text)."""
    name = f"abstractive/{meeting}.abssumm.xml"
    if not archive.has(name):
        return {}, {}
    root = archive.xml(name)
    section_of: dict[str, str] = {}
    texts: dict[str, list[str]] = {s: [] for s in SECTIONS}
    for section in SECTIONS:
        for block in root.iter(section):
            for sentence in block.iter("sentence"):
                section_of[sentence.get(f"{NITE}id", "")] = section
                text = " ".join((sentence.text or "").split())
                if text:
                    texts[section].append(text)
    return section_of, {s: " ".join(v) for s, v in texts.items() if v}


def read_links(archive: AmiArchive, meeting: str, section_of: dict[str, str]) -> dict[str, str]:
    """DA id → highest-precedence linked summary section."""
    name = f"extractive/{meeting}.summlink.xml"
    if not archive.has(name):
        return {}
    best: dict[str, str] = {}
    for link in archive.xml(name).iter("summlink"):
        da, sentence = None, None
        for pointer in link.iter(f"{NITE}pointer"):
            _, a, _ = parse_href(pointer.get("href", ""))
            if pointer.get("role") == "extractive":
                da = a
            elif pointer.get("role") == "abstractive":
                sentence = a
        section = section_of.get(sentence or "")
        if da is None or section is None:
            continue
        if da not in best or SECTIONS.index(section) < SECTIONS.index(best[da]):
            best[da] = section
    return best


def label(
    da_id: str, extracted: set[str], links: dict[str, str], conf: AmiConfig
) -> tuple[int, Zone, float]:
    """PLAN §5 mapping from annotation status to (importance, zone, confidence)."""
    if da_id not in extracted:
        return 0, Zone.DISCUSSION, conf.confidence.not_extracted
    section = links.get(da_id)
    if section in ("decisions", "actions", "problems"):
        return 1, SECTION_ZONE[section], conf.confidence.extracted_linked
    return 1, Zone.DISCUSSION, conf.confidence.extracted_other


def convert_meeting(
    archive: AmiArchive, meeting: str, cfg: GenerationConfig
) -> tuple[list[dict[str, Any]], dict[str, Any], Counter[str]]:
    """Rows, notes row and counters for one meeting."""
    stats: Counter[str] = Counter()
    split = cfg.ami.split_of(meeting).value
    extracted = read_extracted(archive, meeting)
    section_of, summaries = read_abstractive(archive, meeting)
    links = read_links(archive, meeting, section_of)
    rng = random.Random(f"{cfg.seed}:{meeting}")
    rows: list[dict[str, Any]] = []
    for act in read_dialogue_acts(archive, meeting):
        if not act.text:
            stats["empty_da_dropped"] += 1
            if act.id in extracted:
                stats["empty_extracted_da_dropped"] += 1
            continue
        importance, zone, confidence = label(act.id, extracted, links, cfg.ami)
        rows.append(
            {
                "note_id": meeting,
                "source": Source.AMI.value,
                "meeting_type": AMI_MEETING_TYPE,
                "domain": AMI_DOMAIN,
                "split": split,
                "sentence_id": len(rows),
                "speaker": act.speaker,
                "sentence_text": act.text,
                "sentence_text_asr": render_asr(act.text, rng, cfg.noise, disfluent=False),
                "importance": importance,
                "confidence": confidence,
                "zone": zone.value,
                "segment_id": None,
                "family_id": None,
                "hard_negative": False,
                "token_count": len(act.text.split()),
                "spans": [],
            }
        )
    stats["das"] += len(rows)
    stats["extracted"] += sum(r["importance"] for r in rows)
    note = {
        "note_id": meeting,
        "source": Source.AMI.value,
        "meeting_type": AMI_MEETING_TYPE,
        "split": split,
        "n_sentences": len(rows),
        "n_words": sum(r["token_count"] for r in rows),
        "n_participants": len({r["speaker"] for r in rows}),
        **{f"summary_{s}": summaries.get(s) for s in SECTIONS},
    }
    return rows, note, stats


def download(url: str, target: Path) -> Path:
    """Fetch the archive into ``target`` unless it is already there."""
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as fh:
        while chunk := response.read(1 << 20):
            fh.write(chunk)
    tmp.rename(target)
    return target


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--download", action="store_true", help="fetch the archive and exit")
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR / "ami_labelled.parquet")
    parser.add_argument("--notes-out", type=Path, default=PROCESSED_DIR / "ami_notes.parquet")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg = cfg.model_copy(update={"seed": args.seed})
    archive_path = args.archive or RAW_DIR / "ami" / f"ami_public_manual_{cfg.ami.version}.zip"
    if args.download:
        print(f"archive at {download(cfg.ami.url, archive_path)}")
        return 0
    archive = AmiArchive(archive_path)
    rows, notes, stats = [], [], Counter()
    for meeting in archive.meetings_with_extractive_summary():
        meeting_rows, note, meeting_stats = convert_meeting(archive, meeting, cfg)
        rows.extend(meeting_rows)
        notes.append(note)
        stats.update(meeting_stats)
    write_parquet(to_table(rows, SENTENCE_SCHEMA), args.out)
    write_parquet(to_table(notes, NOTE_SCHEMA), args.notes_out)
    splits = Counter(n["split"] for n in notes)
    zones = Counter((r["zone"], r["importance"]) for r in rows)
    print(f"wrote {args.out}: {len(notes)} meetings {dict(splits)}, {len(rows)} DAs")
    print(f"importance-1 rate {stats['extracted'] / max(len(rows), 1):.3f}; labels {dict(zones)}")
    print(
        f"dropped: {stats['empty_da_dropped']} empty DAs "
        f"({stats['empty_extracted_da_dropped']} of them extracted)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
