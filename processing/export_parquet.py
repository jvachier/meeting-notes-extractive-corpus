"""Write the local release folder (PLAN §6): Parquet per config and split, card, stats.

    uv run python -m processing.export_parquet

Layout (Hugging Face configs select files, so each config has its own folder):

    data/release/v1/
    ├── README.md                      dataset card with Hugging Face YAML front matter
    ├── stats.json                     counts, label rates, provenance
    ├── dataset-metadata.json          Kaggle metadata
    └── data/{synthetic,ami}/{sentences,notes}/{train,validation,test}.parquet

The ``synthetic`` config holds sources ``synthetic`` and ``synthetic_hn``. Publishing to the
Hub and Kaggle is a separate, manual step (PLAN §6, phase G6).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from generation.config import DEFAULT_CONFIG, REPO_ROOT, GenerationConfig, load_config
from generation.output import PROCESSED_DIR, RELEASE_DIR, read_parquet, write_parquet
from processing.assign_splits import SPLIT_ORDER
from processing.release_config import (
    DEFAULT_RELEASE_CONFIG,
    ReleaseConfigError,
    ReleaseSettings,
    load_release_settings,
)

CONFIG_SOURCES: dict[str, tuple[str, ...]] = {
    "synthetic": ("synthetic", "synthetic_hn"),
    "ami": ("ami",),
}


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _subset(table: pa.Table, sources: tuple[str, ...], split: str) -> pa.Table:
    mask = pc.and_(pc.is_in(table["source"], pa.array(sources)), pc.equal(table["split"], split))
    return table.filter(mask)


def write_files(sentences: pa.Table, notes: pa.Table, out: Path) -> dict[str, Any]:
    """Write every config/table/split file; return per-file stats."""
    files: dict[str, Any] = {}
    for config, sources in CONFIG_SOURCES.items():
        for name, table in (("sentences", sentences), ("notes", notes)):
            for split in SPLIT_ORDER:
                part = _subset(table, sources, split.value)
                rel = f"data/{config}/{name}/{split.value}.parquet"
                path = write_parquet(part, out / rel)
                files[rel] = {"rows": part.num_rows, "sha256": _sha256(path)}
    return files


def compute_stats(sentences: pa.Table, notes: pa.Table, cfg: GenerationConfig) -> dict[str, Any]:
    """Counts and label rates per config and split."""
    rows = sentences.select(["source", "split", "importance", "zone", "hard_negative"]).to_pylist()
    stats: dict[str, Any] = {}
    for config, sources in CONFIG_SOURCES.items():
        stats[config] = {}
        for split in SPLIT_ORDER:
            part = [r for r in rows if r["source"] in sources and r["split"] == split.value]
            meetings = sum(
                1
                for n in notes.select(["source", "split"]).to_pylist()
                if n["source"] in sources and n["split"] == split.value
            )
            stats[config][split.value] = {
                "meetings": meetings,
                "sentences": len(part),
                "importance_rate": round(sum(r["importance"] for r in part) / len(part), 4)
                if part
                else 0.0,
                "zones": dict(sorted(Counter(r["zone"] for r in part).items())),
                "hard_negative_sentences": sum(r["hard_negative"] for r in part),
            }
    return stats


def reuse_stats(sentences: pa.Table) -> dict[str, Any]:
    """How often seed segments are reused across synthetic meetings."""
    rows = sentences.select(["note_id", "segment_id"]).to_pylist()
    placements = {
        (r["note_id"], r["segment_id"])
        for r in rows
        if r["segment_id"] and not r["segment_id"].startswith("fill-")
    }
    uses = sorted(Counter(seg for _, seg in placements).values())
    if not uses:
        return {}
    return {
        "segments_used": len(uses),
        "median_uses": int(statistics.median(uses)),
        "max_uses": uses[-1],
    }


def dataset_card(
    stats: dict[str, Any],
    provenance: dict[str, Any],
    reuse: dict[str, Any],
    release: ReleaseSettings,
) -> str:
    """Hugging Face dataset card with YAML front matter."""
    configs = []
    for config in CONFIG_SOURCES:
        for table, suffix in (("sentences", ""), ("notes", "_notes")):
            files = "\n".join(
                f"      - split: {s.value}\n        path: data/{config}/{table}/{s.value}.parquet"
                for s in SPLIT_ORDER
            )
            default = (
                "    default: true\n" if config == "synthetic" and table == "sentences" else ""
            )
            configs.append(f"  - config_name: {config}{suffix}\n{default}    data_files:\n{files}")
    rows = []
    for config in CONFIG_SOURCES:
        for split in SPLIT_ORDER:
            s = stats[config][split.value]
            rows.append(
                f"| {config} | {split.value} | {s['meetings']:,} | {s['sentences']:,} | {s['importance_rate']:.3f} |"
            )
    stat_table = "\n".join(rows)
    front = f"""---
license: cc-by-4.0
language:
  - en
pretty_name: {release.title}
size_categories:
  - 100K<n<1M
task_categories:
  - text-classification
  - summarization
  - token-classification
tags:
  - meetings
  - extractive-summarization
  - action-items
  - decision-detection
  - synthetic
  - ami
configs:
{chr(10).join(configs)}
---
"""
    body = f"""
# {release.title}

Meeting transcripts in which **every sentence** carries two labels:

- `importance` (0/1): whether the sentence belongs in extractive meeting notes;
- `zone`: `context`, `discussion`, `decision`, `action_item` or `closing`.

Action-item sentences in the synthetic layer also carry character `spans` for the **owner**
and the **deadline**.

Source code, seeds and the full build pipeline: [{release.source_repo_url}]({release.source_repo_url})
(built from commit `{provenance["git_commit"][:12]}`).

## Configurations

| Config | Content |
|---|---|
| `synthetic` (default) | Sentences of the synthetic meetings (`source` = `synthetic` or `synthetic_hn`) |
| `synthetic_notes` | One row per synthetic meeting |
| `ami` | Sentences (dialogue acts) of 137 AMI scenario meetings, labelled from human annotation |
| `ami_notes` | One row per AMI meeting, with the human abstractive summary sections |

```python
from datasets import load_dataset

synthetic = load_dataset("{release.hf_repo_id}", "synthetic", revision="{release.tag}")
ami = load_dataset("{release.hf_repo_id}", "ami", revision="{release.tag}")
```

## Statistics

| Config | Split | Meetings | Sentences | importance = 1 |
|---|---|---|---|---|
{stat_table}

## How it was built

**Synthetic layer.** Claude Code (an AI coding assistant) wrote a bank of short labelled
meeting segments once, by hand, following fixed briefs and labelling rules. Each segment has
a paraphrase variant; a segment and its variants form a *family*. The bank also contains
*hard negatives*: sentences that look like decisions or action items but are not (rejected
proposals, tentative ideas, ownerless tasks, reversed decisions, decisions reported from
elsewhere, hypotheticals, unanswered questions). A deterministic Python program then composed
the segments into full meetings of five types (quick sync, standup, project review,
stakeholder update, retrospective) across eight business domains. It maps speakers to
participants, fills slots (names, projects, dates, figures) from lexicons of invented
values, inserts backchannels, and renders an ASR-style copy of every sentence. **No language
model runs in the pipeline, and the program never creates or changes a label.**

**Splits** are assigned per seed family (80/10/10), so no text from a test family appears in a
training meeting.

**AMI layer.** Dialogue acts from the AMI Meeting Corpus manual annotations (v1.6.2), labelled
from AMI's human extractive summaries and their links to the abstractive summary:

| Dialogue act | importance | zone | confidence |
|---|---|---|---|
| Extracted, linked to a *decisions* summary sentence | 1 | decision | 0.9 |
| Extracted, linked to an *actions* summary sentence | 1 | action_item | 0.9 |
| Extracted, linked to a *problems* summary sentence | 1 | discussion | 0.9 |
| Extracted, linked to the abstract only, or not linked | 1 | discussion | 0.8 |
| Not extracted | 0 | discussion | 0.8 |

AMI splits follow the standard scenario partition (test: ES2004, ES2014, IS1009, TS3003,
TS3007; validation: ES2003, ES2011, IS1008, TS3004, TS3006).

## Columns (`sentences`)

| Column | Type | Description |
|---|---|---|
| `note_id` | string | Meeting id |
| `source` | string | `synthetic`, `synthetic_hn` or `ami` |
| `meeting_type` | string | Synthetic meeting type, or `ami_design` |
| `domain` | string | Synthetic domain, or `product_design` |
| `split` | string | `train`, `validation`, `test` |
| `sentence_id` | int32 | 0-based position in the meeting |
| `speaker` | string | Participant name, or AMI speaker letter |
| `sentence_text` | string | Clean text, cased and punctuated |
| `sentence_text_asr` | string | Lowercased, unpunctuated, light disfluencies (synthetic only) |
| `importance` | int8 | 0 or 1 |
| `confidence` | float32 | Label confidence (hard negatives 0.6–0.8; AMI 0.8–0.9) |
| `zone` | string | Sentence zone |
| `segment_id`, `family_id` | string | Seed provenance (`fill-*` for backchannels; null for AMI) |
| `hard_negative` | bool | Sentence comes from a hard-negative segment |
| `token_count` | int32 | Whitespace word count |
| `spans` | list<struct<start, end, role>> | Character offsets of action-item `owner` / `deadline` |

`notes` columns: `note_id`, `source`, `meeting_type`, `split`, `n_sentences`, `n_words`,
`n_participants`, and for AMI `summary_abstract`, `summary_decisions`, `summary_actions`,
`summary_problems`.

## Known limitations

- **Synthetic style.** The synthetic meetings come from one author's seed bank and are cleaner
  and more regular than real speech. Evaluate on the AMI test split.
- **Seed reuse.** Each seed segment appears in about {reuse.get("median_uses", "?")} meetings
  (maximum {reuse.get("max_uses", "?")}), with different names, values, speakers and order.
  Splits are by seed family, so test text never appears in training meetings.
- **Topic coherence.** A long synthetic meeting draws on segments from several domains, so it
  can move between unrelated topics more than a real meeting would.
- **Slot values.** Names, dates and figures come from lexicons; a figure can occasionally be
  implausible for its context.
- **AMI domain.** AMI meetings are role-played product-design meetings, with no `context` or
  `closing` annotation; zone metrics on AMI cover `discussion`, `decision` and `action_item`.
- **Annotator agreement.** AMI extractive summaries have modest agreement between annotators
  (κ ≈ 0.48), which the confidence column reflects.
- Backchannels and opening/closing small talk are labelled `discussion`, importance 0.

## Licence and attribution

CC BY 4.0. Names, companies and products in the synthetic layer are invented or very common
values. The AMI-derived layer is derived from the AMI Meeting Corpus (CC BY 4.0); please cite:

> Carletta, J. et al. (2005). The AMI Meeting Corpus: A Pre-announcement. *Machine Learning for
> Multimodal Interaction (MLMI 2005)*, LNCS 3869, Springer.

Provenance: version {provenance["version"]}, generation seed {provenance["seed"]}, commit
`{provenance["git_commit"]}`.
"""
    return front + body


def kaggle_metadata(release: ReleaseSettings) -> dict[str, Any]:
    """``dataset-metadata.json`` for the Kaggle CLI."""
    return {
        "title": release.title,
        "id": release.kaggle_dataset_id,
        "subtitle": release.subtitle,
        "description": f"See README.md. Source and pipeline: {release.source_repo_url}",
        "licenses": [{"name": release.kaggle_licence}],
        "keywords": list(release.kaggle_keywords),
    }


def export(
    sentences: pa.Table,
    notes: pa.Table,
    cfg: GenerationConfig,
    out: Path,
    release: ReleaseSettings,
) -> dict[str, Any]:
    """Write the full release folder and return ``stats.json`` content."""
    files = write_files(sentences, notes, out)
    provenance = {
        "version": release.version,
        "seed": cfg.seed,
        "git_commit": _git_commit(),
        "repository": release.source_repo_url,
    }
    reuse = reuse_stats(sentences.filter(pc.not_equal(sentences["source"], "ami")))
    stats = {
        "provenance": provenance,
        "splits": compute_stats(sentences, notes, cfg),
        "seed_reuse": reuse,
        "files": files,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        dataset_card(stats["splits"], provenance, reuse, release), encoding="utf-8"
    )
    (out / "dataset-metadata.json").write_text(
        json.dumps(kaggle_metadata(release), indent=2) + "\n", encoding="utf-8"
    )
    return stats


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sentences", type=Path, default=PROCESSED_DIR / "unified_corpus.parquet")
    parser.add_argument("--notes", type=Path, default=PROCESSED_DIR / "unified_notes.parquet")
    parser.add_argument("--out", type=Path, default=RELEASE_DIR / "v1")
    parser.add_argument("--release-config", type=Path, default=DEFAULT_RELEASE_CONFIG)
    args = parser.parse_args(argv)
    try:
        release = load_release_settings(args.release_config)
    except ReleaseConfigError as exc:
        parser.error(str(exc))
    stats = export(
        read_parquet(args.sentences),
        read_parquet(args.notes),
        load_config(args.config),
        args.out,
        release,
    )
    print(f"wrote {args.out}: {len(stats['files'])} Parquet files, README.md, stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
