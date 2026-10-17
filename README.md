# meeting-notes-extractive-corpus

[![CI](https://github.com/jvachier/meeting-notes-extractive-corpus/actions/workflows/ci.yaml/badge.svg)](https://github.com/jvachier/meeting-notes-extractive-corpus/actions/workflows/ci.yaml)
[![Code licence: MIT](https://img.shields.io/badge/code%20licence-MIT-green.svg)](LICENSE)
[![Data licence: CC BY 4.0](https://img.shields.io/badge/data%20licence-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

This repository builds **meeting-notes-extractive-v1**, a corpus of meeting transcripts in
which every sentence carries two labels:

| Label | Values | Meaning |
|---|---|---|
| `importance` | `0` / `1` | Whether the sentence belongs in extractive meeting notes |
| `zone` | `context`, `discussion`, `decision`, `action_item`, `closing` | What the sentence does in the meeting |

The corpus is meant for training and evaluating extractive note-taking models: sentence
selection, decision and action-item detection, and owner/deadline span extraction.

> **Status: built locally, not yet released.** The seed bank, generation pipeline, AMI
> conversion and quality checks are complete; publishing to the Hugging Face Hub and Kaggle
> is the remaining step. See the [roadmap](#roadmap).

## Contents

- [Dataset design](#dataset-design)
- [Data format](#data-format)
- [Quality controls](#quality-controls)
- [Roadmap](#roadmap)
- [Development](#development)
- [Repository layout](#repository-layout)
- [Licence and attribution](#licence-and-attribution)

## Dataset design

The corpus has two layers, released as two configurations.

| Layer | Source | Labels from | Meetings | Splits |
|---|---|---|---|---|
| **Synthetic** | Seed segments composed by Python | The seed author, by construction | ~3,000 | train / validation / test |
| **AMI** | [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) manual annotations | Human annotators | ~140 | standard AMI partition |

### Synthetic layer

1. **Seeds, written once by hand.** Claude Code writes a bank of about 1,460 short labelled
   meeting segments (2–12 sentences each), following fixed briefs and labelling rules. The
   bank covers 5 zones × 5 meeting types (quick sync, standup, project review, stakeholder
   update, retrospective) × 8 business domains. It includes 300 *hard negatives*: sentences
   that look like decisions or action items but are not (rejected proposals, tentative ideas,
   ownerless tasks, reversed decisions, and so on).
2. **Composition by Python only.** A deterministic composer assembles full meetings from the
   seeds, following a grammar per meeting type. It maps speakers to participants, fills
   slots (names, projects, dates, figures) from lexicons of invented values, inserts
   backchannels, and renders an ASR-style copy of every sentence.
3. **No model in the pipeline.** No LLM, API or GPU is involved after seeding, and Python
   never creates or changes a label. It only carries seed labels through composition.

Each seed comes with paraphrase variants, which together form a *family*. Splits are
assigned **per family**, so text from a test family never appears in a training meeting.

### AMI layer

AMI dialogue acts, labelled from AMI's own human extractive summaries and their links to the
abstractive summary sections:

| Dialogue act | `importance` | `zone` | `confidence` |
|---|---|---|---|
| Extracted, linked to a *decisions* summary sentence | 1 | decision | 0.9 |
| Extracted, linked to an *actions* summary sentence | 1 | action_item | 0.9 |
| Extracted, linked to a *problems* summary sentence | 1 | discussion | 0.9 |
| Extracted, linked to the abstract only, or not linked | 1 | discussion | 0.8 |
| Not extracted | 0 | discussion | 0.8 |

AMI has no `context` or `closing` annotation, and those zones are never invented for it.
The AMI test split is the main measure of how well synthetic training transfers to real
meetings.

## Data format

Release layout: `sentences/{train,validation,test}.parquet` and
`notes/{train,validation,test}.parquet`, plus a dataset card and `stats.json`.

**`sentences`**: one row per sentence.

| Column | Type | Description |
|---|---|---|
| `note_id` | string | Meeting identifier |
| `source` | string | `synthetic`, `synthetic_hn` or `ami` |
| `meeting_type` | string | Synthetic meeting type, or `ami_design` |
| `domain` | string | Synthetic domain, or `product_design` |
| `split` | string | `train`, `validation` or `test` |
| `sentence_id` | int32 | 0-based position in the meeting |
| `speaker` | string | Participant name, or AMI speaker letter |
| `sentence_text` | string | Clean text, cased and punctuated |
| `sentence_text_asr` | string | Lowercased, unpunctuated, light disfluencies |
| `importance` | int8 | 0 or 1 |
| `confidence` | float32 | Label confidence |
| `zone` | string | One of the five zones |
| `segment_id`, `family_id` | string | Seed provenance (null for AMI) |
| `hard_negative` | bool | Sentence comes from a hard-negative segment |
| `token_count` | int32 | Whitespace word count (tokenizer-neutral) |
| `spans` | list&lt;struct&gt; | `{start, end, role}` character offsets of action-item `owner` / `deadline` |

**`notes`**: one row per meeting: `note_id`, `source`, `meeting_type`, `split`,
`n_sentences`, `n_words`, `n_participants`. For AMI it also has the human abstractive summary
sections (`summary_abstract`, `summary_decisions`, `summary_actions`, `summary_problems`), so
abstractive metrics such as ROUGE can be computed.

**Planned usage** (after the v1.0 release on the Hugging Face Hub):

```python
from datasets import load_dataset

synthetic = load_dataset("jvachier/meeting-notes-extractive-v1", "synthetic", revision="v1.0")
ami = load_dataset("jvachier/meeting-notes-extractive-v1", "ami", revision="v1.0")
```

## Quality controls

- **Seed validation** on every batch and in CI: schema, ID and file conventions, slot syntax,
  variant consistency, duplicate detection, coverage targets and lexicon sizes.
- **Leakage**: no family in more than one split; 8-gram overlap between test and train text
  (slot values removed) must stay at or below 1%.
- **Duplicates**: exact duplicates are rejected; near-duplicates across families are flagged.
- **Distributions**: importance rate and zone mix per meeting type, split and source; length
  and segment-reuse histograms.
- **Shortcut probes**: a keyword rule baseline is scored on synthetic test and AMI test. A
  large gap means the synthetic labels are too easy.
- **Determinism**: the same seed gives byte-identical Parquet files.

The full specification is in [PLAN.md](PLAN.md).

## Roadmap

| Phase | Work | Status |
|---|---|---|
| P0 | Scaffolding: schemas, slots, seed validation, configs, skills, CI | ✅ Done |
| S1 | Lexicons and fillers | ✅ Done |
| S2 | Pilot seed batches (one per zone), human review | ✅ Done |
| S3 | Full segment bank (1,160 segments) | ✅ Done |
| S4 | Hard negatives (300 segments) | ✅ Done |
| G1–G2 | Meeting composition: synthetic and hard-negative meetings | ✅ Done |
| G3 | AMI conversion | ✅ Done |
| G4–G5 | Splits, unification, quality report | ✅ Done |
| G6 | Release v1.0 on Hugging Face Hub and Kaggle | ⏳ Local release built; publishing pending |

## Development

Requirements: Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                   # create the environment
uv run pre-commit install                 # enable git hooks
uv run pytest -q                          # tests
uv run ruff check . --fix && uv run ruff format .
uv run python -m processing.validate_seeds                    # seed checks + coverage
uv run python -m processing.validate_seeds --strict-coverage  # also fail on shortfalls

uv run python -m generation.relabel_ami --download            # fetch AMI annotations (once)
uv run python -m processing.build                             # whole pipeline → data/release/v1
uv run python -m processing.quality_checks --determinism      # report + byte-identity check
uv run python -m processing.publish --dry-run --hf --kaggle   # preview the release upload
```

The build takes about 25 seconds on a laptop and writes the quality report to
`data/processed/report.md`.

All volumes, ratios, probabilities and length ranges are in
[`configs/generation.yaml`](configs/generation.yaml). Every script takes `--seed`.
[`configs/generation_tiny.yaml`](configs/generation_tiny.yaml) is a smoke configuration that
runs in seconds.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution rules and
[SECURITY.md](SECURITY.md) for reporting security or personal-data issues. Contributions to
seeds follow the skills in [`.claude/skills/`](.claude/skills/) and the
briefs in [`generation/prompts/`](generation/prompts/). Each seed batch is committed
separately and must pass `validate_seeds`.

## Repository layout

```
configs/              generation settings (full and tiny)
seeds/                hand-written seed bank: segments, hard negatives, fillers, coverage targets
generation/           schemas, slot filling, prompts, lexicons, composition and AMI conversion
processing/           seed validation, splits, unification, quality checks, export
tests/                pytest suite (offline, no downloads)
.claude/skills/       procedures for writing seeds and converting AMI
data/                 local working data, git-ignored
```

## Licence and attribution

- **Code** (pipeline, tests, configs, tooling): [MIT](LICENSE).
- **Synthetic data**, seeds, lexicons and prompts: [CC BY 4.0](LICENSE-DATA). All names,
  companies and products are invented or very common values.
- **AMI-derived data**: CC BY 4.0, derived from the AMI Meeting Corpus manual annotations
  (v1.6.2). Please also cite:

  > Carletta, J. et al. (2005). *The AMI Meeting Corpus: A Pre-announcement.* Machine
  > Learning for Multimodal Interaction (MLMI 2005), LNCS 3869, Springer.

- **ICSI** is not included in v1, until the licence of its summary annotations is verified.
