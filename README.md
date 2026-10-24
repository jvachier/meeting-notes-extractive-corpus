# meeting-notes-extractive-corpus

[![CI](https://github.com/jvachier/meeting-notes-extractive-corpus/actions/workflows/ci.yaml/badge.svg)](https://github.com/jvachier/meeting-notes-extractive-corpus/actions/workflows/ci.yaml)
[![Code licence: MIT](https://img.shields.io/badge/code%20licence-MIT-green.svg)](LICENSE)
[![Data licence: CC BY 4.0](https://img.shields.io/badge/data%20licence-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Source and build pipeline for **meeting-notes-extractive-v1**, a corpus of meeting
transcripts in which every sentence is labelled for extractive note-taking.

| Label | Values | Meaning |
|---|---|---|
| `importance` | `0` / `1` | Whether the sentence belongs in the meeting notes |
| `zone` | `context`, `discussion`, `decision`, `action_item`, `closing` | The sentence's role in the meeting |
| `spans` | `owner`, `deadline` | Character offsets of who owns an action item and when it is due |

The corpus supports training and evaluating models for sentence selection, decision and
action-item detection, and owner/deadline extraction. It pairs a large synthetic layer, with
labels fixed by construction, with a human-annotated layer from the AMI Meeting Corpus that
serves as the real-world benchmark.

> **Status:** v1.0 is built and passes all quality checks locally. It is not yet published
> on the Hugging Face Hub or Kaggle.

## Contents

- [Example](#example)
- [Statistics](#statistics)
- [Usage](#usage)
- [Data format](#data-format)
- [How the corpus is built](#how-the-corpus-is-built)
- [Quality controls](#quality-controls)
- [Known limitations](#known-limitations)
- [Reproducing the build](#reproducing-the-build)
- [Repository layout](#repository-layout)
- [Licence and attribution](#licence-and-attribution)
- [Citation](#citation)

## Example

An excerpt from a synthetic quick sync in the test split (`hn-00025`, research domain). It
includes hard negatives: sentences that look like decisions but are not.

| # | Speaker | Sentence | `importance` | `zone` | Hard negative |
|---|---|---|:-:|---|:-:|
| 2 | Bianca | Please do, it's about the Harlowe Textiles renewal. | 0 | context | |
| 3 | Bianca | Nobody has reached out to them, and their contract runs out on July 11th. | **1** | context | |
| 4 | Wei | That's close. | 0 | context | |
| 5 | Wei | Supplier gets paid in full this week. | 0 | decision | ✓ |
| 6 | Selam | Only half the order has been delivered. | 0 | discussion | ✓ |
| 7 | Wei | Okay, half now, the rest when it all arrives. | **1** | decision | ✓ |
| … | | | | | |
| 13 | Selam | We still don't have the product screenshots for the campaign. | 0 | discussion | |
| 14 | Bianca | I asked the design team for them last week. | 0 | discussion | |
| 15 | Selam | Okay, Bianca follows up with them by Monday morning, and if nothing comes, we use the old ones. | **1** | action_item | |
| 16 | Bianca | Ah, Monday morning, good. | 0 | action_item | |

Sentence 5 is a proposal that is reversed two turns later, so only the revised decision
(sentence 7) is kept. Sentence 16 belongs to the action-item zone but only repeats the
deadline, so it is not kept.

The full row for sentence 15:

```json
{
  "note_id": "hn-00025",
  "source": "synthetic_hn",
  "meeting_type": "quick_sync",
  "domain": "research",
  "split": "test",
  "sentence_id": 15,
  "speaker": "Selam",
  "sentence_text": "Okay, Bianca follows up with them by Monday morning, and if nothing comes, we use the old ones.",
  "sentence_text_asr": "okay bianca uh follows up with them by monday morning and if nothing comes we use the old ones",
  "importance": 1,
  "confidence": 1.0,
  "zone": "action_item",
  "segment_id": "act-quick_sync-mkt-003-v1",
  "family_id": "act-quick_sync-mkt-003",
  "hard_negative": false,
  "token_count": 18,
  "spans": [
    {"start": 6, "end": 12, "role": "owner"},
    {"start": 37, "end": 51, "role": "deadline"}
  ]
}
```

`sentence_text[6:12]` is `"Bianca"` and `sentence_text[37:51]` is `"Monday morning"`.

AMI rows have the same shape and contain real, disfluent speech. Their labels come from the
AMI annotators' extractive summaries:

| Meeting | Speaker | Sentence | `importance` | `zone` | `confidence` |
|---|---|---|:-:|---|:-:|
| ES2004b | A | Mm-hmm. | 0 | discussion | 0.8 |
| ES2004b | D | Th the internal d engineering design has got to remain the same, yes, absolutely. | **1** | decision | 0.9 |
| ES2004b | B | It's gonna be the same, | 0 | discussion | 0.8 |

## Statistics

| Layer | Split | Meetings | Sentences | `importance = 1` |
|---|---|--:|--:|--:|
| Synthetic | train | 1,200 | 110,230 | 25.6% |
| | validation | 150 | 13,881 | 25.0% |
| | test | 150 | 13,772 | 24.9% |
| AMI | train | 97 | 75,996 | 17.5% |
| | validation | 20 | 16,476 | 17.5% |
| | test | 20 | 15,876 | 18.6% |
| **Total** | | **1,637** | **246,231** | |

The synthetic layer has 1,250 normal meetings and 250 meetings that each contain three or
four hard-negative segments (`source = synthetic_hn`). It covers five meeting types (quick
sync, standup, project review, stakeholder update, retrospective) and eight business domains.
Meeting length ranges from 150–300 words for a quick sync to 600–1,100 words for a project
review. AMI meetings are longer (median about 5,000 words).

The full breakdown, by zone, meeting type and source, is in `stats.json` and in the quality
report each build writes.

## Usage

After the v1.0 release on the Hugging Face Hub:

```python
from datasets import load_dataset

synthetic = load_dataset("<hf-user>/meeting-notes-extractive-v1", "synthetic", revision="v1.0")
ami = load_dataset("<hf-user>/meeting-notes-extractive-v1", "ami", revision="v1.0")
```

Until then, [build it locally](#reproducing-the-build) and read the Parquet files directly:

```python
import pandas as pd

sentences = pd.read_parquet("data/release/v1/data/synthetic/sentences/test.parquet")
notes = sentences.groupby("note_id")["sentence_text"].apply(" ".join)
```

**Recommended protocol:** train and tune on synthetic train and validation, then report
results on **AMI test**. The synthetic test split measures in-distribution fit. AMI test
measures transfer to real meetings.

## Data format

Each layer (`synthetic`, `ami`) is released as its own configuration with two tables, each
split into `train`, `validation` and `test`:

```
data/{synthetic,ami}/sentences/{train,validation,test}.parquet
data/{synthetic,ami}/notes/{train,validation,test}.parquet
```

### `sentences`: one row per sentence

| Column | Type | Description |
|---|---|---|
| `note_id` | string | Meeting identifier |
| `source` | string | `synthetic`, `synthetic_hn` or `ami` |
| `meeting_type` | string | Synthetic meeting type, or `ami_design` |
| `domain` | string | Synthetic business domain, or `product_design` |
| `split` | string | `train`, `validation` or `test` |
| `sentence_id` | int32 | 0-based position in the meeting |
| `speaker` | string | Participant name, or AMI speaker letter |
| `sentence_text` | string | Clean text, cased and punctuated |
| `sentence_text_asr` | string | ASR-style copy: lowercased, unpunctuated, light disfluencies |
| `importance` | int8 | 1 if the sentence belongs in the notes |
| `confidence` | float32 | 1.0 for synthetic labels; 0.8–0.9 for AMI (see below) |
| `zone` | string | One of the five zones |
| `segment_id`, `family_id` | string | Seed provenance (null for AMI) |
| `hard_negative` | bool | The sentence comes from a hard-negative segment |
| `token_count` | int32 | Whitespace word count (tokenizer-neutral) |
| `spans` | list&lt;struct&gt; | `{start, end, role}` character offsets into `sentence_text`; `role` is `owner` or `deadline` |

### `notes`: one row per meeting

`note_id`, `source`, `meeting_type`, `split`, `n_sentences`, `n_words`, `n_participants`.
AMI rows also carry the human abstractive summary sections (`summary_abstract`,
`summary_decisions`, `summary_actions`, `summary_problems`), so abstractive metrics such as
ROUGE can be computed against them.

## How the corpus is built

### Synthetic layer

1. **Hand-written seeds.** A bank of 1,460 short labelled meeting segments (2–12 sentences
   each) was written once, in Claude Code sessions, following fixed briefs
   ([`generation/prompts/`](generation/prompts/)) and labelling rules. It covers every
   combination of zone, meeting type and domain. It includes 300 hard-negative segments in
   seven categories: rejected proposals, tentative ideas, ownerless tasks, reversed
   decisions, hypotheticals, questions that are not commitments, and work reported
   elsewhere.
2. **Deterministic composition.** A Python composer assembles full meetings from the seeds,
   following a grammar per meeting type. It maps speakers to participants, fills slots
   (names, projects, dates, figures) from lexicons of invented values, inserts backchannels
   and small talk, and renders the ASR-style copy of each sentence.
3. **No model in the pipeline.** After seeding, no language model, API or GPU is involved.
   The pipeline never creates or changes a label. It only carries seed labels through
   composition, slot filling and noise.

Each seed has paraphrase variants, which together form a **family**. Splits are assigned per
family (80 / 10 / 10, stratified by zone and domain) before composition, so text from a test
family never appears in a training meeting.

### AMI layer

AMI dialogue acts are labelled from AMI's own human extractive summaries and their links to
the sections of the abstractive summary:

| Dialogue act | `importance` | `zone` | `confidence` |
|---|:-:|---|:-:|
| Extracted, linked to a *decisions* summary sentence | 1 | decision | 0.9 |
| Extracted, linked to an *actions* summary sentence | 1 | action_item | 0.9 |
| Extracted, linked to a *problems* summary sentence | 1 | discussion | 0.9 |
| Extracted, linked to the abstract only, or not linked | 1 | discussion | 0.8 |
| Not extracted | 0 | discussion | 0.8 |

AMI has no annotation for `context` or `closing`, so these zones never occur in the AMI
layer. Splits follow the standard AMI full-corpus partition, by meeting.

## Quality controls

Every build writes a quality report. The v1.0 build passes all of these checks:

- **Schema:** every row validates against the row model, and Arrow types match the
  specification.
- **Leakage:** no family appears in more than one split. Test 8-grams also found in train
  (slot values and fillers removed): 0.00%, against a 1% limit.
- **Duplicates:** no exact duplicate segments, and no near-duplicates across families
  (word 3-gram Jaccard ≥ 0.8).
- **Distributions:** importance rate and zone mix are reported per meeting type, split and
  source. 100% of synthetic meetings fall inside their target length range.
- **Shortcut probe:** a 21-keyword rule baseline is scored on each test set (see
  [Known limitations](#known-limitations)).
- **Determinism:** the same seed produces byte-identical Parquet files.

Seed validation (schema, ID conventions, slot syntax, variant consistency, coverage and
lexicon size) runs on every commit in CI.

## Known limitations

- **Synthetic register.** The synthetic meetings are cleaner and more regular than real
  speech, and topics change at segment boundaries rather than flowing naturally. The ASR copy
  adds light noise but does not reproduce real recognition errors.
- **Keyword shortcuts.** The keyword baseline scores F1 0.55 on synthetic test but 0.18 on
  AMI test. The gap of 0.37 is above the 0.25 warning threshold, which means surface cues
  predict importance more often in the synthetic data than in real meetings. Report results
  on AMI test, not synthetic test alone.
- **AMI domain and labels.** AMI meetings are role-played design sessions in one domain
  (remote-control product design). Their importance labels come from one annotator's
  extractive summary per meeting, and inter-annotator agreement on AMI extractive summaries
  is modest (κ ≈ 0.48 reported). Zones are derived from summary links. Both are reflected in
  `confidence`.
- **Zone of fillers.** Backchannels and small talk inserted between segments, including
  greetings and sign-offs placed at the start and end of a meeting, always carry zone
  `discussion` and importance 0.
- **Segment reuse.** Each seed segment appears in about 22 meetings on average (at most 72).
  Splits by family prevent leakage, but models can still memorise recurring phrasing within
  the training set.
- **English only.**

## Reproducing the build

Requirements: Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                                  # create the environment
cp .env.example .env                                     # then fill in your release IDs
uv run python -m generation.relabel_ami --download       # fetch AMI annotations (once)
uv run python -m processing.build                        # full pipeline → data/release/v1
uv run python -m processing.quality_checks --determinism # quality report + byte-identity check
```

The build takes about 25 seconds on a laptop and writes the quality report to
`data/processed/report.md`. All volumes, ratios, probabilities and length ranges are in
[`configs/generation.yaml`](configs/generation.yaml), and every script accepts `--seed`.
[`configs/generation_tiny.yaml`](configs/generation_tiny.yaml) is a smoke configuration that
runs in a few seconds. Release metadata (version, tag, titles) is in
[`configs/release.yaml`](configs/release.yaml); the Hugging Face repo, Kaggle dataset and
source repository URL are read from the git-ignored `.env` (template:
[`.env.example`](.env.example)) or from environment variables of the same names.

### Development

```bash
uv run pre-commit install                                     # enable git hooks
uv run pytest -q                                              # tests (offline)
uv run ruff check . --fix && uv run ruff format .             # lint and format
uv run python -m processing.validate_seeds                    # seed checks and coverage
uv run python -m processing.validate_seeds --strict-coverage  # also fail on coverage gaps
uv run python -m processing.publish --dry-run --hf --kaggle   # preview the release upload
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution rules and
[SECURITY.md](SECURITY.md) for reporting security or personal-data issues. New seeds follow
the procedures in [`.claude/skills/`](.claude/skills/), and each seed batch must pass
`validate_seeds`. The full design specification is in [PLAN.md](PLAN.md).

## Repository layout

```
configs/              generation settings (full and tiny)
seeds/                hand-written seed bank: segments, hard negatives, fillers, coverage targets
generation/           schemas, slot filling, prompts, lexicons, composition, AMI conversion
processing/           seed validation, splits, unification, quality checks, export, publishing
tests/                pytest suite (offline, no downloads)
.claude/skills/       procedures for writing seeds and converting AMI
data/                 local working data (git-ignored)
```

## Licence and attribution

- **Code** (pipeline, tests, configs, tooling): [MIT](LICENSE).
- **Synthetic data**, seeds, lexicons and prompts: [CC BY 4.0](LICENSE-DATA). All names,
  companies and products are invented or very common values. Any resemblance to real people
  or organisations is coincidental.
- **AMI-derived data**: CC BY 4.0, derived from the AMI Meeting Corpus manual annotations
  (v1.6.2). If you use the AMI layer, please also cite:

  > Carletta, J. et al. (2005). *The AMI Meeting Corpus: A Pre-announcement.* Machine
  > Learning for Multimodal Interaction (MLMI 2005), LNCS 3869, Springer.

- **ICSI** is not included in v1, because the licence of its summary annotations has not
  been verified.

## Citation

```bibtex
@misc{vachier2026meetingnotes,
  author       = {Vachier, Jeremy},
  title        = {meeting-notes-extractive-v1: Sentence-Labelled Meeting Transcripts for Extractive Note-Taking},
  year         = {2026},
  howpublished = {\url{https://github.com/jvachier/meeting-notes-extractive-corpus}}
}
```
