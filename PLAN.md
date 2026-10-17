# PLAN: meeting-notes-extractive-v1

## 0. Goal and contract

The goal is a public corpus of meeting transcripts with **one label per sentence**:
- `importance`: keep (1) or discard (0) for extractive notes
- `zone`: context | discussion | decision | action_item | closing

The corpus has two layers:

| Layer | Source | Labels from | Role |
|---|---|---|---|
| Synthetic | Seed segments written by Claude Code, composed by Python | Seed author (by construction) | Train / val / test |
| Real | AMI meeting corpus (manual annotations) | Human annotators | Val / **real test** (+ optional train) |

Consumers (e.g. the Note_taker Model 1) download the published Parquet files at a pinned
version. Nothing in this repo depends on any consumer.

## 1. Repository layout

```
meeting-notes-extractive-corpus/
├── CLAUDE.md  PLAN.md  README.md  pyproject.toml
├── .claude/skills/
│   ├── write-seed-segments/   SKILL.md  explanation.md
│   ├── write-hard-negatives/  SKILL.md  explanation.md
│   └── convert-ami/           SKILL.md  explanation.md
├── configs/
│   ├── generation.yaml          # volumes, ratios, probabilities, length ranges, seed
│   └── generation_tiny.yaml     # smoke config, runs in < 30 s
├── seeds/                       # COMMITTED: written by Claude Code
│   ├── coverage.yaml            # target counts per (zone × meeting type × domain) cell
│   ├── segments/<zone>/<meeting_type>__<domain>__<batch>.jsonl
│   ├── hard_negatives/<category>__<batch>.jsonl
│   └── fillers/fillers.jsonl    # backchannels and small talk
├── generation/
│   ├── prompts/                 # seeding briefs for Claude Code, one per meeting type
│   │   ├── quick_sync.txt  standup.txt  project_review.txt
│   │   ├── stakeholder_update.txt  retrospective.txt
│   ├── lexicons/                # COMMITTED: slot vocabularies (names, projects, dates…)
│   ├── schema.py                # pydantic models for seeds and output rows
│   ├── slots.py                 # slot parsing and filling
│   ├── noise.py                 # disfluencies and ASR-style lowercasing
│   ├── compose.py               # segment bank → meetings
│   ├── generate_synthetic_notes.py
│   ├── generate_hard_negatives.py
│   └── relabel_ami.py
├── processing/
│   ├── validate_seeds.py
│   ├── assign_splits.py
│   ├── unify.py                 # merge layers, add sentence-level columns
│   ├── quality_checks.py        # leakage, duplicates, distributions → report
│   └── export_parquet.py        # final files + dataset card + publish
├── tests/
└── data/                        # git-ignored: raw/ synthetic/ processed/ release/
```

## 2. Schemas

### 2.1 Seed segment (one JSON object per line)

A **segment** is a short exchange (2–12 sentences) with one main purpose. A **family** is a
seed segment plus its paraphrase variants (written by Claude in the same batch). Splits are
assigned per family.

```json
{
  "segment_id": "dec-project_review-eng-003-v1",
  "family_id": "dec-project_review-eng-003",
  "primary_zone": "decision",
  "meeting_types": ["project_review", "stakeholder_update"],
  "domain": "engineering",
  "hard_negative": false,
  "hn_category": null,
  "confidence": 1.0,
  "n_speakers": 3,
  "sentences": [
    {"speaker": "S1", "text": "So for {PROJECT} we either ship on {DATE_REL} or wait for the load tests.", "importance": 0, "zone": "discussion"},
    {"speaker": "S2", "text": "The load tests are the only thing blocking us, honestly.", "importance": 0, "zone": "discussion"},
    {"speaker": "S1", "text": "Okay, decision: we hold the release until the load tests pass.", "importance": 1, "zone": "decision"},
    {"speaker": "S3", "text": "Works for me.", "importance": 0, "zone": "decision"}
  ]
}
```

Rules:
- **Every sentence carries its own `importance` and `zone`.** Filler sentences inside a
  decision segment ("Works for me.") keep the segment's zone but have importance 0.
- `speaker` uses segment-local IDs (`S1`…`Sn`). The composer maps them to meeting participants.
- Slots are `{UPPER_CASE}` names from §3.2. Speakers refer to each other with `{S1}`, `{S2}`,
  … so names stay consistent after mapping.
- One sentence per entry. Long turns are split into consecutive entries from the same speaker.
- **Slot roles (action items):** in `action_item` sentences with importance 1, the owner and
  deadline slots carry a role, e.g. `"{S2|owner} will send the draft by {DATE_REL|deadline}."`.
  The composer records the character offsets of filled role slots in the output column `spans`
  (list of `{start, end, role}`, empty for other sentences). No role marks for AMI.

### 2.2 Output row (one per sentence, final Parquet)

| Column | Type | Notes |
|---|---|---|
| `note_id` | str | One meeting (synthetic or AMI) |
| `source` | str | `synthetic` \| `synthetic_hn` \| `ami` |
| `meeting_type` | str | the 5 synthetic types; `ami_design` for AMI |
| `domain` | str | synthetic domain; `product_design` for AMI |
| `split` | str | train \| validation \| test |
| `sentence_id` | int32 | Position in the meeting, 0-based |
| `speaker` | str | Participant name / AMI speaker letter |
| `sentence_text` | str | Clean text (cased, punctuated) |
| `sentence_text_asr` | str | Lowercased, no punctuation, light disfluency (§3.6) |
| `importance` | int8 | 0 / 1 |
| `confidence` | float32 | From seed (synthetic) or fixed by annotation type (AMI, §5) |
| `zone` | str | context \| discussion \| decision \| action_item \| closing |
| `segment_id`, `family_id` | str | Provenance (null for AMI) |
| `hard_negative` | bool | Sentence comes from a hard-negative segment |
| `token_count` | int32 | Whitespace word count of `sentence_text` (tokenizer-neutral) |
| `spans` | list<struct> | `{start, end, role}` char offsets in `sentence_text`, roles `owner` / `deadline` (§2.1) |

A second file, `notes.parquet`, has one row per meeting: `note_id`, `source`, `meeting_type`,
`split`, `n_sentences`, `n_words`, `n_participants`. For AMI it also has the human abstractive
summary sections (`summary_abstract`, `summary_decisions`, `summary_actions`,
`summary_problems`), so consumers can compute ROUGE.

## 3. Synthetic layer: seeds (written by Claude Code)

### 3.1 Meeting types

| Type | Length (words) | Participants | Main features |
|---|---|---|---|
| quick_sync | 150–300 | 2–3 | Dense in action items |
| standup | 300–450 | 3–6 | Status / blockers / next steps |
| project_review | 600–1,100 | 4–7 | Decisions, risks, owners |
| stakeholder_update | 450–750 | 3–6 | High-level decisions |
| retrospective | 500–900 | 4–8 | Improvements, action items |

Lengths are in words, not tokens, so they don't depend on any tokenizer (about 1.3 subword
tokens per word). Ranges live in `configs/generation.yaml`.

### 3.2 Lexicons (`generation/lexicons/*.txt`, one value per line)

Slots:
- `PERSON` (≥ 400 first names, diverse origins, invented surnames)
- `TEAM`, `PROJECT` (invented), `PRODUCT` (invented), `CLIENT` (invented companies)
- `TOOL` (generic: "the dashboard", "the CI pipeline"), `DATE_REL` ("next Tuesday", "end of
  sprint"), `DATE_ABS`, `NUMBER`, `PERCENT`, `MONEY`, `METRIC`

Within one meeting, each entity slot (`PERSON`, `TEAM`, `PROJECT`, `PRODUCT`, `CLIENT`,
`TOOL`, `METRIC`) is drawn once and reused, so `{PROJECT}` stays the same project throughout.
Quantity and date slots (`NUMBER`, `PERCENT`, `MONEY`, `DATE_REL`, `DATE_ABS`) are drawn once
per segment, so unrelated exchanges don't share a number (`composition.segment_scoped_slots`). An indexed slot `{PROJECT_2}` … `{PROJECT_9}` names a second,
different value of the same lexicon (always distinct from `{PROJECT}`).

### 3.3 Domains (8)

engineering, product, sales, marketing, customer_support, hr_people, finance_ops, research.

### 3.4 Segment bank: target volumes (`seeds/coverage.yaml`)

| Zone | Families | Variants / family | Segments |
|---|---|---|---|
| context | 60 | 2 | 120 |
| discussion | 240 | 2 | 480 |
| decision | 120 | 2 | 240 |
| action_item | 120 | 2 | 240 |
| closing | 40 | 2 | 80 |
| **Total** | **580** | | **1,160** |

- Cells (zone × meeting type × domain) are filled roughly evenly. Each zone has at least one
  family per domain and at least 3 families per (zone, meeting type).
- Variants are real paraphrases: different wording, sentence order and speaker distribution,
  same labels. They are not slot swaps.
- `discussion` families must include both sentence kinds: important (risks, key facts,
  numbers, blockers; importance 1) and chatter (importance 0). Target about 30% importance 1
  within discussion.
- **Writing protocol:** each batch is about 10 families (20 segments) for one cell, following
  `generation/prompts/<meeting_type>.txt`. Run `validate_seeds` after every batch. Batches are
  committed separately, so one bad batch can be reverted without losing the others.

### 3.5 Hard negatives (`seeds/hard_negatives/`)

These are sentences that *look* like a decision or action item but are not. Importance is 0
unless noted. Confidence is between 0.6 and 0.8 (they are ambiguous on purpose).

| Category | Example pattern | Families |
|---|---|---|
| rejected_proposal | "We could move it to Q3." → "No, that won't work." | 25 |
| tentative_idea | "Maybe someone should look into caching at some point." | 25 |
| ownerless_task | "This needs fixing before launch." (no owner, no date) | 25 |
| reversed_decision | Decision made, then overturned. Only the final one has importance 1 | 20 |
| reported_elsewhere | "Finance already decided that last week." (status, not a new decision) | 20 |
| hypothetical | "If the client signs, we'd need two more people." | 20 |
| question_not_commitment | "Should we do a retro on this?" with no answer | 15 |
| **Total** | | **150 families × 2 = 300 segments** |

### 3.6 Fillers (`seeds/fillers/fillers.jsonl`)

About 250 one-sentence backchannels and small talk ("Can you hear me?", "Sorry, go ahead.",
"Yeah."), each importance 0, zone discussion. Some are tagged `opening` or `closing_chatter`
so they're only placed at the start or end.

## 4. Synthetic layer: composition (Python)

### 4.1 Meeting grammar (per type, in `configs/generation.yaml`)

```
meeting := context{1,2} body closing{1}
body    := (discussion | decision | action_item){k}  with type-specific zone weights
```

For example, quick_sync weights action_item highest and project_review weights decision
highest. `k` is chosen so the meeting fits the word range in §3.1.

### 4.2 Composition steps (per meeting)
1. Choose a meeting type and domain. Draw participants (count from §3.1) and slot values.
2. Sample segments for the split only (§4.4). Prefer segments whose `meeting_types` include
   the chosen type and whose domain matches, falling back to other domains with probability
   `p_cross_domain`. Preferences are soft weights (`type_weight`, `domain_weight`), damped as
   a segment approaches `max_segment_reuse` (`reuse_penalty_power`).
3. Map each segment's `S1…Sn` to participants. The meeting chair is S1 in context and closing.
4. Insert fillers between segments (`p_filler`) and occasionally inside them (`p_interrupt`,
   only between sentences with importance 0).
5. Fill slots consistently. Render `sentence_text`.
6. Render `sentence_text_asr`: lowercase, strip punctuation, and with probability `p_disfl`
   per sentence add one disfluency ("um", "uh", a one-word repetition).
   **The noise never changes labels.**
7. Never reuse a segment within the same meeting. Cap how often each segment appears across
   the corpus at `max_segment_reuse`.

### 4.3 Volumes

| File | Meetings | Notes |
|---|---|---|
| `synthetic_notes.parquet` | 1,250 (250 per type) | Normal composition |
| `hard_negatives.parquet` | 250 (50 per type) | Each meeting contains 3–4 hard-negative segments |

Cross-check against reuse: segments average about 34 words, so a meeting uses about 20 segments,
not 8. At 3,000 meetings that meant about 45 uses per segment; at 1,500 it is about 22. That's
acceptable only because splits are by family and the measured result is the real AMI test
(§7). If the quality checks show memorisation risk, more seeds are the fix, not more meetings.

### 4.4 Splits

- Families are assigned train / validation / test at 80 / 10 / 10, stratified by
  (primary_zone, domain), with a fixed seed. Hard-negative families are split the same way,
  stratified by category.
- **A meeting is composed only from segments of a single split.** Fillers are shared across
  splits; they are generic and all importance 0.
- Meeting counts per split follow 80/10/10.
- Family splits are assigned **before** composition (`processing/assign_splits.py`), because
  each meeting draws only on its own split.

## 5. Real layer: AMI

- **Source:** AMI manual annotations v1.6.2 (NXT format), CC BY 4.0, downloaded into
  `data/raw/ami/` by script. Only text annotations, no audio.
- **Meetings:** the ones with extractive summary annotations (about 140 scenario meetings).
- **Sentence unit:** AMI dialogue acts (DAs), in time order. Text is rebuilt from word
  tokens, with the cased and punctuated form where available.
- **Labels:**

  | DA status | importance | zone | confidence |
  |---|---|---|---|
  | Extracted, linked to a *decisions* summary sentence | 1 | decision | 0.9 |
  | Extracted, linked to an *actions* summary sentence | 1 | action_item | 0.9 |
  | Extracted, linked to a *problems* summary sentence | 1 | discussion | 0.9 |
  | Extracted, linked to *abstract* only, or not linked | 1 | discussion | 0.8 |
  | Not extracted | 0 | discussion | 0.8 |

  AMI has no context or closing annotation. Those zones are not invented for AMI, and
  `zone` metrics on AMI cover decision, action_item and discussion only. Confidence < 1
  reflects the modest agreement between annotators (κ ≈ 0.48 reported).
- **Splits:** the standard AMI full-corpus partition (train / dev → validation / test),
  by meeting.
- **ICSI:** excluded from v1. Its transcripts are CC BY 4.0, but the summary annotations are
  in the "contributed" package, whose licence is not verified. Revisit for v1.1.

## 6. Publishing

- Release folder: `data/release/v1/data/{synthetic,ami}/{sentences,notes}/{train,validation,test}.parquet`
  + `README.md` (dataset card) + `stats.json` + `dataset-metadata.json` (Kaggle). One folder
  per layer, because Hub configs select files, not rows.
- **Hugging Face Hub dataset** with configs `synthetic` (synthetic + synthetic_hn, default),
  `ami`, and their one-row-per-meeting tables `synthetic_notes` and `ami_notes`. Tag `v1.0`, which consumers pin.
- **Kaggle Datasets:** the same files.
- **Dataset card:**
  - how it was built (Claude Code seeds + Python composition, stated plainly)
  - label definitions
  - known limits: synthetic style, AMI role-play domain, annotator agreement
  - licence and AMI attribution

## 7. Quality checks (`processing/quality_checks.py` → `data/processed/report.md`)

1. **Schema:** every row is valid; no missing labels; zones from the enum.
2. **Leakage:** no `family_id` in more than one split. 8-gram overlap between test and train
   `sentence_text`, after removing slot values: must be ≤ 1% of test 8-grams outside fillers.
3. **Duplicates:** no exact duplicate segments. Near-duplicates (normalised text, Jaccard on
   word 3-grams ≥ 0.8) across different families are flagged and listed.
4. **Distributions:** importance rate and zone mix per meeting type, split and source; length
   histograms vs §3.1; segment reuse histogram.
5. **Shortcut probes, no ML:**
   - importance rate given the top 50 trigger words ("decided", "will", "by Friday")
   - accuracy of a keyword rule baseline on synthetic test vs AMI test

   A large gap between the two warns that the synthetic labels are too easy. Hard negatives
   should lower the rule's precision.
6. **Determinism:** running the pipeline twice with the same seed gives identical file
   hashes (tested on the tiny config).

## 8. Order of work (stop and report after each)

| Phase | What | Output |
|---|---|---|
| P0 | Scaffolding: pyproject, schema.py, slots.py, validate_seeds, tests, configs, skills | `uv run pytest` green |
| S1 | Lexicons + fillers (Claude Code) | `generation/lexicons/`, `seeds/fillers/` |
| S2 | Pilot: 5 batches (one per zone), reviewed by the user before scaling | ~100 segments |
| S3 | Full segment bank (§3.4) | 1,160 segments |
| S4 | Hard negatives (§3.5) | 300 segments |
| G1 | `compose.py` + `generate_synthetic_notes.py` | `data/synthetic/synthetic_notes.parquet` |
| G2 | `generate_hard_negatives.py` | `data/synthetic/hard_negatives.parquet` |
| G3 | `relabel_ami.py` | `data/processed/ami_labelled.parquet` |
| G4 | `assign_splits.py` + `unify.py` | `data/processed/unified_corpus.parquet` |
| G5 | `quality_checks.py` | `data/processed/report.md` |
| G6 | `export_parquet.py`: dataset card, HF + Kaggle publish (only after my go-ahead) | v1.0 release |

## 9. Tests (`tests/`)
- Schema: valid and invalid seeds; output row model.
- Slots: parsing, consistent filling within a meeting, unknown slot raises an error.
- Noise: labels and sentence count unchanged; ASR text lowercased with no punctuation.
- Compose: respects the word range, uses only same-split segments, maps speakers
  consistently, never repeats a segment within a meeting, deterministic for a fixed seed.
- Splits: no family in two splits; stratification within tolerance.
- AMI: label mapping on a tiny hand-made NXT fixture (no download in tests).
- Export: column types match §2.2.

## 9b. Amendments from the v1 build (approved)

- Indexed slots `{PROJECT_2}` (§3.2); participants may be mentioned without speaking (§2.1).
- Quantity and date slots scoped per segment (§3.2); soft, reuse-damped selection (§4.2).
- Synthetic volumes halved to 1,250 + 250 meetings to keep reuse near the §4.3 estimate.
- Splits assigned before composition (§4.4); release layout with one folder per layer (§6).
- Seeds include importance-0 echoes ("Wait, {PERCENT}?") and people addressing each other by
  name, so no slot or name predicts importance on its own (checked by §7.5).
