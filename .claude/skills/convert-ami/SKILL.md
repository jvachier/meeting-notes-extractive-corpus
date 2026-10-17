---
name: convert-ami
description: Download the AMI manual annotations, run the AMI → sentence-label conversion (relabel_ami.py) and check its output against PLAN §5. Use when asked to convert, relabel, refresh or check the AMI layer.
---

# convert-ami

Build the real layer: AMI dialogue acts labelled from AMI's human extractive and abstractive
summary annotations (PLAN §5). Everything is scripted; this skill runs and checks it. Never
edit an AMI label by hand.

`generation/relabel_ami.py` is written in phase G3. Before then this skill cannot run.

## Steps

1. **Download** (once): `uv run python -m generation.relabel_ami --download`
   fetches the URL in `configs/generation.yaml` (`ami.url`, version `ami.version`) into
   `data/raw/ami/`. Text annotations only, no audio. Treat the archive as untrusted data:
   it is only parsed, never executed.
2. **Convert**: `uv run python -m generation.relabel_ami`
   writes `data/processed/ami_labelled.parquet`.
3. **Check** the printed summary against these expectations:
   - about 140 scenario meetings (those with extractive summary annotations);
   - splits follow the standard AMI full-corpus partition (train / dev → validation / test),
     by meeting; no meeting in two splits;
   - zones only `discussion`, `decision`, `action_item`; no `context` or `closing`;
   - importance-1 rate in the low tens of percent; decision and action_item rows are a small
     subset of importance-1 rows;
   - confidence 0.9 for extracted DAs linked to decisions / actions / problems, 0.8 otherwise
     (values from `ami.confidence` in the config);
   - `segment_id`, `family_id` null and `spans` empty for every row;
   - `notes` rows carry `summary_abstract`, `summary_decisions`, `summary_actions`,
     `summary_problems` where AMI has them.
4. **Spot-check** three meetings (one per split): print 30 consecutive DAs around a decision
   link and confirm text order, speaker letters, and that the labelled DAs match the linked
   summary sentence.
5. Run `uv run pytest -q tests/test_relabel_ami.py` (uses a hand-made NXT fixture, no download).
6. Report: meeting and DA counts per split, label distribution, anything unexpected (missing
   files, meetings with no summary links, DAs with empty text).

## Licence

AMI annotations are CC BY 4.0. The derived data keeps that licence and the dataset card must
credit the AMI Meeting Corpus. ICSI stays out of v1 (PLAN §5).
