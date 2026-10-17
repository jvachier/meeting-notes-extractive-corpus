# CLAUDE.md

Builds and publishes **meeting-notes-extractive-v1**: meeting transcripts where every sentence
is labelled with an importance flag (keep/discard) and a zone (context, discussion, decision,
action_item, closing).

How the data is made:
- **Claude Code writes the seeds once**: a bank of short labelled meeting segments, the hard
  negatives, and the slot vocabularies (lexicons).
- **Python builds everything else** by composing those seeds into full meetings. No GPU, no API
  and no model runs in the pipeline.
- **A real layer** comes from the AMI corpus's human annotations, converted by script.

`PLAN.md` is the source of truth for the schema, phases, volumes, splits and checks. Read it
before writing code or seeds. If something here conflicts with PLAN.md, ask.

This repo is fully independent of `Note_taker` (the model repo). Never import from it, copy its
tokenizer, or depend on its configs. The only contract between the two is the published dataset.

## How to work
- Work **one phase at a time**, following "Order of work" in PLAN.md.
- At the end of each phase, run the tests, lint and `validate`, then stop and report:
  - what was built
  - the dataset statistics
  - any deviation from PLAN.md

  Do not start the next phase without a go-ahead.
- Any deviation from PLAN.md (schema, volumes, labels, splits, licence) needs my approval first.
- Small, readable modules with type hints and docstrings. Every new module gets tests.

## Hard rules
- **No model in the pipeline.** No LLM calls, no API keys, no torch/tensorflow/transformers.
  The only exception is seed writing, which Claude Code does by hand in a session, following
  `generation/prompts/`.
- **Labels come from seeds or human annotation only.** Python never invents or changes a
  label. It only carries labels through composition, slot filling and noise.
- **Splits are by segment family, never by generated meeting.** A test family's text must
  never appear in a train meeting. `validate` enforces this.
- **Deterministic generation.** Every script takes `--seed` (default from
  `configs/generation.yaml`) and uses its own `random.Random(seed)` or `numpy.random.Generator`.
  The same seed must give byte-identical Parquet.
- **No real people or companies** in seeds. Names, companies and products come from lexicons
  of invented or very common values.
- **Licences:** synthetic data is CC BY 4.0. AMI-derived data is CC BY 4.0 with attribution.
  ICSI is excluded until the licence of its summary annotations is verified (PLAN §6).

## Data
- `seeds/` and `generation/lexicons/` are **source** and are committed.
- `data/` (raw downloads, generated and processed Parquet) is git-ignored. Never commit it.
- Keep local data under ~1 GB.
- Every volume, ratio, probability and length range lives in `configs/generation.yaml`.
  None are hard-coded in scripts.

## Stack
- Python 3.11+, `uv`, pandas + pyarrow, pydantic (schemas), lxml (AMI NXT XML), pyyaml.
- Publishing: `huggingface_hub`, `kaggle` CLI.
- Tests: pytest. Lint: ruff.

## Commands
- Environment: `uv sync`
- Tests: `uv run pytest -q`
- Lint/format: `uv run ruff check . --fix && uv run ruff format .`
- Seed validation: `uv run python -m processing.validate_seeds`
- A phase is done only when tests pass, lint is clean and validation passes.

## Skills (`.claude/skills/<name>/SKILL.md` + `explanation.md`)
- `write-seed-segments`: write one batch of seed segments for a coverage cell (PLAN §3.4).
- `write-hard-negatives`: write one batch of hard-negative segments (PLAN §3.5).
- `convert-ami`: run and check the AMI conversion (PLAN §5).
