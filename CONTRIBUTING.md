# Contributing

Thanks for your interest in **meeting-notes-extractive-v1**. This repository builds a
labelled dataset, so the bar for contributions is label quality and reproducibility, not
just working code.

## Ground rules

- [PLAN.md](PLAN.md) is the source of truth for the schema, volumes, splits and checks.
  Open an issue before proposing any change to it.
- **No model in the pipeline.** No LLM or API calls and no ML frameworks in generation or
  processing code.
- **Labels come only from seeds or human annotation.** Code may carry labels through
  composition, slot filling and noise, but must never create or change them.
- **Splits are by segment family.** Text from a test family must never reach a train meeting.
- **Determinism.** Every script takes `--seed` and uses its own seeded generator. The same
  seed must produce byte-identical Parquet files.
- **No real people, companies, products or brands** in seeds or lexicons.
- Never commit anything under `data/`.

## Setup

Requirements: Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pre-commit install
```

## Before opening a pull request

```bash
uv run pytest -q
uv run ruff check . --fix && uv run ruff format .
uv run python -m processing.validate_seeds
```

All three must pass. CI runs the same checks, plus pre-commit, on Python 3.11 and 3.13.

## Code contributions

- Keep modules small and readable, with type hints and docstrings.
- Every new module gets tests in `tests/`. Tests run offline: no downloads.
- Put every volume, ratio, probability and length range in `configs/generation.yaml`, never
  in code.
- Follow the existing style (ruff, line length 100).

## Seed contributions

Seeds are dataset source. Each label you write goes into the dataset unchanged.

1. Follow the procedure in
   [`.claude/skills/write-seed-segments/SKILL.md`](.claude/skills/write-seed-segments/SKILL.md)
   (or [`write-hard-negatives`](.claude/skills/write-hard-negatives/SKILL.md)) and the brief
   for the meeting type in [`generation/prompts/`](generation/prompts/).
2. Use slots for every name, company, product, date and figure, in the forms described in
   [`generation/lexicons/README.md`](generation/lexicons/README.md).
3. Run `validate_seeds`, fix every error, and read every warning on your files.
4. Submit **one batch per commit**, so a bad batch can be reverted alone. In the PR
   description, list the cells covered and any labelling calls you were unsure about.

Lexicon changes must keep each lexicon's value form, add no duplicates, and use only invented
or very common values.

## Commits and pull requests

- Use short, imperative commit subjects ("Add decision seeds for project_review").
- Keep pull requests focused: one phase, one batch, or one fix.
- Changes to `seeds/`, `generation/lexicons/`, `generation/prompts/` or `PLAN.md` need review
  from the code owner (see [`.github/CODEOWNERS`](.github/CODEOWNERS)).

## Licence

By contributing, you agree that your code contributions are licensed under the MIT licence
([LICENSE](LICENSE)) and your data contributions (seeds, lexicons, prompts) under CC BY 4.0
([LICENSE-DATA](LICENSE-DATA)).
