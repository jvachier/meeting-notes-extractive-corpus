## Summary

<!-- What does this PR change, and why? Link the issue or PLAN.md section if there is one. -->

## Type of change

- [ ] Code (pipeline, validation, tooling)
- [ ] Seeds (segments, hard negatives, fillers): one batch per PR
- [ ] Lexicons or prompts
- [ ] Config (`configs/generation*.yaml`)
- [ ] Docs only

## Seed batch details (seed PRs only)

- Cells covered (zone / meeting type / domains, or hard-negative category):
- Families added:
- Labelling calls I was unsure about:

## Checklist

- [ ] `uv run pytest -q` passes
- [ ] `uv run ruff check . && uv run ruff format --check .` is clean
- [ ] `uv run python -m processing.validate_seeds` passes, and I read every warning on my files
- [ ] No real people, companies, products or brands; all names and figures use slots
- [ ] No label is created or changed by code (labels come from seeds or human annotation only)
- [ ] Nothing under `data/` is committed
- [ ] Any deviation from PLAN.md is described above and approved
- [ ] If the dataset output changes: rebuilt with `uv run python -m processing.build` and checked `data/processed/report.md`
