---
name: write-hard-negatives
description: Write one batch of hard-negative seed segments (sentences that look like decisions or action items but are not) for one category of the meeting-notes-extractive corpus, following PLAN §3.5, then validate it. Use when asked to write hard negatives or a hard-negative batch.
---

# write-hard-negatives

Write segments whose key sentences *look* like a decision or an action item but are not one.
They teach the model that trigger words ("we could", "someone should", "decided") are not
enough. Labels go into the dataset unchanged.

Read `.claude/skills/write-seed-segments/SKILL.md` first: the format, slot, variant and text
rules there all apply. This file lists only what differs.

## Inputs

- `category`: one of the seven below.
- optional batch size: default about 10 families (20 segments).

## Files and ids

- File: `seeds/hard_negatives/<category>__<batch>.jsonl`, `<batch>` = next free `bNN`.
- `family_id` = `hn-<category>-<nnn>` (next free three-digit number for the category);
  `segment_id` = `<family_id>-v1`, `-v2`.
- `hard_negative: true`, `hn_category: <category>`.
- `confidence` between **0.6 and 0.8** (these are ambiguous on purpose): 0.6 for the most
  borderline, 0.8 for clear-cut ones.
- Spread `domain` over all eight domains across the category, and list every meeting type in
  `meeting_types` where the segment fits.

## Categories (PLAN §3.5)

| Category | The trap | Labels |
|---|---|---|
| rejected_proposal | A proposal phrased like a decision, then turned down. | All 0, zone discussion. |
| tentative_idea | "Maybe someone should…", "we might want to…" with no commitment. | All 0, discussion. |
| ownerless_task | "This needs fixing before launch." No owner, no date, nobody takes it. | All 0, discussion (not action_item). |
| reversed_decision | A decision is made, then overturned in the same segment. | First decision: zone decision, importance 0. Final decision: decision, importance 1. `primary_zone: decision`. |
| reported_elsewhere | "Finance already decided that last week." Status about a decision made elsewhere. | 0, discussion. |
| hypothetical | "If the client signs, we'd need two more people." | 0, discussion. |
| question_not_commitment | "Should we do a retro on this?" and nobody answers or commits. | 0, discussion. |

Except for reversed_decision, `primary_zone` is `discussion` and every sentence is
importance 0. `validate_seeds` warns if a non-reversed hard negative has an importance-1
sentence; only keep one if it is a genuine key fact unrelated to the trap, and say so in the
report.

## Writing the trap well

- Use the vocabulary of real decisions and action items: "let's", "we'll", "decided",
  "by Friday", "I think we should", names and dates. That is what makes them hard.
- Keep the trap sentence central, with 1–4 surrounding sentences that make its status clear
  to a careful reader (the rejection, the silence, the "if").
- Vary the surface form across families: do not start every tentative idea with "Maybe".

## Steps

1. Run `uv run python -m processing.validate_seeds` and read the `hard_negative_families`
   line for the category.
2. Skim existing files for the category so new families use new scenarios.
3. Write the batch, then validate. Fix every error and every warning on your files.
4. Report: file, families written, domain spread, and any borderline call.
