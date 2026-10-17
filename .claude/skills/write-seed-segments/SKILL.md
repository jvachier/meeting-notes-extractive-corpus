---
name: write-seed-segments
description: Write one batch of labelled seed segments (≈10 families × 2 variants) for one zone and meeting type of the meeting-notes-extractive corpus, following PLAN §2.1 and §3.4, then validate it. Use when asked to write, add or fill seed segments, a seed batch, or a coverage cell.
---

# write-seed-segments

Write one batch of seed segments by hand. You are the labeller: every label you write goes
into the dataset unchanged, so labelling care matters more than volume.

## Inputs

- `zone`: context | discussion | decision | action_item | closing
- `meeting_type`: quick_sync | standup | project_review | stakeholder_update | retrospective
- optional `domains`: if not given, pick the domains with the largest shortfall in that
  (zone, meeting type) row.

## What one batch is

A batch is about **10 families (20 segments)** for one (zone, meeting type) row, spread over a
few domains so each touched cell reaches its target in `seeds/coverage.yaml`. Write one file
per cell: `seeds/segments/<zone>/<meeting_type>__<domain>__<batch>.jsonl`, where `<batch>` is
the next free `bNN` for that cell. All files of a batch are committed together, in one commit
per batch, when the user approves.

## Steps

1. Read `PLAN.md` §2.1 and §3, then `generation/prompts/<meeting_type>.txt`.
2. Run `uv run python -m processing.validate_seeds` and read the coverage line for the zone.
   List existing files in `seeds/segments/<zone>/` for this meeting type and skim them, so the
   new families use different scenarios, openings and wording.
3. Plan the batch before writing: for each family, one line with domain, scenario, speaker
   count and which sentences will be important. Vary length (within 2–12 sentences),
   speaker count, the position of the important sentence, and how it is phrased.
4. Write each family as `-v1` then `-v2`, one JSON object per line (format below).
5. Run `uv run python -m processing.validate_seeds`. Fix every error. Read every warning
   about your files and fix it unless it is clearly intended.
6. Report: files written, families per cell, importance-1 rate in the batch, and any
   labelling call you were unsure about.

## Format (PLAN §2.1)

```json
{"segment_id": "dec-project_review-eng-003-v1", "family_id": "dec-project_review-eng-003", "primary_zone": "decision", "meeting_types": ["project_review", "stakeholder_update"], "domain": "engineering", "hard_negative": false, "hn_category": null, "confidence": 1.0, "n_speakers": 3, "sentences": [{"speaker": "S1", "text": "…", "importance": 0, "zone": "discussion"}]}
```

- `family_id` = `<prefix>-<meeting_type>-<domain_abbr>-<nnn>`. Prefixes: ctx, disc, dec, act,
  clo. Domain abbreviations: eng, prod, sales, mkt, sup, hr, fin, res. `nnn` is the next free
  number in that cell (three digits). `segment_id` = `<family_id>-v1`, `-v2`.
- `meeting_types` must include the file's meeting type; add other types where the segment
  fits naturally.
- `confidence`: 1.0 for clear segments; 0.8–0.9 if a label is genuinely debatable.
- Speakers are `S1…Sn`, all used, no gaps. In context and closing segments S1 is the chair.

## Slots (PLAN §3.2)

- Use `{PERSON} {TEAM} {PROJECT} {PRODUCT} {CLIENT} {TOOL} {DATE_REL} {DATE_ABS} {NUMBER}
  {PERCENT} {MONEY} {METRIC}` instead of any concrete name, company, product, date or figure.
  Each slot has **one value per meeting**: `{PROJECT}` is the same project everywhere. For a
  second, different value use an indexed slot, `{PROJECT_2}` … `{PROJECT_9}` (always distinct
  from `{PROJECT}`). Use indexes sparingly: most segments need none.
- Every value has a fixed form (see `generation/lexicons/README.md`). Write around it:
  - `{DATE_REL}` only after "by", "before" or "until" ("by {DATE_REL}", never "ships {DATE_REL}").
  - No hedge before `{NUMBER}`, `{MONEY}` or `{PERCENT}`: values may already contain
    "about"/"roughly", so "about {MONEY}" can render as "about about…".
  - `{NUMBER}` is a count of 2 or more: "{NUMBER} tickets", never "{NUMBER} ticket". Write
    durations and small specific quantities literally ("nine hours", "two people").
  - Lexicons are not domain-specific. Write domain-specific terms literally ("first response
    time", "IT", "engineering", "the applicant tracking system") and keep `{METRIC}`,
    `{TOOL}` and `{TEAM}` for sentences where *any* value fits. Test: would "the payroll
    team" or "the lab notebook" still make sense here? If not, write it literally.
  - Dates must not contradict the text around them ("after year-end, on {DATE_ABS}" can
    render as "after year-end, on May 3"). Let the slot carry the date alone.
- Before validating, render a few segments with real lexicon values and read them.
  - When the sentence constrains the size of a number ("uptime", "out of forty seats", "with
    a team of three"), write the number literally: lexicon values are random magnitudes.
  - Quantities and dates are redrawn for every segment, entities are fixed per meeting
    (`composition.segment_scoped_slots`), so an echo inside the same segment repeats the
    same value, but two segments never share a number by accident.
- Refer to participants with `{S1}`, `{S2}`… never with `{PERSON}`. A speaker never names
  themselves (`S2` saying "{S2} will…" renders as third-person self-reference): let someone
  else state their commitment, or use the first person.
- **No slot may predict importance on its own.** Numbers, dates and participant names must
  also appear in importance-0 sentences: echoes ("Wait, {PERCENT}?"), confirmations
  ("{DATE_REL} works for me."), and people addressing each other ("Thanks, {S2}."). Aim for
  roughly a third of slot-bearing sentences at importance 0; the quality report's shortcut
  probe flags words and slots that are almost always importance 1. `{PERSON}` is someone not
  in the meeting.
- No real people, companies, brands or products, not even in passing.
- In `action_item` sentences with importance 1, mark the owner and the deadline:
  `{S2|owner}`, `{TEAM|owner}`, `{PERSON|owner}`, `{DATE_REL|deadline}`, `{DATE_ABS|deadline}`.
  Roles are allowed nowhere else. A first-person commitment ("I'll do it") has no owner slot;
  that is allowed but `validate_seeds` warns, so prefer having the owner named in the
  sentence, e.g. the chair saying "Okay, {S2|owner} takes the export, done by
  {DATE_REL|deadline}." Keep a few first-person commitments per batch for realism.

## Labelling rules

**importance = 1** if a careful note-taker would put this sentence in the notes:
- the final statement of a decision;
- an action item with an owner (and deadline if given);
- a key fact: a risk, blocker, number, date, dependency, status change, finding;
- the meeting's purpose (context) or the next meeting / review date (closing).

**importance = 0** for everything else:
- agreement and acknowledgements ("Works for me.", "Sounds good.", "Agreed.");
- framing, hedging, rhetorical or clarifying questions, jokes, venting without content;
- restatements: if something is said twice, only the first clear statement is 1;
- routine status with no news; recaps of decisions already made.

**zone** is the function of the sentence inside its segment:
- In a decision segment, the deliberation is `discussion`; the decision sentence and the
  reactions right after it are `decision` (reactions with importance 0).
- In an action_item segment, the lead-in is `discussion`; the commitment and its
  confirmations are `action_item`.
- context and closing segments use `context` / `closing` for every sentence.
- A segment's `primary_zone` must appear on at least one sentence; decision and action_item
  segments need at least one importance-1 sentence in that zone.

**Discussion target:** about 30% of discussion sentences are importance 1. Every discussion
family should mix important facts with chatter.

## Variants

`-v2` is a real paraphrase of `-v1`: same situation and same important content (same number
of importance-1 sentences per zone), but different wording, sentence order and speaker
distribution, and possibly a different number of chatter sentences. Not a slot swap and not
a synonym swap. `validate_seeds` warns when two variants share too many word 3-grams.

## Text

- One sentence per entry, cased and punctuated. Split long turns into consecutive entries of
  the same speaker.
- Spoken style: contractions, "okay", "so", false starts are fine. No stage directions.
