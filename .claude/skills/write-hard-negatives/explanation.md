# write-hard-negatives: why it works this way

**The problem they solve.** In clean synthetic data, decisions and action items tend to carry
obvious trigger words. A model, or the keyword rule baseline in PLAN §7.5, can then score well
by spotting "decided" or "will … by Friday" without understanding the exchange. Real meetings
(the AMI test) are full of proposals that die, ideas nobody owns and decisions reported from
elsewhere. Hard negatives put those patterns into training with importance 0.

**Why seven categories.** Each category isolates a different reason a decision-looking sentence
is not one: rejected, tentative, unowned, superseded, not new, conditional, unanswered. Covering
them separately makes it possible to measure, per category, whether the model learned the
distinction.

**Why confidence 0.6–0.8.** These are borderline by design, and human note-takers would not all
agree. A lower confidence lets consumers down-weight them, and it keeps them distinguishable
from the clear seeds (confidence ≥ 0.8, usually 1.0).

**Why reversed_decision has one importance-1 sentence.** The final decision is a real decision
and belongs in the notes. Only the overturned one is the trap. This teaches the model that the
*last* decision in an exchange wins.

**How they are used.** `generate_hard_negatives.py` (G2) builds 500 meetings, each with at least
three hard-negative segments mixed with normal ones, so the traps appear in realistic context.
Hard-negative families are split like normal families, stratified by category.
