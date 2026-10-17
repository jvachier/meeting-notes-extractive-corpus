# write-seed-segments: why it works this way

**What it produces.** The segment bank (PLAN §3.4) is the only place synthetic labels come
from. Python composes meetings from these segments and never invents or edits a label, so the
quality of the whole synthetic layer is the quality of these seeds.

**Why batches of about 10 families per (zone, meeting type) row.** The 580 target families
are spread over 200 cells (5 zones × 5 meeting types × 8 domains), so most cells only need
1–6 families. A batch therefore covers one row of cells, a few domains at a time, with one file
per cell. Ten families is small enough to review in one sitting and to revert as one commit
if it is bad, and large enough that the families in a batch can be made deliberately different
from each other.

**Why families and variants.** Splits are assigned per family (PLAN §4.4). A variant is a
paraphrase written alongside its seed, so it is the only text close enough to leak; keeping it
in the same family keeps test text out of train. Variants also give the model the same labels
with different surface forms, which discourages learning phrasing shortcuts.

**Why sentence-level labels inside a segment.** A decision segment is mostly not decisions:
deliberation and "works for me" lines surround one decisive sentence. Labelling every sentence
teaches the model where in the exchange the important line is, instead of tagging whole
blocks.

**Why slots and roles.** Slots keep names, companies and figures invented and consistent
within a meeting, and they let the same seed appear with different surface values. The
`owner`/`deadline` roles give span supervision for action items at no extra annotation cost.

**Why the checks.** `validate_seeds` catches mechanical problems (schema, ids, file placement,
missing lexicons, variant count) as errors, and labelling smells (action items without an
owner, variants that are too similar, discussion importance far from 30%) as warnings that
need a human look.
