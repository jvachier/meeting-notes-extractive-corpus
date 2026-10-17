# convert-ami: why it works this way

**Why AMI.** The synthetic layer is composed from seeds written by one author, so its style is
narrow. AMI gives real, messy, multi-party speech with human labels, and the AMI test split is
the measurement that matters (PLAN §4.3, §7). If synthetic training transfers to AMI, the
synthetic data is doing its job.

**Where the labels come from.** AMI annotators wrote abstractive summaries in four sections
(abstract, decisions, actions, problems) and then linked dialogue acts (the extractive
summary) to summary sentences. A DA that was extracted is important; the section of the
summary sentence it links to gives its zone. No label is invented: unlinked or abstract-only
DAs are `discussion`, and AMI never gets `context` or `closing` because nobody annotated them.

**Why confidence below 1.** Agreement between AMI summary annotators is modest (κ ≈ 0.48), so
AMI labels are less certain than seed labels. Linked DAs (0.9) are a bit more reliable than the
rest (0.8) because two annotation layers agree on them.

**Why the standard partition.** Using AMI's own full-corpus train/dev/test split makes results
comparable with published work and keeps whole meetings in one split.

**Why a fixture test.** The NXT format spreads one meeting over many XML files (words, DAs,
extractive and abstractive summaries, links). A tiny hand-made fixture tests the mapping
without downloading the corpus, so CI stays offline and fast.
