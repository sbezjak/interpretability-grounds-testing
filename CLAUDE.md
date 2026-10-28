# CLAUDE.md

## What this repo is

`interpretability-grounds-testing` (Sara shortens it to "interp") looks inside an open model to see what drives its
verdict. It carries the question of the published `../judge-outcome-bias-probe` - is the verdict governed by the
grounds it should rest on? - from behavior (outside) to mechanism (inside).

Both checks are done: the Dana outcome leak in `gemma-2-2b-it` (found, located at layer ~15, one direction removed
after layer 17 fixes it on 2 of 3 stories, at a small cost) and eval awareness (stopped at its gates). Results:
`README.md` and `reports/`. `notes.md` explains the method plainly; keep it that way - anyone should be able to get
in.

## How work is done here

- **Judge-free.** The verdict is read from the model's own number probabilities (expected blame), never from a second
  model's opinion. New measurement logic gets a canned test (no model) before any run - `test_outcome_leak.py`.
- **Local only, $0.** M1, 8 GB RAM - too small for gemma-2-2b-it, which needs 16 GB. A bigger model on a rented GPU
  is a paid step - Sara decides.
- **Smoke first, rules fixed before the run.** Thresholds and stop rules (e.g. the leak threshold) are written down
  before the run, never chosen after seeing the numbers. If the model cannot do the instrument (no clear number
  answer) or does not leak, stop - that is a result.
- **Preserve the FULL run log.** Every run goes through `scripts/run-evidence.sh <name> --eta "<est (source)>"
  --cost "<est (source)>" -- <cmd>` (it refuses an estimate with no number and never overwrites a log). State ETA
  unprompted. Before stating a finding, quote its receipt line verbatim from `evidence/raw-logs/<name>.log`. A log is
  never regenerated; a replication gets a new dated name.
- **Frozen files.** `scorer.py` (a verbatim copy of judge-outcome-bias-probe's), the fixtures and everything in
  `evidence/raw-logs/` are never edited - flag a problem and leave them.
- **Read the asymmetry.** A clean on/off effect at a specific place is strong; finding nothing is weak (spread out,
  repaired downstream, wrong place). Claims are about this model, never about Claude's internals.
- **Absence words need a count.** Before "only", "the only difference", "nothing else", "not anywhere", "never",
  "every", "all": name what was compared and quote `scripts/count.py <regex> <paths>` whole, or show the whole diff.
  A 0 over ONLY-paths means "not in those paths", never "not anywhere" - rerun on `.`. An errored or empty search is
  no result.
- **Explain before building, beginner pace.** Interp is new to Sara: plain words, a worked number example before
  code, real printouts / shapes, one step at a time. Terminal steps as exact commands, starting with the `cd` into
  the repo.
- **Stop after every run.** Report, go through the results with Sara, and start nothing next without her signal.
- **Smallest solution that solves the problem.** One script, plain hooks, no framework nobody asked for.
- Name committed artifacts by finding / case + model, never by build stage.
- In all prose (docs, comments, commits), the only dash character is a single `-`.
- **Sara is the sole author.** End commit messages at the body - no co-author trailer (`.claude/settings.json` turns
  attribution off, `scripts/guard-coauthor-trailer.sh` blocks it). Commit or push only when asked.
