# CLAUDE.md

## What this repo is

`interpretability-grounds-testing` (Sara shortens it to "interp") looks inside an open model to see what drives its
verdict. It carries the question of `../theory-grounds-testing` and the published `../judge-outcome-bias-probe` - is
the verdict governed by the grounds it should rest on? - from behavior (outside) to mechanism (inside).

First check: the moral-luck outcome leak (Dana, harm vs no harm) in `gemma-2-2b-it` - does it leak, and at which
layer does the outcome reach the answer (activation patching). Second: eval awareness (does a "this is a test"
direction hold the verdict steady). `notes.md` explains the method plainly; keep it that way - anyone should be able
to get in.

## How work is done here (carried from theory-grounds-testing)

- **Judge-free.** The verdict is read from the model's own number probabilities (expected blame), never from a second
  model's opinion. New measurement logic gets a canned test (no model) before any run.
- **Local only, $0.** M1, 8 GB RAM, about 19 GB disk free (2026-10-22). A bigger model on a rented GPU is a paid step -
  Sara decides.
- **Smoke first, stop rules fixed before the run.** If the model cannot do the instrument (no clear number answer) or
  does not leak, stop - that is a result.
- **Preserve the FULL run log.** Every run goes through `scripts/run-evidence.sh <name> --eta "<est (source)>"
  --cost "<est (source)>" -- <cmd>`. State ETA unprompted. Before stating a finding, quote its receipt line verbatim
  from `evidence/raw-logs/<name>.log`. A log is never regenerated; a replication gets a new dated name.
- **Read the asymmetry.** A clean on/off effect at a specific place is strong; finding nothing is weak (spread out,
  repaired downstream, wrong place). Claims are about this model, never about Claude's internals.
- **An absence claim needs a count** from `scripts/count.py <regex> <paths>`, quoted whole.
- **Thresholds are written down before the run** (e.g. the leak threshold), never chosen after seeing the numbers.
- **Explain before building.** Sara wants to understand each step before it is built; plain words, concrete case first.
- **Smallest solution that solves the problem.** One script, plain hooks, no framework nobody asked for.
- Name committed artifacts by finding / case + model, never by build stage.
- In all prose (docs, comments, commits), the only dash character is a single `-`.
- **Sara is the sole author.** End commit messages at the body - no co-author trailer. Commit or push only when asked.
