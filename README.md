# interpretability-grounds-testing

Why does a small AI model blame a driver more when, by pure luck, a child gets hurt? This repo looks inside
gemma-2-2b-it to find out - reading the numbers inside the model while it works, then changing them to see what
causes what. Everything runs on a laptop for $0.

## Results

- The model gives Dana's careless choice **2.72 more blame** on a 0-9 scale when it ends in harm.
- The outcome reaches the answer **around layer 15** of 26, and removing one direction cuts most of the leak (70%).
- On two new stories the fix **worked once, and once the leak flipped the other way**, at a small cost. Dana's numbers
  picked the layer, so Dana isn't a clean test.

Every number with its log line:
[the audit report](https://sbezjak.github.io/interpretability-grounds-testing/reports/dana-outcome-leak-gemma-2-2b-it-audit.html).
How it was done, in plain words: [notes.md](notes.md).

## Quick start (no model)

You need [uv](https://docs.astral.sh/uv/). The install takes a few minutes (it downloads torch); then each command
takes a few seconds:

```bash
git clone https://github.com/sbezjak/interpretability-grounds-testing.git
cd interpretability-grounds-testing
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python test_outcome_leak.py          # the measurement logic on hand-made numbers
.venv/bin/python outcome_leak.py intervals     # 95% intervals for the sampled test, read from its log
```

## Run the model (16 GB, about 15 minutes)

Needs 16 GB of memory (8 GB is not enough), a free Hugging Face account with
[Gemma's license](https://huggingface.co/google/gemma-2-2b-it) accepted, and a 5 GB download the first time.
Four runs, one at a time, then one page:

```bash
.venv/bin/hf auth login     # once, so the model can download
bash scripts/run-evidence.sh my-smoke     --eta "2-5 min" --cost 0 -- .venv/bin/python -u outcome_leak.py smoke
bash scripts/run-evidence.sh my-patching  --eta "2-5 min" --cost 0 -- .venv/bin/python -u outcome_leak.py patching my-patching
bash scripts/run-evidence.sh my-removal   --eta "7 min"   --cost 0 -- .venv/bin/python -u outcome_leak.py removal-sturdy my-removal
bash scripts/run-evidence.sh my-new-cases --eta "5 min"   --cost 0 -- .venv/bin/python -u outcome_leak.py removed-cases my-new-cases
.venv/bin/python outcome_leak.py my-report     # no model: your numbers next to the repo's, on one page
```

What each run does: **smoke** loads the model and checks the swap works. **patching** measures the leak and finds
the layer it enters. **removal** takes one direction out after each layer and reads Dana, the control and 10 random
directions. **new-cases** tries that removal on three stories the direction never saw. The wrapper saves every run
whole to `evidence/raw-logs/<name>.log` and never overwrites one. The log line to expect after each step, and how
to remove the model afterwards: [docs/tutorial.md](docs/tutorial.md).
