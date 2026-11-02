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

You need a Mac or PC with 16 GB of memory (8 GB is not enough), a free Hugging Face account with
[Google's Gemma license](https://huggingface.co/google/gemma-2-2b-it) accepted, and about 5 GB of download the first
time. Then, one at a time:

```bash
.venv/bin/hf auth login
bash scripts/run-evidence.sh my-smoke --eta "2-5 min (107 s on an M5)" --cost "0 (local)" -- .venv/bin/python -u outcome_leak.py smoke
bash scripts/run-evidence.sh my-patching --eta "2-5 min (22 s on an M5)" --cost "0 (local)" -- .venv/bin/python -u outcome_leak.py patching my-patching
bash scripts/run-evidence.sh my-removal --eta "~7 min (393 s on an M5)" --cost "0 (local)" -- .venv/bin/python -u outcome_leak.py removal-sturdy my-removal
bash scripts/run-evidence.sh my-new-cases --eta "3-5 min (301 s on an M5)" --cost "0 (local)" -- .venv/bin/python -u outcome_leak.py removed-cases my-new-cases
```

`.venv/bin/python outcome_leak.py smoke` alone runs the same; the wrapper keeps the whole output as
`evidence/raw-logs/my-smoke.log`, with the time estimate and the real elapsed time, and never overwrites a log. What
each step does and the log line to expect after it: [docs/tutorial.md](docs/tutorial.md).
