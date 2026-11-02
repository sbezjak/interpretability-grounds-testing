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

Running the model itself needs 16 GB and Google's Gemma license. Setup: [docs/running-the-model.md](docs/running-the-model.md).
Then the four runs in order, with the log line to expect after each: [docs/tutorial.md](docs/tutorial.md).
