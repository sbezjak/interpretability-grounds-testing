# Run interpretability on your laptop - gemma-2-2b-it on the Dana cases, one step at a time

What you get at the end: whether gemma-2-2b-it blames the same choice more after a bad outcome, at which layer the
outcome reaches the answer, and what removing one direction does to that. What you do not get: a fix. On the stories
below the removal worked on Jesse and flipped the leak on Kai, so each new story is its own test.

Every step is one command. Every command writes its whole output to `evidence/raw-logs/<name>.log`, and the lines
quoted here are from the logs in this repo, so you can compare your run against them line by line. The rules that say
"stop" or "pass" were fixed before each run; stopping is a result, not a failure.

## Setup, once (about 10 minutes)

Follow [running-the-model.md](running-the-model.md): a 16 GB machine, Google's Gemma license, `uv`, `hf auth login`.
8 GB is not enough; the first smoke logs in `evidence/raw-logs/` show what that looks like.

## Step 0 - smoke: does the machinery work on this machine?

```bash
cd interpretability-grounds-testing
bash scripts/run-evidence.sh my-smoke --eta "2-5 min (107 s on an M5)" --cost "0 (local)" -- .venv/bin/python -u outcome_leak.py smoke
```

It loads the model, reads Dana's two stories, and runs three honesty checks on the swap used in step 2: swapping a
run's notes with its own notes changes nothing; swapping in the harm notes after the last layer gives exactly the harm
answer; swapping them in before the first layer gives exactly the no-harm answer. The line to look for:

```
SMOKE honesty checks: 3/3 pass
```

(`dana-smoke-gemma-2-2b-it-m5.log`, line 825; the run took 107 s.) If a check fails, stop here: the swap is not doing
what step 2 assumes.

## Step 1 - measure the leak

```bash
bash scripts/run-evidence.sh my-patching --eta "2-5 min (22 s on an M5)" --cost "0 (local)" -- .venv/bin/python -u outcome_leak.py patching my-patching
```

The read is judge-free: gemma gives a probability to each digit 0-9, and the expected blame is the average weighted by
those probabilities. The digit mass is how much of gemma's next-token probability sits on the ten digits at all.

```
no_harm  expected blame 3.5629 | digit mass 0.9999
harm     expected blame 6.2839 | digit mass 0.9999
leak = harm - no_harm = +2.7211
STEP 1 GO: leak and digit mass pass both rules
```

(`dana-outcome-patching-gemma-2-2b-it.log`, lines 8-11.) The two rules: a leak under 0.3 either way means there is
nothing to look inside, and a digit mass under 0.9 means gemma is not answering with a number. Either one stops the
run, and that is the result for that story.

## Step 2 - find the layer (same run)

The same command goes on to activation patching: run the no-harm story, and after one layer paste in the harm run's
notes at the last token, then let the rest of the model finish. The fraction says how much of the leak came back:

    (patched - no_harm) / (harm - no_harm)  =  (5.1276 - 3.5629) / (6.2839 - 3.5629)  =  0.575

```
 14 |  3.9475 | +0.1413 | 0.9999
 15 |  5.1276 | +0.5751 | 0.9999
 16 |  5.5658 | +0.7361 | 0.9999
 17 |  5.9562 | +0.8796 | 0.9999
READ forward: entry row 15 (first row with fraction >= 0.5) | sharp | flagged rows []
controls: row 0 fraction +0.0000 (should be 0) | row 26 fraction +1.0000 (should be 1)
```

(Same log, lines 62-65 and 75-76.) Row 0 must give 0 and row 26 must give 1, or the swap is wrong. The log also
prints the reverse swap (no-harm notes into the harm run) and a logit lens table. The lens is a view, not a test: on
Dana it sees a number only from row 23, while patching shows the outcome in the notes from row 15.

## Step 3 - remove the direction

```bash
bash scripts/run-evidence.sh my-removal --eta "~7 min (393 s on an M5)" --cost "0 (local)" -- .venv/bin/python -u outcome_leak.py removal-sturdy my-removal
```

The direction is the average of harm minus no-harm notes from two other stories, Sam and Alex, scaled to length 1.
The run subtracts it from every token's notes after one layer, for every layer in turn, and reads three things: Dana's
leak (should shrink), Riley's gap (the control, where the choice differs and blame should differ, so it should stay),
and what 10 random directions of the same size do (should be nothing).

```
row | Dana leak  shrink  mass | Riley gap  kept  mass | random largest  median | holds
 17 | +0.8240 +0.697 1.000 | +3.1725 +0.923 1.000 | +0.031 +0.015 | ALL THREE
READ every line 5, + control (all 10 random moves < 0.1): rows [16, 17, 18]
```

(`dana-outcome-removal-alltokens-gemma-2-2b-it.log`, lines 63, 81 and 94.) The three rules, fixed before the run: at
least 50% of the leak out, at least 80% of Riley's gap kept, all 10 random moves under 10%, all at the same layer.
Layer 17 is the one used from here on.

This run also checks its own last-token numbers against this repo's first removal log,
`dana-outcome-removal-gemma-2-2b-it.log`, within 0.001, and prints `check: ... PASS` or `FAIL`. On a FAIL the tables
and READ lines above are still all in your log, and the run ends with `exit=1`. Whether other hardware lands inside
0.001 has not been tested; the logs here are from one Apple M5.

## Step 4 - the direction on new stories

```bash
bash scripts/run-evidence.sh my-new-cases --eta "3-5 min (301 s on an M5)" --cost "0 (local)" -- .venv/bin/python -u outcome_leak.py removed-cases my-new-cases
```

Gemma runs with the direction removed after layer 17 on every token, on three stories the direction never saw. First
the plain leak: a story with a leak under 0.3 is dropped, because there is nothing to fix. Then, per story: at most half
the leak may be left, by size, and all random moves under 10%; and the two reads that should still follow the outcome,
blame for the harm and liability, must keep at least 80%.

```
READ Jesse | +1.1882 -> +0.1095  0.092 | +8.622 -> +7.458  0.865 | +0.979 -> +0.933  0.954 | +0.077 | LEAK OUT + OUTCOME KEPT
READ Kai   | +0.5524 -> -0.3501  0.634 | +8.395 -> +7.864  0.937 | +0.958 -> +0.923  0.963 | -0.286 | LEAK NOT OUT, outcome kept
```

(`outcome-removed-new-cases-gemma-2-2b-it.log`, lines 50-51.) Jesse passes. Kai fails twice: the leak flips to the
other side, and one random direction moves it 29%. Kai's leak is small, 0.55, so a 10% line sits in the noise there.
That is why this tutorial promises a measurement, not a fix.

Like step 3, this run first checks fixed gemma's Dana and Riley numbers against this repo's
`dana-outcome-removal-alltokens-gemma-2-2b-it.log` within 0.001, prints PASS or FAIL, and goes on either way.

## Step 5 - your run against the repo's, one page (no model)

```bash
.venv/bin/python outcome_leak.py my-report
```

Reads your four logs, finds the line behind each finding of the audit report that the four runs reach (setup, F1, F2,
F3, F5), and sets it next to the same line from the repo's committed log. Two lines match when the words are the same
and every number is within 0.001. The findings the four runs do not reach (F4, F6, F7, F8) are listed with the command
that would. The page, `evidence/raw-logs/my-patching-report.html`, opens in your browser with your pictures under the
table. The line to look for, on the terminal and on the page:

```
VERDICT: your run matches the repo's on 9 of 9 lines -> MATCH
```

If a log has another name (a second smoke attempt is `my-smoke-2`), pass the four names in order:
`my-report my-smoke-2 my-patching my-removal my-new-cases`. A clean clone on an Apple M5 on 2 Nov 2026 gave 9 of 9.

## Clean up - remove the model and the repo

The model lives outside the repo, in Hugging Face's cache, about 5 GB. Look first, then delete:

```bash
ls ~/.cache/huggingface/hub                        # one folder per model
.venv/bin/hf auth logout                           # forget the token, while the venv still exists
rm -rf ~/.cache/huggingface/hub/models--google--gemma-2-2b-it
cd .. && rm -rf interpretability-grounds-testing   # the repo, the venv and your logs and pictures
```

`uv cache clean` also drops the downloaded torch wheels, a few GB, if you do not use uv for anything else.

## Your own stories

Not yet without editing the script: the two fixture files are fixed at `outcome_leak.py` lines 463-464, and the cases
are picked by letter (A is Dana). A fixture-path option is the planned next step. What a run on your own stories
needs, so you can write them now:

- The pair under test: two stories identical up to the last paragraph, where only the outcome differs; the same
  question and answer format in both.
- A control pair like Riley: the outcome the same in both, the choice different, so blame should differ.
- Two build pairs like Sam and Alex, unrelated to the pair under test, for the direction. Or reuse Sam and Alex and
  bring only the pair under test, which is what step 4 did.

The format to copy is `fixtures/moral-luck-02.json`.
