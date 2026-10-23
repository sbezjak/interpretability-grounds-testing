# Notes - what interpretability is, and how this repo does it

Written so anyone can get in: no background in machine learning needed, one concrete case all the way through.

## The question

My other repos test AI from the outside. `theory-grounds-testing` and the published `judge-outcome-bias-probe` change
one thing in the prompt and watch whether the answer moves. That's how I found the outcome leak: the same careless
choice gets more blame when, by luck, it ends in harm.

From the outside I can see *that* the outcome moves the answer. I can't see *where* or *how*. Interpretability opens
the model up and looks at the numbers inside while it works - and then changes them, to see what causes what.

Same question as the rest of my work - is the verdict governed by the grounds it should rest on? - one level down:
behavior (outside) -> mechanism (inside).

## The case: Dana

Two prompts, identical except the last paragraph:

- Dana reads a phone for 3 seconds while driving past parked cars. Same street, same risk, same choice in both.
- **No harm:** nothing crossed the road, no one was hurt.
- **Harm:** a 7-year-old ran out, Dana braked too late, the child was seriously injured.

How blameworthy is Dana's *choice*? The choice is identical, so a governed answer gives the same number both times.
Frontier models and qwen gave more blame after harm (`theory-grounds-testing`, moral-luck-01 and 02).

## How a model works inside

1. **Tokens.** The prompt is cut into tokens - roughly word pieces.
2. **Vectors.** Each token becomes a list of numbers (2,304 of them in gemma-2-2b). Think of it as that token's
   running notes.
3. **Layers.** The model has 26 layers. Each layer reads all the notes and writes additions back into them. This
   running set of notes is called the **residual stream**.
4. **The answer.** After the last layer, the notes at the final token turn into a probability for every possible next
   token. For Dana, that's how likely the model is to write each blame number.

## Step 1 - measure the leak exactly

No sampling, no repeats. Read the probability the model gives each number and take the average weighted by
probability - the *expected blame*. Do it for both prompts.

**Leak = expected blame (harm) - expected blame (no harm).**

Exact, judge-free, and the same every run. If it's near zero, the model doesn't leak and there's nothing to look
inside - stop there.

## Step 2 - find where it enters: activation patching

It's a bisect, the QA kind.

- Run the **no-harm** prompt.
- At one layer - say layer 10 - swap in the notes from the **harm** run, at the final token.
- Let the rest of the model run and read the expected blame.

If blame jumps up, then "the child was hurt", as far as blame is concerned, is carried in those notes at layer 10.
Repeat for every layer and you get a curve: where the outcome reaches the answer.

Like swapping one component from the failing build into the passing build to find where the bug lives.

**Fraction restored** - how much of the leak one swap brought back. Say no harm gives 3.56, harm gives 6.28 (leak
2.72), and no-harm with harm notes at layer 10 gives 5.40:

    (5.40 - 3.56) / (6.28 - 3.56) = 1.84 / 2.72 = 0.68  ->  68% of the leak came back

0 = the swap did nothing, 1 = it brought the whole leak.

### The reverse direction

The swap above asks: are the harm notes here **enough** to raise the blame? The reverse asks the other half: if I take
the harm out here, does the blame **fall**?

- Run the **harm** prompt.
- At one layer, swap in the notes from the **no-harm** run, at the final token.
- Read the expected blame. Same formula, legs swapped - the fraction of the leak removed:

      (patched - harm) / (no_harm - harm)
      blame drops to 4.00:  (4.00 - 6.28) / (3.56 - 6.28) = -2.28 / -2.72 = 0.84  ->  84% of the leak removed

Why both: we only swap the notes at **one token** (the last one). The harm paragraph itself is still in the prompt, on
earlier tokens, and later layers can read it again from there. So a swap can fail not because the outcome isn't in
those notes, but because the model fetches it again a few layers later. Running both directions shows which.

## A view along the way: the logit lens

At the very end, the model does one fixed step: it takes the final notes and turns them into a score for every
possible next token, then into probabilities. Call that step the **reader**. Normally it runs once, after layer 26.

The logit lens runs the same reader on the notes after **every** layer: "if the model had to answer right now, after
layer 14, what blame would it give?" Like printing a variable halfway through a pipeline instead of only the final
output.

A made-up example of what it looks like:

| after layer | no harm | harm |
|---|---|---|
| 5 | 4.5 | 4.5 |
| 14 | 4.1 | 5.0 |
| 26 | 3.56 | 6.28 |

Read: the outcome shows up in the answer from around layer 14 on.

It is a **view, not a proof**. The reader was built for the final notes. Early layers may keep the information in a
form the reader can't read yet, so a flat line early doesn't mean the information isn't there. It only watches - it
changes nothing. Patching is the test that shows cause; the lens is a cheap picture to compare it with.

One honesty check is built in: after layer 26 the lens IS the real answer, so the two must match.

### The squash (gemma's score cap)

Gemma's reader has one extra step: before turning scores into probabilities, it squashes every score so none can go
past 30:

    squashed = 30 x tanh(score / 30)

`tanh` is a curve that is almost a straight line near 0 and flattens out towards 1. So small scores barely change and
big ones get bent down:

| score | squashed |
|---|---|
| 3 | 2.99 |
| 10 | 9.65 |
| 30 | 22.85 |
| 100 | 29.92 |

Why gemma does it: it's a safety rail Google added for training. Without a cap, a model can push one score far above
the rest and become absurdly sure of itself - and huge numbers make training unstable. Two words scored 40 and 30:
without the cap the winner gets 99.995%; squashed (26.10 vs 22.85) it gets 96%. The cap keeps the model from ever
being *that* certain. It is part of how gemma was trained, so it is part of the model - not something we add.

When the model runs normally it does this inside its own code. The logit lens re-does the reader by hand, so it has to
do the squash too - or even its layer-26 reading would not match the real answer.

## Reading the result - same rule as my other repos

- **Strong:** a clean effect you can switch on and off at a specific place, with the model still answering sensibly.
- **Weak:** finding nothing. The information may be spread out, the model may repair it downstream, or I looked in the
  wrong place. A null here is not "the outcome isn't used".
- **The limit:** this shows a mechanism in a model I can open. It says nothing about Claude's or GPT's internals.

## The code, in plain terms

- **PyTorch** does the math. **Hugging Face `transformers`** loads gemma.
- A **hook** is a small function attached to one layer. During a run it can read that layer's output, or replace it.
  That's all patching is: save the harm run's notes, then rerun no-harm with a hook that swaps them in.
- About 100 lines: load the model -> format both prompts with gemma's chat template -> read the number probabilities
  -> save the harm notes per layer -> rerun no-harm once per layer with the hook -> one table, one curve.
- A detail the smoke has to settle: gemma splits numbers into single digits, so "10" is two tokens. Either use a scale
  where every answer is one token (1-9), or handle the two-token case. Decided before the run, not after.

## Why gemma-2-2b-it

- Small enough for an 8 GB Mac at half precision (about 5.2 GB - tight, it may swap).
- **Gemma Scope**: pretrained dictionaries (SAEs) that label what directions in gemma's notes mean. Not needed for
  patching; needed later for eval awareness.
- The model most public interpretability tutorials and tools are built on - good for anyone following along.
- Gated: accept Google's license on Hugging Face and log in with a read token (one time).

## Next: eval awareness, with the same tools

Frontier models often read my test prompts as tests. If a model holds steady only because it knows it's being
watched, the hold is performance, not governance.

Same machinery: contrast "this is a test" prompts with "real use" prompts, find the direction that separates them
(the difference of the averages, or a Gemma Scope feature), turn it down while the model rates Dana. Holds that
survive are real; holds that vanish were test performance. Prior-art check first - this has been done on larger
models.

## Words used here

- **Residual stream** - the running notes per token that every layer reads and writes.
- **Activation** - the numbers in those notes at one layer.
- **Activation patching** - swapping activations from one run into another to see what changes.
- **Hook** - a function attached to a layer that can read or replace its output.
- **SAE (sparse autoencoder)** - a dictionary that splits a layer's notes into labelled features.
- **Expected blame** - the probability-weighted average of the blame numbers.
- **Fraction restored / removed** - how much of the leak one swap brought back (or took away): 0 = none, 1 = all.
- **Logit lens** - running the model's final reader on the notes after every layer, to watch the answer form.
- **Softcap (the squash)** - gemma's cap on scores, 30 x tanh(score / 30), so no score goes past 30.
