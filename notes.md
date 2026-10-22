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
