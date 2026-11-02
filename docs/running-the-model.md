# Running the model

You need a free Hugging Face account with [Google's Gemma license](https://huggingface.co/google/gemma-2-2b-it)
accepted, about 5 GB of download, and a Mac or PC with 16 GB of memory - 8 GB wasn't enough (the first smoke logs
show it). Then:

```bash
.venv/bin/hf auth login
bash scripts/run-evidence.sh my-smoke --eta "2-5 min" --cost "0 (local)" -- .venv/bin/python -u outcome_leak.py smoke
```

All the commands are listed at the top of `outcome_leak.py`. Each finding in the report maps to one command:

| finding | command | log |
|---|---|---|
| 1-2. the leak, and where it gets in | `patching` | `dana-outcome-patching-gemma-2-2b-it.log` |
| 3. taking it out | `removal`, then `removal-sturdy` | `dana-outcome-removal-gemma-2-2b-it.log`, `dana-outcome-removal-alltokens-gemma-2-2b-it.log` |
| 3. the fix on new stories | `removed-cases` | `outcome-removed-new-cases-gemma-2-2b-it.log` |
| 3. a different direction removed instead | `other-direction` | `dana-other-direction-gemma-2-2b-it.log` |
| 4. the outside test | `normal-test`, then `intervals` | `moral-luck-06-sep-gemma-2-2b-it-plain-vs-fixed.log`, `moral-luck-06-sep-gemma-2-2b-it-intervals.log` |
| 5. eval awareness | `eval-smoke`, `eval-natural`, `eval-direction`, ... | `eval-awareness-*.log` |

Every log starts with the command that made it.
