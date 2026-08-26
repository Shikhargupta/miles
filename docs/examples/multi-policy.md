---
title: "Multi Policy Solver and Verifier"
description: "Two policies in one run — a solver answering gsm8k, and a verifier scored on ruling correctly about the solver's answers."
# Generated from examples/multi_policy/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
Two policies trained against each other in one run: a solver answering a gsm8k question and a
verifier ruling on its work. Each rollout yields one sample per policy; each policy has its own
trainer and its own inference engines inside the same job.

## Quick Start

Eight GPUs: 2 trainer GPUs and 2 single-GPU engines per policy. Needs a cluster backend and a
namespace, from `MILES_SCRIPT_CLUSTER_BACKEND` / `MILES_SCRIPT_NAMESPACE` or `--cluster-backend` /
`--namespace`:

```bash
python examples/multi_policy/run_solver_verifier_gsm8k.py
```

## Recipe

The hyperparameters mirror `tests/e2e/long/test_qwen2.5_0.5B_gsm8k_async.py`, the
single-policy GSM8K baseline that demonstrably learns: 32 groups of 8 samples per
rollout, global batch 256, response length 1024, temperature 1.0, and the
nonzero-std dynamic-sampling filter. With the filter on, `rollout/raw_reward`
averages only the accepted mixed groups, whose mean sits near .5 by construction,
so read learning from the held-out eval curves instead.

## Evaluation

Every 20 rollouts (off in the short CI variant) the run evaluates both policies on
the GSM8K test split through the same solver-verifier chain. The eval data is
split per policy before logging, so the run reports `eval/gsm8k/solver` and
`eval/gsm8k/verifier` rather than one mean over both policies.
