# Fully Asynchronous Rollout Worker

Reference implementation of the **fully async** rollout pattern: a single global
worker is created once and keeps running in the background, continuously pulling
prompts and launching generation tasks. Training only drains already-finished
results, which removes the per-step wait of the synchronous style.

This is shared plumbing rather than a standalone recipe — see
[`examples/swe-agent/run-glm47-flash-agentic-async.py`](../../swe-agent/run-glm47-flash-agentic-async.py)
for a launcher that uses it, and `examples/infra_features/random_async/` for a
dataset-free stress-test variant.

## Files

* `fully_async_rollout.py`: global async worker + `generate_rollout_fully_async` entry.

## Enabling it

Two changes compared to a synchronous run:

1. Use the async training driver: `train_async.py` (not `train.py`).
2. Point the rollout function at this module:
   ```bash
   --rollout-function-path fully_async_rollout.generate_rollout_fully_async
   ```
   This directory must be on `PYTHONPATH` for the path to resolve.

## How it works

* First call: create `AsyncRolloutWorker` (thread + asyncio loop).
* The loop keeps up to `--rollout-batch-size` tasks in flight using `generate_and_rm_group`.
* Completed groups are pushed into a queue; the caller drains until it has enough samples.
* The worker is stopped automatically at process exit.

Each call from `train_async.py` only drains completed samples from the worker's
output queue; the worker has been generating continuously since the first call,
so generation and training overlap with minimal waiting.

## Limitations

* No evaluation mode.
* Ordering is best effort (sorted at the end by index).
* Minimal error handling.
