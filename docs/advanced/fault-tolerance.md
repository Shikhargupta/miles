---
title: Fault Tolerance
description: Rollout-side health checks and engine recovery, gated by --use-fault-tolerance.
---
The `--use-fault-tolerance` flag enables Miles's rollout-side
fault-tolerance machinery. It gates two code paths:

1. A `RolloutHealthMonitor` thread per server group, started in
   `miles/ray/rollout.py`, which periodically heart-beats each SGLang
   engine.
2. A recovery hook in the trainer's weight-update step
   (`miles/backends/megatron_utils/actor.py`), which restarts engines
   that the health monitor has killed.

```bash
--use-fault-tolerance
```

The flag is `action="store_true"`, default `False`
(`miles/utils/arguments.py`).

## Health monitor

`RolloutHealthMonitor` (`miles/utils/health_monitor.py`) runs in a daemon
thread. Lifecycle: `start` (called once during init), `pause` and `resume`
(called when engines offload / onload), `stop` (called during dispose).
`pause` / `resume` are wired up in `miles/ray/rollout.py` and called
around offload / onload events.

Each loop iteration does:

1. After a `resume`, wait `--rollout-health-check-first-wait` seconds before
   the first check (intended to cover model compilation and initialization).
2. For every active engine in the group, call `engine.health_generate.remote(timeout=self._check_timeout)`.
3. If the call raises, run `_kill_engine`: `engine.shutdown.remote()`,
   `ray.kill(engine)`, and the engine slot is set to `None`
   (`miles/utils/health_monitor.py`).
4. Sleep `--rollout-health-check-interval` seconds, then repeat.

### Flags

| Flag | Default | Source help text |
|---|---|---|
| `--rollout-health-check-interval` | `30.0` | "Interval in seconds between rollout engine `/health_generate` checks during generate/eval." |
| `--rollout-health-check-timeout` | `30.0` | "Timeout in seconds to wait for a rollout engine `/health_generate` response before killing it." |
| `--rollout-health-check-first-wait` | `0` | "Initial grace period (in seconds) before starting health checks. This allows time for model compilation and initialization. Increase this value significantly when using deepgemm." |

## Engine recovery

Engine lifecycle commands never pass through the rollout consumers. A dead or
suspended cell is restarted at the ops boundary: the api server's cell handles
(or any operator) call `start_cell` / `restart_cell` on the `RayWorkerManager`
directly. The `InferenceController` only observes — its reconcile watcher
(`miles/ray/rollout/inference_controller.py`) notices the fresh workers through
the worker provider, re-attaches the cell, and flags `has_new_engines` so the
next weight update connects to the new engines. Re-attached cells of an
updatable server stay out of the router until that update promotes them; a
frozen server's cells serve again right away.

## P2P weight transfer timeouts

When `--update-weight-transfer-mode p2p` is on, every P2P transfer is
bounded by `--p2p-transfer-timeout` (default `30.0`s, defined in
`miles/utils/arguments.py`; consumed at
`miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p.py`).
On timeout the failed transfer is logged (`[P2P] Transfer future failed: ...`)
in `p2p_transfer_utils.py`. There is no automatic retry or automatic
broadcast-mode fallback in the source today.

## Dumper-mode interaction

In dumper mode (`miles/utils/arguments.py`), Miles forces
`use_fault_tolerance = False` and `rollout_health_check_interval = 1e18`
to keep heartbeats from firing.
