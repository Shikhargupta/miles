---
title: Fault Tolerance
description: Rollout-side health checks and engine recovery, gated by --use-fault-tolerance.
---
The `--use-fault-tolerance` flag enables Miles's rollout-side
fault-tolerance machinery. It gates two code paths:

1. A `RolloutHealthMonitor` thread per server group, started in
   `miles/ray/rollout.py`, which periodically heart-beats each SGLang
   engine.
2. Engine recovery, which is not a hook of its own: the controller
   reconciles the observed cells and the weight-update window waits for
   them to become ready (see "Engine recovery" below).

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

There is no dedicated recovery call. Recovery is the sum of two always-on
mechanisms: the controller's reconcile loop, and the weight-update window
that waits for reconciled cells to be ready.

### Reconcile loop

`InferenceController` (`miles/ray/rollout/inference_controller.py`) watches
worker cells through `RayWorkerProvider.watch_cells` and handles every
observation in `_reconcile`, which compares the observed cell against the
tracked `ServerCell` by `workers_hash`:

- Observed but untracked: `add_cell` on the server named by the cell's `model_id`.
- Tracked but no longer observed: `remove_cell`, which unregisters it from the router.
- Tracked with a different `workers_hash` (the cell was relaunched): `remove_cell` then `add_cell`.

A new `ServerCell` starts `Uninitialized` (`miles/ray/rollout/cell_state.py`).
`SimpleTicker` runs `_tick_cells` every `TICK_INTERVAL_SECONDS` (`5.0`); once
`probe_server_healthy` succeeds, `ServerCell.tick` moves the cell
`Initializing` → `PendingWeights`, releasing and re-onloading the weights
memory first if `needs_offload` is set.

### Weight-update window

The trainer brackets each weight update with `update_weights_window`
(`miles/ray/actor_group.py` for v1, `miles/ray/train/group.py` for v2):

1. `start_update_weights` pauses health probing, then `_ensure_cells_ready`
   polls every `CELLS_READY_POLL_INTERVAL_SECONDS` (`2.0`) up to
   `CELLS_READY_TIMEOUT_SECONDS` (`3600.0`) until every cell is `PendingWeights`
   or `Serving`, and returns the engine snapshot plus each cell's `workers_hash`.
2. The trainer broadcasts the weights to the engines in that snapshot.
3. `end_update_weights` marks every still-pending cell whose `workers_hash` is
   unchanged as weights-ready, which registers it with the router and moves it
   to `Serving`.

The controller's context lock is held for the whole window, so no reconcile can
change the engine set under the trainer. If the broadcast raises,
`update_weights_window` calls `abort_update_weights`, which releases the lock
without marking anything ready; the next reconcile and the next weight update
then proceed normally.

A cell that died is therefore recovered without any explicit restart call: the
worker manager relaunches it, the reconcile loop observes the new
`workers_hash` and re-adds the cell, and the next weight-update window waits for
it and hands it the current weights before it serves traffic.

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
