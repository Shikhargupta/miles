---
title: Hot Restarting the Orchestration Script
description: Replace the orchestration script and the rollout executor of a running run while its trainers and inference deployments keep their weights loaded.
---
Changing the training loop normally means restarting everything: megatron reloads the checkpoint, the
sglang engines start from scratch, and the gpus idle through the whole cold start. A hot restart
replaces only the two components that carry the loop - the orchestration script and the rollout
executor - and hands the still-running trainers and inference deployments to the new script.

<Warning>

**Status.** Under active development, verified by hand on kubernetes rather than in CI.

</Warning>

## What a hot restart is

- `--hot-restart orchestration,rollout_executor` reruns the normal `execute_train` launch, so any
  change to the arguments of *these two components* propagates, and forces exactly those two
  StatefulSets to be replaced. Trainer-side arguments do not; see the limitations below.
- Everything else - trainer controllers, megatron ranks, inference controllers, sglang engines,
  routers, the mooncake master - stays up and is taken over by the new script.
- Semantics: **the run rolls back to its last checkpoint and continues.** A hot restart is not
  cheaper than a full restart in what it preserves, only in what it avoids reloading.

## Preconditions

| Precondition | Why |
| --- | --- |
| `--deploy-component primary` | the trainers and the inference side must be releases of their own, or restarting this release restarts them too |
| `--inference-controller-addrs` | the inference controller is a release of its own, and every engine of the run - deployed here or registered from another deployment - is a cell of *its* process, so a new script reconnects to one address and finds the whole fleet |
| `--inference-router-addrs` | the routers keep serving across the restart, so this launch must deploy none of its own; it names the running ones instead |
| `--trainer-controller-addrs` | the trainers are addressed statically, so a new script reconnects to them trivially |
| not `--indep-dp` | see the limitations below |
| not `--multi-lora` | see the limitations below |
| both components together | `--hot-restart` takes `orchestration,rollout_executor` and nothing narrower |

The launch refuses with an explanatory assertion when any of these is missing.

## What the new orchestration script does

1. **Takes over the inference side.** The controller outlives the orchestration script and owns the
   engine cells itself, including the ones other deployments registered into it, so an orchestration
   restart is invisible to registration and nothing has to be rediscovered or re-registered. The
   take-over, in order, each phase under a budget of its own:
    - **waits for the calls of the previous script to end.** A call that is still running holds the
      controller's lock without having detached it, so the weight-update window below would read shut
      while it is in flight - the script that died *inside* `start_update_weights` is exactly the case
      this protects;
    - **aborts a weight-update window the previous script left open.** The controller is still holding
      the lock that update opened, and every locked call would wait on it forever. A hot restart does
      not restart the trainer, so the broadcast that window was opened for may still be running: the
      window is aborted only after the trainer answered that it is idle *and* that no rank is still
      broadcasting into it. The window records the model id the weight update was *asked for*, which is
      the name the trainer side knows, so with several policies that question reaches the one trainer
      that owns the broadcast rather than being unroutable - and a single policy run, whose sglang model
      may be named anything, records no model id and reaches its one trainer as it always did. A trainer
      that cannot be reached, or that answers that it is still broadcasting, stops the take-over with
      an error instead: the window keeps its lock and its paused health checking, because resuming
      health checking here would let an engine the broadcast is writing into be restarted
      mid-broadcast. Each of the two waits carries its **own** budget, and each budget is derived from the
      inner bound that wait is waiting on rather than from a shared clock: the confirmation wait is the
      trainer's own liveness deadline plus a grace period, derived once and used by every caller that waits
      on that deadline - here and in the orchestration script's own abort of a failed broadcast - and the
      idle wait, which waits out the previous script's broadcast call itself, is derived from the same
      number. A wait can therefore never expire together with the inner deadline it guards and read a
      "still broadcasting" answer as "never answered", *including* when the idle wait was the slow one -
      which is exactly the case this take-over exists for, because the previous script's `update_weights`
      is still running on the trainer. The price of that guarantee is stated rather than hidden: a
      take-over that stalls in both waits costs their sum, not one of them. The trainer's answer is
      therefore waited out rather than cut short, and only an unreachable trainer is reported as one: a bug
      in this run surfaces as itself rather than as a trainer that never answered, and is logged where it
      happened;
    - **aborts every in-flight generation**, so no request survives into the new loop under new
      weights, and the fleet stops burning kv cache on a dead script's requests right away rather than
      after the trainers have reloaded. A cell that refuses the abort is named in an error and the
      other cells are aborted anyway; the take-over reports a quiet fleet only when every cell
      answered.

   A cold start does none of this: it initializes the controller and never calls any of the three, so
   an engine that is not answering yet cannot fail a launch that has not started training.

   Every phase is bounded by the take-over's own wait rather than by the implementation it calls, so the
   whole take-over is bounded, and the worst case is their sum:

   | Phase | Upper bound |
   | --- | --- |
   | wait for the previous script's calls to end | 300s |
   | wait for the trainer to go idle | 660s |
   | wait for the trainer to confirm it stopped broadcasting | 660s |
   | abort the weight-update window | 300s |
   | abort every in-flight generation | 300s |
   | wait for the whole fleet | 300s |

   A take-over that stalls in every phase therefore reports its failure after about 2520s (42 minutes).
   The two 660s budgets are the price of never reading a trainer that answered as a trainer that never
   answered; shortening either below the trainer's own 600s liveness deadline would buy time back by
   reintroducing exactly that misreading.
2. **Waits for a fresh rollout executor.** The old executor may still be answering while kubernetes
   replaces it; the script waits until the executor it talks to reports itself *not initialized*.
3. **Resumes the trainers.** For each trainer it asks `is_initialized()`:
    - not initialized: `init()`, exactly as a cold start;
    - already initialized: wait until the trainer is idle, then `load_state()`, which reloads
      weights, optimizer, scheduler and rng from the checkpoint in place and answers the rollout id
      to resume at. `load_state()` re-derives which checkpoint that is against the filesystem, exactly
      as a fresh parse of the same command would, so a run that started before its `--load` directory
      existed resumes from the checkpoint it has written since rather than from `--ref-load`.
4. **Pushes weights into the engines**, unconditionally, exactly as a cold start does.

Every component's `init` asserts it runs exactly once per process, so a mistaken second init fails
loudly instead of re-initializing a live system.

## What survives and what does not

| | Across a hot restart |
| --- | --- |
| megatron weights, optimizer, scheduler, rng | rolled back to the last checkpoint, onto exactly the schedule state a cold restart from that checkpoint would produce |
| the reference model under `--ref-update-interval` | reloaded from `--ref-load`, exactly as a cold start builds it |
| the distillation teacher under `--use-opd` | reloaded from `--opd-teacher-load`, exactly as a cold start builds it |
| sglang engines, kv cache, cuda graphs | kept, then weight-updated |
| trainer weight version counter | keeps counting; it never restarts at zero |
| dataset position | restored from the checkpoint's data source state |
| pending-sample buffer | **lost** |
| in-flight generations | **aborted** |
| the launch record behind `--env-report` | **stale**: the pods keep pointing at the record of the launch that installed them |

- The buffer loss is deliberate: it is exactly what a full restart from a checkpoint loses today, and
  a hot restart does not promise more.
- The stale launch record is deliberate too, and it is the one thing a hot restart reports wrongly.
  `run.launchRecord` is injected into **every** container of the release, not only into the two
  objects a hot restart replaces, so pointing it at this launch's record would make the session
  server, the mooncake master and the prepare-cmd job differ from what is installed, and the relaunch
  gate would refuse every hot restart of a run that deploys any of them. The record path is therefore
  left as it was, and the `launcher_env_report` in wandb's config and in the event log still describes
  the **first** launch of the release: its argv, its env and its values file. The argv and env the
  process itself reports are collected from the live process and stay accurate, so only that one
  field is out of date. A test pins it, so it cannot drift into being wrong by accident.
- With `--non-persistent-ckpt-type local` the surviving trainer reloads through the in-memory
  checkpoint manager its own `init` built, so it may roll back to a *newer* iteration than a cold
  restart, which only sees the persistent checkpoint.

## Where the sample buffer lives

- The pending-sample backlog belongs to the **rollout executor process**, inside its data source.
- It is persisted through the `save_buffer` / `load_buffer` hooks of
  `RolloutDataSourceWithBuffer`, called from `save()` and `load()`.
- The default pair is a no-op, which is why a restart drops the backlog. A future replay buffer
  implements those two hooks and nothing about the process topology changes.

## Generation mismatch and the boot uuid

- Every rpc response carries the server process's boot uuid, and every production client pins it.
  A client that sees a different uuid throws `ServerRestartedError` rather than driving a process it
  never initialized.
- The uuid may change **during `wait_ready`** and only there: each ready attempt re-baselines the
  pin, so an expected restart is tolerated while a silent one afterwards is not.
- The sglang router is plain HTTP and carries no uuid. A hot restart never replaces a router - the
  routers are named by `--inference-router-addrs` and are a deployment of their own - but a router
  that dies and is rescheduled *by the cluster* under a live client is still **not** detected. Known
  gap, and unrelated to which script is driving.

## Relaunch safety

A hot restart is a helm upgrade of an existing release, so the usual "a relaunch may only resize a
run" gate applies, with one widening:

- any field of the orchestrator and rollout-executor **StatefulSets** may change - that is the point
  of the feature - and the exemption is keyed by kind and name together, so the Service,
  ServiceAccount, Role and RoleBinding that share those names are not exempted with them;
- every other object must diff to zero, so "I thought I was hot restarting but I changed the
  trainer" is refused;
- `force=True` (`MILES_SCRIPT_FORCE=true`) overrides the refusal.

An ordinary relaunch after a hot restart renders the `restart-at` stamp the installed manifest
already carries, rather than dropping it, so it stays a zero diff: relaunching a hot-restarted run id
to resize it or to attach to it neither is refused by the gate nor rolls the two pods again.

- The stamp is the launch's wall-clock time to the microsecond, so two hot restarts of one run inside
  one second still differ, and both objects really do roll the second time.
- An installed manifest carrying two different stamps - an interrupted upgrade, or an object patched
  by hand - is a warning, not a refusal: the orchestrator's stamp is carried forward, and it is
  rendered only onto the pools that really carry that stamp, so a relaunch does not stamp a pool that
  never got one and turn its own zero diff into a refusal.

The orchestrator gets a fresh exit-state file, so the replacement pod runs the script instead of
reporting the predecessor's verdict. A nonzero orchestrator exit is still an experiment failure, and
never a reason to restart.

A launch also deletes the pending auto-uninstall job of the release it is launching, whether it
installs the release or upgrades one that is already there. An orchestration script that exits arms a
job that sleeps two minutes and then uninstalls the release, so the most natural way to use a hot
restart - the script died, replace it - used to succeed, bring up the new pods, and then have the
whole primary release removed under it about two minutes later. The delete runs after the relaunch
gate, so a refused launch still leaves the running release, job included, exactly as it was.

## Limitations

<Warning>

- **`--indep-dp` is rejected.** The independent-DP store and quorum id are built by the trainer
  controller's one-time `init`, and nothing yet re-derives that state for a second orchestration
  script.
- **Ray is not supported.** A ray actor does not track the calls it is running, so nobody can wait
  for a trainer to go idle; hot restart needs the rpc communication backend, i.e. kubernetes.
- **Only `--train-backend megatron`.** Reloading a trainer in place is a megatron-only operation, and
  the launcher refuses any other backend up front rather than letting the run discover it after the
  take-over already aborted every generation in flight.
- **`--multi-lora` is rejected.** A reload hides the adapter parameters while it loads the base
  checkpoint, so they survive it physically while the bookkeeping that owns their megatron slots is
  reset; the next reconcile would load every adapter into an occupied slot.
- **Only these two components, and only together.** `--hot-restart` accepts exactly
  `orchestration,rollout_executor`. Either alone is refused: a new script cannot drive the executor
  its predecessor initialized, and an executor replaced under a live script kills the run it belongs
  to.
- **Trainer-side arguments do not propagate.** The trainers are a release of their own and are
  resumed through `load_state()`, which carries no arguments, so a changed `--lr`,
  `--global-batch-size` or `--ref-update-interval` reaches the orchestration script and the rollout
  executor only. The relaunch gate diffs the primary release, so it cannot catch this either.
  Changing how the trainers train needs a full restart. The one group this does not cover is the
  checkpoint source (`--load`, `--critic-load`, `--finetune`, `--no-load-optim`, `--no-load-rng`,
  `--ckpt-step`): miles derives it from the filesystem at parse time, so the trainer's copy expires as
  soon as the run saves, and `load_state()` therefore re-derives it rather than trusting what its
  process was started with. What that re-derivation produces is exactly what a cold start of the same
  command would produce at that moment, for a single policy, for each policy of a multi policy run
  (whose per-policy `policies/<model id>` directory is part of the requested source rather than
  something layered on after it), and for the critic (whose `--critic-load` is re-derived with the
  actor's rather than frozen at whatever the very first parse fell back to).
- **`--megatron-to-hf-mode=bridge` does not roll back to its checkpoint.** That parse branch pins
  `--start-rollout-id` to 0 whether or not the run has written a checkpoint, so the run restarts its
  rollout numbering from the beginning rather than continuing after the checkpoint. This is
  pre-existing behavior and a cold restart of the same command does exactly the same thing, so a hot
  restart is no worse than the alternative - but "rolls back to its last checkpoint and continues"
  above describes the other modes.
- **Metrics split across two wandb runs.** The surviving trainers stay attached to the run id they
  were initialized with, while the new orchestration script opens a run of its own, so trainer-side
  series (lr, grad norm, train timers) and orchestration-side series (rollout, reward, eval) land in
  two different wandb runs, and the old one keeps looking alive.
- **The router gap above.**

</Warning>

## Running one

```bash
MILES_SCRIPT_RUN_ID=$RUN_ID MILES_SCRIPT_DEPLOY_COMPONENT=primary \
  MILES_SCRIPT_HOT_RESTART=orchestration,rollout_executor \
  python train_multi_policy.py --trainer-controller-addrs policy_a=<host>:8000 \
    --inference-controller-addrs http://<inference-host>:8000 \
    --inference-router-addrs policy_a=<router-host>:8000
```

Watch the new script take the trainers over:

```bash
kubectl -n "$MILES_NS" logs -l "app.kubernetes.io/instance=miles-run-$RUN_ID-primary,app.kubernetes.io/component=orchestrator" --tail=-1 | grep "already initialized"
```

Check that only the two StatefulSets were replaced:

```bash
kubectl -n "$MILES_NS" get pods -l "app.kubernetes.io/instance=miles-run-$RUN_ID-primary" -o wide
```
