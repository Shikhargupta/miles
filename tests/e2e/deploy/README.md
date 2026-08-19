# Deployment E2E Tests

## Running

Needs `PYTHONPATH=.` and a miles-workbench pod (`MILES_SCRIPT_*` env preset). Kubernetes only.

```bash
PYTHONPATH=. python tests/e2e/deploy/test_split_deterministic.py                          # as CI
PYTHONPATH=. python tests/e2e/deploy/conftest_deploy/scenario_split_deterministic.py run  # via app
```

- **Subcommands**: comparison scenarios expose `run` / `baseline` / `target` / `compare` (no GPU) /
  `generate-data`; the multi policy one exposes `run` / `verify`.
- **Dump dirs**: `/node_public/dumps/<TEST_NAME>/` (only `run` deletes it; `--dump-dir` overrides);
  multi policy: `<output_dir>/multi_policy_solver_verifier/<run_id>/`.
- **Comparison criterion** (`compare_deterministic_sides`): dumps and metrics bitwise, plus engine
  count/checksums, weights-moved and nonzero-gradient gates.

## `scenario_split_deterministic`

```
3 rollouts
  baseline = one release; target = TRAINER + INFERENCE e0,e1 (one engine each) + PRIMARY last
  (blocks until the run ends). Addresses from the example's address_book; ordering, shared run
  uuid and uninstall from conftest_deploy/split_deployment.py.
```

## `scenario_hot_restart_deterministic`

```
6 rollouts, --save-interval 2 (saves after steps 1, 3, 5), 2 restarts, ONE release
  Target only: relaunch the same command + --hot-restart orchestration,rollout_executor on an
  exact schedule - restart 1 at (save=1, finished=2), step 3 in flight; restart 2 at (save=3,
  finished=4), step 5 in flight. Every take-over rolls back at least one unsaved step; the
  recorded trigger pairs are asserted to equal the schedule.
  Asserts: only orchestrator + rollout-executor rolled (pod uid/restartCount/stamps); one trainer
  rpc boot uuid throughout; redo measured off the logs - one .trash_* per restart, resume point =
  the snapshot beside a checkpoint, per-step attempts all 1 or 2, every window non-empty;
  comparison bitwise, engine checksums included.
  Twice because the second take-over hits trainers already taken over once.
```

## `scenario_hot_restart_no_checkpoint`

```
6 rollouts, --save-interval 4, ONE restart in window 0..2
  Gate opens on the first finished step while no checkpoint exists; a save seen first fails.
  Asserts: workloads/process as above; NO .trash_* (nothing to restore - --load resolves to
  --ref-load); the one log holds steps 0..F twice, no hole, nothing thrice; the run saves after
  the restart; comparison with no checksum exemption.
  Own scenario because load_state without a tracker re-seeds, resets the optimizer, returns 0.
```

## `scenario_hot_restart_random`

```
ft's scenario_realistic_gsm8k with hot restarts instead of kills, ONE release
  Reuses nearly all of the realistic gsm8k convergence test; the injection plan schedules a
  HotRestartFaultForm - hot restart as a pseudo fault-injection action - at random intervals the
  way it schedules pod kills, so a future soak can mix the two. A moment is eligible when a save
  exists and a step finished after it; an ineligible draw waits rather than fires. Seed logged.
  Asserts: the gsm8k reward improves as in scenario_realistic_gsm8k; every scheduled restart
  happened; only orchestrator + rollout-executor ever rolled, one trainer boot uuid throughout.
```

## `scenario_split_multi_policy`

```
3 rollouts, single run (multi trainer is not bitwise)
  Five releases: TRAINER solver-actor / verifier-actor, INFERENCE solver / verifier, PRIMARY last.
  Asserts: every rank trained with its own policy's args; every policy learned; the leader
  reported every rollout; finite nonzero grad_norm/loss; train_rollout_logprob_abs_diff <= 0.5
  per policy - the cheapest wiring bug (trainer scoring another engine's tokens) shows up there.
```
