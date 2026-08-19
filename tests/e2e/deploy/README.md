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
6 rollouts, --save-interval 1, 2 restarts, ONE release
  Target only: relaunch the same command + --hot-restart orchestration,rollout_executor once a
  save and a step after it exist (second gate also demands disjoint redo windows).
  Asserts: only orchestrator + rollout-executor rolled (pod uid/restartCount/stamps); one trainer
  rpc boot uuid throughout; redo measured off the logs - one .trash_* per restart, resume point =
  the snapshot beside a checkpoint, per-step attempts all 1 or 2; comparison bitwise, target may
  miss each resume point's engine checksum (see limitations).
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

## `scenario_split_multi_policy`

```
3 rollouts, single run (multi trainer is not bitwise)
  Five releases: TRAINER solver-actor / verifier-actor, INFERENCE solver / verifier, PRIMARY last.
  Asserts: every rank trained with its own policy's args; every policy learned; the leader
  reported every rollout; finite nonzero grad_norm/loss; train_rollout_logprob_abs_diff <= 0.5
  per policy - the cheapest wiring bug (trainer scoring another engine's tokens) shows up there.
```
