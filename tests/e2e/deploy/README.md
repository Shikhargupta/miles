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

## Test Specifications

### `scenario_split_deterministic`

```
Type: comparison (baseline=one release, target=one release per deployment)
Steps: 3 rollouts

1. Baseline: the whole run in one release
2. Target, installed in order: TRAINER; INFERENCE e0, e1 (one engine each); PRIMARY last
   - installing PRIMARY blocks until the run ends
   - addresses from the example's address_book; ordering, shared run uuid and uninstall from
     conftest_deploy/split_deployment.py
3. Compare: dumps and metrics bitwise; engine checksums identical per (rollout, engine); engine
   count; weights moved; nonzero gradients >= 2 rollouts
```

### `scenario_split_multi_policy`

```
Type: single run (multi trainer is not bitwise-reproducible)
Steps: 3 rollouts
Releases: TRAINER solver-actor / verifier-actor, INFERENCE solver / verifier, PRIMARY last

1. Install the five releases via the example, one command per part
2. Assert: every rank trained with its own policy's args; every policy learned
   (TRAIN_REWARD_BOUNDS); the leader reported every rollout; finite nonzero grad_norm/loss
3. Assert per policy: train_rollout_logprob_abs_diff <= 0.1

The cheapest wiring bug - a trainer scoring another engine's tokens - shows up in assertion 3.
```
### `scenario_hot_restart_deterministic`

```
Type: comparison (baseline=untouched, target=same command, script replaced twice mid-run)
Steps: 6 rollouts, --save-interval 2 (saves after steps 1, 3, 5)
Trigger schedule, asserted on the records: restart 1 at (save=1, finished=2), step 3 in flight;
                                           restart 2 at (save=3, finished=4), step 5 in flight

1. Relaunch the same command + --hot-restart orchestration,rollout_executor per the schedule
2. Assert workloads: only orchestrator + rollout-executor rolled (pod uid / restartCount / stamps)
3. Assert process: one trainer rpc boot uuid throughout
4. Assert redo, measured off the logs: one .trash_* per restart; resume point = the snapshot
   beside a checkpoint, >= the pinned save; per-step attempts all 1 or 2; every window non-empty
5. Compare: bitwise as in scenario_split_deterministic, engine checksums included

Every take-over lands on a non-save step, so at least one unsaved step is rolled back and redone.
```

### `scenario_hot_restart_no_checkpoint`

```
Type: comparison (baseline=untouched, target=same command, script replaced ONCE before any save)
Steps: 6 rollouts, --save-interval 4 (saves after steps 3 and 5), take-over window 0..2

1. Gate: opens on the first finished step while no checkpoint exists; a save seen first fails
2. Assert workloads/process: as scenario_hot_restart_deterministic with one restart
3. Assert redone-from-scratch: record carries no saved iteration; NO .trash_* (the run's --load
   resolves to --ref-load, which holds no snapshot to restore); steps 0..F appear twice, no hole,
   nothing thrice; the run still saves after the restart
4. Compare: bitwise, no checksum exemption

Production saves every ~20 steps, so a restart at step 10 is this case: load_state finds no
tracker, re-seeds, resets the optimizer and returns start rollout 0.
```

### `scenario_hot_restart_realistic_gsm8k`

```
Type: single run, ft's scenario_realistic_gsm8k with hot restarts instead of kills
Steps: as scenario_realistic_gsm8k
Injection: HotRestartFaultForm at random intervals via the ft fault-injection plan, seed logged
Eligibility: a save exists and a step finished after it; an ineligible draw waits, never fires

1. Run the realistic gsm8k recipe while the plan injects hot restarts
2. Assert: gsm8k reward improves as in scenario_realistic_gsm8k
3. Assert: every scheduled restart happened; only orchestrator + rollout-executor ever rolled;
   one trainer boot uuid throughout

Hot restart rides the ft injection machinery so a future soak can mix it with pod kills.
```

