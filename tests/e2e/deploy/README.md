# Deployment E2E Tests

How a run is deployed, not what it trains. Tests import the examples
(`examples/infra_features/{split_deployment,hot_restart}`, `examples/multi_policy`); train args and
comparison machinery come from `tests/e2e/ft`. Never run on a cluster: every claim below is what
the code is written to do.

| Entry | Scenario | Suite | `est_time` | Labels |
| --- | --- | --- | --- | --- |
| `test_split_deterministic.py` | `scenario_split_deterministic` | `stage-c-8-gpu-h200` | 2600 | `deploy` |
| `test_hot_restart_deterministic.py` | `scenario_hot_restart_deterministic` | `stage-c-8-gpu-h200` | 5400 | `deploy` |
| `test_hot_restart_no_checkpoint.py` | `scenario_hot_restart_no_checkpoint` | `stage-c-8-gpu-h200` | 6000 | `deploy` |
| `test_split_multi_policy.py` | `scenario_split_multi_policy` | `stage-c-8-gpu-h100` | 1800 | `deploy`, `multi-policy`, `fully-async` |

CI label `run-ci-deploy`; without kubernetes an entry skips in seconds. Comparison scenarios pin a
private `_MODE`: 1 node, 4+2 GPUs, CP2, dense `Qwen3-0.6B`, 2 engines × 1 GPU. No mode matrix.

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
  count/checksums, weights-moved and nonzero-gradient gates. Bitwise comes from the deterministic
  flags + `--sglang-disable-radix-cache` + `--weight-decay 0`.

## `scenario_split_deterministic`

```
3 rollouts
  baseline = one release; target = TRAINER + INFERENCE e0,e1 (one engine each) + PRIMARY last
  (blocks until the run ends). Addresses from the example's address_book; ordering, shared run
  uuid and uninstall from conftest_deploy/split_deployment.py.
  Risk: request routing differs across the sides - leans on sglang determinism being
  engine-assignment invariant.
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

## Known limitations

- **Hot restart targets fail in the outer launcher**: the take-over takes a new state file, the
  replaced wrapper writes `exited/143` into the old one, the launcher raises `SystemExit(143)`.
  Product settled only the new launcher's side.
- **Missing engine checksum is exempted, not explained**: the log snapshot is cut between a save
  and that rollout's weight update. Fix is to snapshot after the update.
- **No-checkpoint scenario hits two more gaps**: `rollout_executor.load(-1, require_state=True)`
  dies before the train loop, and the un-rolled-back log carries duplicate checksum events the
  comparator refuses. Assertions encode both.
- **The no-checkpoint window is a race**: a take-over landing after the first save fails loudly on
  the no-`.trash_*` assertion; the slack is unmeasured.
- **Install order leans on `SUBMIT_RETRY_WINDOW_SECONDS` (60s)**, driver last; slow image pulls
  between releases are untested. Registration is not a constraint.
- **Split engines both report gpu offset 0**; nothing reads offsets for placement.
- **The boot uuid proves one process** (trainer controller); pod uid/restartCount cover the rest.
- **Thresholds and durations are unmeasured**: `logprob_abs_diff <= 0.5`, `est_time`, gate/join
  timeouts - first guesses.
- **Run uuid**: split scenarios name one themselves (the product refuses to generate one); hot
  restart relaunches inherit it back off the installed orchestrator's argv.
