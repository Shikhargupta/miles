# Deployment E2E Tests

## Overview

- **Covered**: how a run is deployed, not what it trains — see the Scenarios table.
- **The tests import the examples**: `examples/infra_features/split_deployment` for release names,
  dialled addresses and the arguments of one component; `examples/infra_features/hot_restart` and
  `examples/multi_policy/run_solver_verifier_gsm8k.py` for the workloads they run. The take-over
  itself lives in `conftest_deploy/hot_restart/relaunch.py`, and what the two hot restart scenarios
  assemble identically lives in `conftest_deploy/hot_restart/scenario_common.py` — kept inside
  `hot_restart/` so the naming guard's `scenario_*.py` glob does not take it for a scenario. Only
  observing and asserting live here.
- **A split example is a set of commands, not a script**: it composes the arguments of the one
  component an invocation names. A scenario calls that function once per component, which is the
  README's commands typed programmatically, and supplies what a reader supplies by hand: the
  product hands a split launch no run uuid, so each scenario names one itself for the whole run,
  then installs the releases in order and uninstalls them afterwards.
- **Everything else** — train args, comparison machinery — comes from `tests/e2e/ft`, never copied.
- **Never run on a cluster**: every claim below is what the code is written to do, not what was
  observed.

### CI Entries

- **Entry files**: `test_<TEST_NAME>.py`, one per scenario — a scenario here is one example of one
  shape, not a family of topologies, so an entry names nothing else.
- **Content**: `register_cuda_ci(est_time=..., suite=..., labels=[...])` plus `run_ci()` under
  `__main__`, no test logic.
- **Execution**: bare `python3 <file>` from the repo root, exit code = pass/fail
  (`tests/ci/ci_utils.py` `run_unittest_files`).
- **Enforced, not remembered**: `tests/fast/e2e/test_naming_scheme.py` holds the scenario/entry rules
  of this suite and of `tests/e2e/ft` in one place, and fails when a name drifts.

| Entry | Scenario | Suite | `est_time` | Labels |
| --- | --- | --- | --- | --- |
| `test_split_deterministic.py` | `scenario_split_deterministic` | `stage-c-8-gpu-h200` | 2600 | `deploy` |
| `test_hot_restart_deterministic.py` | `scenario_hot_restart_deterministic` | `stage-c-8-gpu-h200` | 5400 | `deploy` |
| `test_hot_restart_no_checkpoint.py` | `scenario_hot_restart_no_checkpoint` | `stage-c-8-gpu-h200` | 6000 | `deploy` |
| `test_split_multi_policy.py` | `scenario_split_multi_policy` | `stage-c-8-gpu-h100` | 1800 | `deploy`, `multi-policy`, `fully-async` |

### Scenarios

- **Each scenario**: a typer app plus a `run_ci()` runner.

| Scenario (`conftest_deploy/scenario_*.py`) | Type | What it verifies |
| --- | --- | --- |
| `scenario_split_deterministic` | comparison | one trainer + two engine releases train bitwise-identically to the same run in one release |
| `scenario_hot_restart_deterministic` | comparison | replacing the orchestration script twice mid-run costs nothing but the steps past the last checkpoint |
| `scenario_hot_restart_no_checkpoint` | comparison | replacing it before the run has saved anything costs every step it had finished, and nothing more |
| `scenario_split_multi_policy` | single run | two policies, each with its own trainer and inference release, train without breaking |

### Topologies

- **Each scenario pins its own**: a private `_MODE: FTTestMode` in the scenario file, fed to the same
  ft pipeline — no mode table, no `--mode`.
- **`FTTestMode` is reused as-is**: fields, defaults and assertions from `conftest_ft/modes.py`.
- **`scenario_split_multi_policy` declares none**: its topology is the split multi policy
  example's own.

| Scenario | Nodes | GPUs (train + rollout) | Megatron DP degree | Parallelism | Rollout | Model |
| --- | --- | --- | --- | --- | --- | --- |
| `scenario_split_deterministic` | 1 | 4 + 2 | 2 | CP2 | 2 engines × 1 GPU | dense `Qwen3-0.6B` |
| `scenario_hot_restart_deterministic` | 1 | 4 + 2 | 2 | CP2 | 2 engines × 1 GPU | dense `Qwen3-0.6B` |
| `scenario_hot_restart_no_checkpoint` | 1 | 4 + 2 | 2 | CP2 | 2 engines × 1 GPU | dense `Qwen3-0.6B` |
| `scenario_split_multi_policy` | 1 | 4 + 4 | 2 per policy | none | 2 engines × 1 GPU per policy | `Qwen2.5-0.5B-Instruct` + `Qwen3-0.6B` |

- **The DP degree is megatron's**, `4` train GPUs over `CP2`, not a cell count: nothing here passes
  `--use-fault-tolerance`, so no run in this directory has cells. `FTTestMode.num_cells` is set only
  because the dataclass demands it; the deploy path never reads it.
- **Why dense and small**: every comparison scenario needs real engines and bitwise equality, and the
  truncated MoE costs launch time without buying either.

## Running the code

### In CI

- **Gating label**: `run-ci-deploy`. Nothing here runs on an unlabelled PR.
- **Broad scopes**: every scope includes `deploy` — `run-ci-all`, nightly, weekly, `run-ci-image`
  (`tests/ci/ci_policy.py` subtracts only `long`, `ft-short`, `ft-long`).
- **Lane without kubernetes**: entries skip in seconds, but their `est_time` still counts toward that
  lane's balancing.
- **Add a scenario**: `conftest_deploy/scenario_<name>.py` plus an entry file named after it. A
  comparison scenario pins a `_MODE` of its own; a single run scenario takes the deployed example's
  topology and declares none.

### Manually

Needs `PYTHONPATH` at the repo root (CI sets it) and a miles-workbench pod on a kubernetes cluster —
see Cluster Backend below.

```bash
# One entry, exactly as CI runs it
PYTHONPATH=. python tests/e2e/deploy/test_split_deterministic.py

# The same scenario through its typer app
PYTHONPATH=. python tests/e2e/deploy/conftest_deploy/scenario_split_deterministic.py run

# One side only, against a dump dir you keep
PYTHONPATH=. python tests/e2e/deploy/conftest_deploy/scenario_hot_restart_deterministic.py target \
    --dump-dir /node_public/dumps/scratch

# The single-run scenario, and its assertions alone over what that run left
PYTHONPATH=. python tests/e2e/deploy/conftest_deploy/scenario_split_multi_policy.py run
PYTHONPATH=. python tests/e2e/deploy/conftest_deploy/scenario_split_multi_policy.py verify
```

Subcommands and backend gating differ by scenario kind:

| Scenario kind | Subcommand | Does | Gated on the backend |
| --- | --- | --- | --- |
| comparison | `run` | full pipeline: prepare + baseline + target + compare | no |
| comparison | `baseline` / `target` | one side only, for debugging | no |
| comparison | `compare` | re-run the comparison on existing dumps (no GPU) | no |
| comparison | `generate-data` | record debug rollout data with real engines, no dumper | no |
| `scenario_split_multi_policy` | `run` | prepare + install every release + verify; the callable the CI entry runs | yes |
| `scenario_split_multi_policy` | `verify` | re-run the assertions over what a previous `run` left | no |

- **Only `run_ci` is gated**: a comparison scenario's is a wrapper the typer app does not expose, so
  its `run` is ungated; `scenario_split_multi_policy`'s gated `run_ci` *is* the app's `run`.
- **Ungated on a non-kubernetes backend is not a skip**: it enters the launch path and fails wherever
  installing a helm release fails there — expected to be loud rather than known to be.
- **Debugging**: prefer the individual subcommands over `run`; a shared `--dump-dir` re-runs only
  what changed. `compare` on a hot restart dump re-reads `hot_restart/evidence.json`, replaying the
  restart assertions without a cluster.

Dump directories differ per entry point, and only `run` deletes what it wrote:

| Entry point | Dump directory | Deleted afterwards |
| --- | --- | --- |
| CI entry or manual `run`, comparison scenario | `/node_public/dumps/<TEST_NAME>/` | yes |
| Manual `baseline` / `target` / `compare` | `--dump-dir` if passed, else `/node_public/dumps/<TEST_NAME>/` | no |
| Either entry point, `scenario_split_multi_policy` | `<output_dir>/multi_policy_solver_verifier/<run_id>/` | no |

- **No `_<mode>` suffix here**: the suffix comes from `run_ci(mode)` appending one, and deploy
  scenarios call `run_ci()` with no mode, so `resolve_dump_dir` (`conftest_ft/app.py`) uses the bare
  `TEST_NAME`; the ft entries pass a mode and get `<test_name>_<mode>`.

### Cluster Backend

- **Selection**: `command_utils.default_config()`, off `MILES_SCRIPT_CLUSTER_BACKEND` /
  `MILES_SCRIPT_NAMESPACE` / `MILES_SCRIPT_RUN_ID`, already set in the miles-workbench pod.
- **Kubernetes only**: a split run installs one helm release per part, a hot restart upgrades one;
  ray does neither.
- **Declared unsupported skips, a failed probe fails**: `is_cluster_ready_for_helm_runs`
  (`conftest_deploy/cluster_gate.py`) skips with its reason on a non-kubernetes backend or no
  namespace, else calls `create_backend_for_run()`, which asserts — exiting 0 there would report
  green for a test that never ran.
- **The gate is about helm, not splitting**: `scenario_hot_restart_deterministic` shares it despite
  installing a single release, because what it decides is whether helm releases install at all.
- **Namespaced probes only**: the evidence collector's `kubectl get
  pods/statefulsets/leaderworkersets` select on the release label inside `config.namespace`, never a
  cluster-scoped read the workbench's Role cannot do.

## Test Specifications

### Comparison Criterion

- **Every comparison scenario shares `compare_deterministic_sides`**
  (`conftest_deploy/comparison.py`): dumps `rel <= 0` (bitwise); metrics `rtol=0` / `atol=0` over
  `train/` and `rollout/`; `perf/` ignored by name; `assert_every_metric_is_classified` fails a
  metric in neither namespace instead of dropping it. Engine count, weights-moved and
  nonzero-gradient checks are part of it, so no scenario can quietly drop one.
- **Model inputs**: `INPUT_TENSORS_ALLOW_FAILED_PATTERN` exempts `input_ids`, `positions`,
  `cu_seqlens_*`, `qkv_format`; `INPUT_TENSORS_SKIP_PATTERN` skips those plus `.*witness.*`. Nothing
  else is exempt.
- **Argument source**: each scenario builds its `ScriptArgs` and lets the example construct the train
  args — for a split scenario, once per component it declares; `conftest_deploy/example_args.py` adds
  the test-only part on top.
- **What makes bitwise possible**: `--deterministic-mode` and
  `--sglang-enable-deterministic-inference` from `build_deterministic_test_args`,
  `--debug-deterministic-collective`, and `NCCL_ALGO=Ring` etc. from `get_train_env_vars_arg`.
- **`--sglang-disable-radix-cache`**: the sides spread requests over engines differently (split) or
  restart the script between them (hot restart), and deterministic inference is nowhere documented as
  prefix-cache-length invariant.
- **`--weight-decay 0`**: overrides the shared optimizer recipe, so a step's update depends on the
  gradients alone.

### `scenario_split_deterministic`

```
Type: comparison (baseline=one release, target=one release per deployment)
Entry: test_split_deterministic.py, deploy
Steps: 3 rollouts (NUM_ROLLOUTS)
Requires: mode.has_real_rollout, and not mode.colocate
Compare: dumps rel <= 0 (bitwise); metrics rtol=0 / atol=0 over train/ and rollout/

Baseline: the whole run in one release, --rollout-num-gpus covering both engines

Target, installed in this order, each a helm release of its own:
  1. TRAINER              - the trainer cells
  2. INFERENCE e0, e1     - one engine each, --rollout-num-gpus 1 per release
  3. PRIMARY              - the orchestration script; last, since installing it blocks until the
                            run ends

Wiring, from the release names in examples/infra_features/split_deployment/address_book.py:
  - engines reach the driver at --inference-controller-addr
  - the driver reaches the trainer at --trainer-controller-addrs actor=<fqdn>
  - every part redeems object references at the PRIMARY release's mooncake master

Assertions:
  1. Reconfigure events: zero on BOTH sides - splitting a run is not a fault
  2. Metrics: rtol=atol=0 over train/ and rollout/
  3. Dumps: rel <= 0
  4. Engine checksums: baseline and target pushed identical weights per (rollout, engine)
  5. Engine count == mode.rollout_num_engines, per side
  6. Weights moved, per side: the engine checksum is not identical across all rollouts
  7. Gradients were finite and nonzero for >= 2 rollouts (MIN_TRAINED_ROLLOUTS), per side

The order, the shared run uuid, the helm get manifest after each install, and the uninstall of every
non-driver release in a finally are conftest_deploy/split_deployment.py's, not the example's; a
silent no-op install would otherwise leave the run waiting forever.
```

- **Why the baseline is unsplit**: the claim under test is that splitting changes nothing, so the
  reference has to be the shape production already trusts.
- **Known bitwise risk**: baseline has one pool of 2 engines, target two pools of 1, so requests
  reach engines in a different order — leaning on sglang's determinism being invariant to which
  engine and which batch a request lands in.

### `scenario_hot_restart_deterministic`

```
Type: comparison; both sides run the identical command, only the target has its orchestration
      script replaced twice, through the pipeline's target_side_context hook
Entry: test_hot_restart_deterministic.py, deploy
Steps: 6 rollouts (NUM_ROLLOUTS), --save-interval 1, 2 restarts (NUM_RESTARTS)
Requires: mode.has_real_rollout, and not mode.colocate
Topology: ONE release (DeployComponent.ALL), not a split run
Compare: dumps rel <= 0 (bitwise); metrics rtol=0 / atol=0 over train/ and rollout/

Restart trigger: conftest_deploy/hot_restart/relaunch.py relaunches the release with
  --hot-restart orchestration,rollout_executor, which helm-upgrades those two workloads and no
  others; it refuses to install any release but the one being watched

Driver (conftest_deploy/hot_restart/driver.py), a daemon thread polling every 5s off two facts
on the dump volume:
  - last saved iteration: the run's one latest_checkpointed_iteration.txt (several is refused)
  - last finished step: max rollout id of a MetricEvent carrying train/grad_norm
  1. Gate for restart i: a save S_i, then a step F_i > S_i finished after it
  2. Gate for restart 2 additionally demands S_2 >= F_1, so the redo windows are disjoint
  3. On an open gate: record (S_i, F_i), relaunch on its own thread, re-arm the gate
  4. A gate shut for 3600s (GATE_TIMEOUT_SECONDS) fails the run
  5. Per-poll failures are counted, not fatal; 5 in a row (CONSECUTIVE_FAILURE_LIMIT) fail the run
  6. On exit: join every thread, snapshot once more, write hot_restart/evidence.json, and only
     then assert no thread is still working

Evidence per snapshot (conftest_deploy/hot_restart/cluster_probe.py), kubectl-selected on the
release label:
  - pods: name, uid, summed container restartCount
  - statefulsets and leaderworkersets: name, generation, the restart-at annotation a hot restart
    stamps - the trainer cells and the engines are leaderworkersets, not statefulsets
  - the trainer rpc server's boot uuid, from the BOOT_UUID_HEADER of its health endpoint
  - reads attempted and reads failed, per kind

Assertions:
  1. Both restarts happened, the driver recorded no failure, and the cluster was seen >= 2 times
  2. Every read answered at least half the time (MINIMUM_OBSERVATION_SUCCESS_RATIO), so a verdict
     about pods is not one about an unreachable cluster
  3. The trainers and engines stayed, at the k8s layer:
     - every pod of the release belongs to a workload that was listed; none is unattributed
     - pods replaced (new uid, higher restartCount, or a changed pod set) == exactly the
       orchestrator and rollout-executor workloads
     - workloads whose generation moved or whose stamp changed == the same two
     - each of those two carries exactly NUM_RESTARTS distinct restart-at stamps, and no other
       workload carries any
  4. The trainers stayed, at the process layer: the trainer rpc server answered with exactly one
     boot uuid across every snapshot, and answered at least once after each restart stamp appeared
  5. Only the steps past a checkpoint were redone, measured off the logs rather than the records:
     - one .trash_* event dir per restart, each holding steps 0..F_i_actual with no hole in it
     - the resume point S_i_actual is the longest prefix the next log carries over byte-identical,
       and must equal the event log snapshot beside one of the run's checkpoints
     - the surviving event log describes each of the 6 steps exactly once
     - per-step attempt counts (distinct events across every log the run left) equal what the
       measured windows (S_i_actual, F_i_actual] predict, and are all 1 or 2
     - the records are used only as a lower bound: F_i_actual >= the step the driver saw
  6. Metrics, dumps, engine checksums, engine count, weights moved and nonzero gradients over
     >= 4 rollouts (MIN_TRAINED_ROLLOUTS), as in scenario_split_deterministic, except that the
     target is allowed to be missing the engine checksum of each measured resume point
     (see Known limitations)
```

- **Why twice**: a second take-over runs against trainers a first one already took over, which is
  where a take-over that leaks state into the process it attaches to shows up.
- **Why gates and not sleeps**: a restart before the first save has nothing to resume from, one with
  no step after the save redoes nothing, and overlapping redo windows would make assertion 5's
  per-step attempt counts ambiguous.
- **Why the records are only a lower bound**: a record is latched at trigger time and the take-over
  lands seconds later against a run that kept training, so the window really redone is read off the
  logs the run left.
- **Why one release rather than a split run**: the trainer and engine workloads then sit in the
  *same* release the relaunch upgrades, so "only two workloads rolled" is a claim about the mechanism
  rather than about releases the relaunch was never going to touch.
- **Why the boot uuid on top of the pod facts**: a pod keeping its uid can still have restarted its
  process; the uuid proves the checkpoint was reloaded into the process that had been training.
- **Why the comparator needs no attempt handling**: the take-over rolls the event log back, so the
  surviving log holds no duplicate rollout id — which assertion 5 checks rather than assumes.

### `scenario_hot_restart_no_checkpoint`

```
Type: comparison; both sides run the identical command, only the target has its orchestration
      script replaced once, before the run has saved anything
Entry: test_hot_restart_no_checkpoint.py, deploy
Steps: 6 rollouts (NUM_ROLLOUTS), --save-interval 4, 1 restart (NUM_RESTARTS)
Topology, requirements, restart trigger, driver and evidence: as in
      scenario_hot_restart_deterministic, whose driver, cluster probe, relaunch and process
      assertions this scenario reuses unchanged

Gate (conftest_deploy/hot_restart/gate.py NoCheckpointGate, handed to the driver as build_gate):
  1. Opens on the first finished step of a run whose checkpoint directory holds no
     latest_checkpointed_iteration.txt; the record it writes carries saved_iteration_at_trigger
     None, which is what tells the comparison which take-over this dump describes
  2. A save observed before the gate opened fails the poll rather than shutting the gate again: a
     take-over from there resumes from that checkpoint, which is the other scenario. The driver's
     CONSECUTIVE_FAILURE_LIMIT turns the repeated failure into a failed run
  3. --save-interval 4 over 6 rollouts saves after step 3 and after step 5, so the take-over has
     steps 0..2 to land in and the run still saves twice afterwards

Assertions:
  1..4. Restart count, observation ratio, workloads and boot uuid: assert_workloads and
     assert_process as in scenario_hot_restart_deterministic, with NUM_RESTARTS = 1
  5. Everything the run had done was redone, measured off the log rather than the record
     (conftest_deploy/hot_restart/assert_redone_from_scratch.py):
     - the driver recorded exactly one restart, and it recorded no checkpoint at trigger time
     - no .trash_* event dir exists: a take-over rolls the log back by restoring the copy beside
       the checkpoint it resumes from, and this run's --load resolves to --ref-load, which holds no
       snapshot of this run, so there is nothing to restore and nothing is moved aside
     - the one event log describes each of the 6 steps, the steps described twice are 0..F_actual
       with no hole in them, and no step is described a third time
     - F_actual >= the step the driver saw, the records being a lower bound as in the other
       scenario
     - the run saved after it was restarted: the checkpoint directory holds a tracker and at least
       one iter_*/debug_events snapshot
  6. Metrics, dumps, engine checksums, engine count, weights moved and nonzero gradients over
     >= 4 rollouts (MIN_TRAINED_ROLLOUTS), as in scenario_split_deterministic, with no rollout
     exempted: nothing is rolled back here, so no checksum event is lost (see Known limitations)
```

- **Why this is a scenario of its own**: production saves every 20 steps, so a run restarted at step
  10 is this case, not the other one. The trainer takes a different branch —
  `MegatronTrainRayActor.load_state` finds no tracker under `--save`, logs that it *goes back to the
  state the run started from*, re-seeds and resets the optimizer, and returns start rollout `0`.
- **Why once and not twice**: the take-over makes the run save, so no later restart finds it without
  a checkpoint again; the second-take-over coverage is `scenario_hot_restart_deterministic`'s.
- **Why the gate fails rather than waits**: a run that saved first is the other scenario, and a
  variant that quietly re-proves it would report green while covering nothing new.
- **The launcher gap of the other scenario applies unchanged**: it is triggered by replacing the
  orchestrator, not by checkpoints — see Known limitations.

### `scenario_split_multi_policy`

```
Type: single run (no baseline, no compare); passes if the run completes and the assertions hold
Entry: test_split_multi_policy.py, deploy + multi-policy + fully-async
Steps: 3 rollouts (ScriptArgs.num_rollout)
Recipe: examples/multi_policy/run_solver_verifier_gsm8k.py, shared with the short and long entries
Launcher: examples/infra_features/split_deployment/run_solver_verifier_gsm8k_split.py, one command
          per part
Policies: solver (Qwen2.5-0.5B-Instruct, the leader) and verifier (Qwen3-0.6B)

Target, installed in this order, each a helm release of its own:
  1. TRAINER solver-actor, TRAINER verifier-actor - one policy's megatron config each
  2. INFERENCE solver, INFERENCE verifier         - one policy's sglang config each, 2 GPUs each
  3. PRIMARY                                      - the orchestration script, installed last

Assertions:
  1. Every rank trained with its own policy's args (EnvReportEvent per model_id)
  2. Every policy learned, against TRAIN_REWARD_BOUNDS
  3. The leader policy reported a step for every one of the NUM_ROLLOUT rollouts - a run cut
     short by a failed deployment is not a finished one
  4. Per policy: train/grad_norm and train/loss finite and nonzero over >= 2 rollouts
  5. Per policy: train/train_rollout_logprob_abs_diff <= 0.5 - a policy's trainer and its engines,
     split across releases, are not serving different weights

Multi trainer is not bitwise-reproducible, so nothing here compares against a reference run.
```

- **Why the log-prob gate is load-bearing**: the cheapest way for this wiring to be wrong is a
  policy's trainer scoring tokens some other engine generated, which shows up in this metric and
  almost nowhere else.
- **`0.5` is a guess, not a calibration**: CI baselines sit around `0.02`; expect to tighten it once
  the test has run.
- **Assertion 3 is deliberately exact**: the first real run may have to relax it.

## Known limitations

Read these before treating a red run here as a regression.

- **The target side of both hot restart scenarios is expected to fail in the launcher already
  watching the run**: the outer launch polls the one state file it picked at install time
  (`launcher/entrypoint.py` `_follow_until_finished`, `orchestrator/observer.py`), and a take-over
  takes a state file of its own, because the restart stamp rebuilds the orchestrator object
  (`launcher/entrypoint.py` `_compute_state_file`, `launcher/hot_restart.py`). The wrapper being
  replaced catches SIGTERM and writes `exited` with exit code `143` into the *old* file
  (`orchestrator/wrapper.py`), so the outer launch raises `SystemExit(143)` within one poll while
  the run it installed trains on under the new orchestrator. The product has settled the new
  launcher's side and not the old one's; the test does not work around it.
- **`scenario_hot_restart_deterministic`'s missing engine checksum is exempted, not explained away**:
  the event log snapshot is cut between a save and the weight update of the same rollout, so the
  checksum event of the rollout a take-over resumes from is rolled back and never rewritten. The fix
  is to snapshot after the update; until then assertion 6's exemption hides anything else that could
  lose exactly that one event.
- **`scenario_hot_restart_no_checkpoint` is expected to fail on two more product gaps than the
  scenario it varies**, both read off the code rather than observed:
  - the take-over is refused before the train loop starts: `create_training_models`
    (`miles/ray/placement_group.py`) calls `rollout_executor.load(start_rollout_id - 1,
    require_state=resumed)`, and `resumed` means *the trainers were already initialized*, not *a
    checkpoint was loaded*. With no checkpoint that is `load(-1, require_state=True)`, and no save
    ever wrote `<save>/rollout/global_dataset_state_dict_-1.pt`.
  - nothing rolls the event log back: with no checkpoint under `--load`,
    `resolve_args_checkpoint_load` (`megatron_utils/megatron_config.py`) rewrites `--load` to
    `--ref-load`, which carries no `iter_*/debug_events` snapshot of this run, so
    `event_logger/checkpoint.py` `restore()` finds nothing to restore and the restarted
    orchestrator appends to the log it inherited. The redone steps are then described twice in one
    log, and `compare_inference_engine_checksums` refuses a duplicate checksum event per rollout.
    Assertion 5 asserts the missing rollback rather than working around it, because it is what the
    code does.
- **The window `scenario_hot_restart_no_checkpoint` aims at is a race**: the gate opens on the first
  finished step, but the take-over lands asynchronously. If it lands after the run's first save
  (rollout 3), the scenario fails loudly on its no-`.trash_*` assertion instead of testing the path
  it means to; the slack between the two is unmeasured.
- **A deployment waits a bounded time for the peers it dials**: an rpc submit retries an unreachable
  peer for `SUBMIT_RETRY_WINDOW_SECONDS` (60s, `miles/utils/workers/rpc/client/call.py`). Install
  order is chosen around it — driver last — but no cluster has been timed, so a slow image pull
  between releases is an untested failure mode. Registration is not the constraint: a reporter waits
  `HUB_READY_TIMEOUT_SECONDS` (3600s, `registration/reporter.py`) for a hub that does not exist yet,
  and `REPORTER_TTL_SECONDS` (240s, `registration/hub.py`) only expires reporters that have already
  reported and then gone quiet, against a 12–18s report interval.
- **Both engines of `scenario_split_deterministic` report gpu offset 0**: each inference release
  carries one engine and numbers its GPUs from its own release, so the offsets the trainer sees do
  not tell the two engines apart. Nothing here reads an offset for placement — none is colocated —
  but engine order across releases is therefore not offset-ordered.
- **The boot uuid proves one process, not the run**: it is read from the trainer controller's health
  endpoint only, so a trainer worker or an engine restarting its process without losing its pod would
  not show up in assertion 4; assertion 3's pod uid and restart count cover those.
- **Thresholds and durations are unmeasured**: `train_rollout_logprob_abs_diff <= 0.5`, every
  `est_time`, and the gate and join timeouts in `conftest_deploy/hot_restart/driver.py` are first
  guesses, to be calibrated on the first real runs.
- **A split scenario names its run uuid itself**: a launch given `--deploy-component` is refused a
  generated one (`launcher/entrypoint.py` and `miles/utils/arguments.py`, both `_resolve_run_uuid`),
  so each side names one — `conftest_deploy/split_deployment.py` for the comparison target,
  `conftest_deploy/scenario_split_multi_policy.py` for its own run. The hot restart scenarios are
  the other case: they relaunch one `all` release carrying no run uuid of their own, so the
  take-over inherits the run's uuid from `_resolve_run_uuid` reading it back off the installed
  orchestrator's argv — the one path here that recovers a run uuid from an installed release.
