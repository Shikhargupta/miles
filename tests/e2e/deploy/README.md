# Deployment E2E Tests

## Overview

- **Covered**: how a run is deployed, not what it trains — split runs and hot restarts.
- **Tests import the examples**: `examples/infra_features/{split_deployment,hot_restart}` and
  `examples/multi_policy`; only observing and asserting live here. Train args and comparison
  machinery come from `tests/e2e/ft`, never copied.
- **Never run on a cluster**: every claim below is what the code is written to do, not what was
  observed.

| Entry | Scenario | Suite | `est_time` | Labels |
| --- | --- | --- | --- | --- |
| `test_split_deterministic.py` | `scenario_split_deterministic` | `stage-c-8-gpu-h200` | 2600 | `deploy` |
| `test_hot_restart_deterministic.py` | `scenario_hot_restart_deterministic` | `stage-c-8-gpu-h200` | 5400 | `deploy` |
| `test_hot_restart_no_checkpoint.py` | `scenario_hot_restart_no_checkpoint` | `stage-c-8-gpu-h200` | 6000 | `deploy` |
| `test_split_multi_policy.py` | `scenario_split_multi_policy` | `stage-c-8-gpu-h100` | 1800 | `deploy`, `multi-policy`, `fully-async` |

- **Entries**: `register_cuda_ci(...)` + `run_ci()` only; CI gating label `run-ci-deploy`; every
  broad scope includes `deploy`. Without kubernetes an entry skips in seconds.
- **Topology**: every comparison scenario pins a private `_MODE` — 1 node, 4+2 GPUs, CP2, dense
  `Qwen3-0.6B`, 2 engines × 1 GPU; `scenario_split_multi_policy` uses its example's topology
  (per policy: 2 train + 2 rollout GPUs). No mode matrix, no `--mode`.

## Running

Needs `PYTHONPATH=.` and a miles-workbench pod (`MILES_SCRIPT_CLUSTER_BACKEND` /
`MILES_SCRIPT_NAMESPACE` / `MILES_SCRIPT_RUN_ID` are preset there). Kubernetes only.

```bash
PYTHONPATH=. python tests/e2e/deploy/test_split_deterministic.py                                    # as CI
PYTHONPATH=. python tests/e2e/deploy/conftest_deploy/scenario_split_deterministic.py run            # full pipeline
PYTHONPATH=. python tests/e2e/deploy/conftest_deploy/scenario_hot_restart_deterministic.py target \
    --dump-dir /node_public/dumps/scratch                                                           # one side
PYTHONPATH=. python tests/e2e/deploy/conftest_deploy/scenario_split_multi_policy.py verify          # assertions only
```

| Scenario kind | Subcommands |
| --- | --- |
| comparison | `run` (prepare+baseline+target+compare), `baseline` / `target`, `compare` (no GPU; re-reads `hot_restart/evidence.json` too), `generate-data` |
| `scenario_split_multi_policy` | `run` (gated on the backend), `verify` |

- **Dump dirs**: comparison scenarios use `/node_public/dumps/<TEST_NAME>/` (only `run` deletes it;
  `--dump-dir` overrides); multi policy uses `<output_dir>/multi_policy_solver_verifier/<run_id>/`.
- **Gate**: `is_cluster_ready_for_helm_runs` skips with a reason on a declared-unsupported backend;
  a failed probe asserts — exit 0 there would report green for a test that never ran.

## Comparison Criterion

- **Shared `compare_deterministic_sides`**: dumps `rel <= 0` (bitwise); metrics `rtol=0` / `atol=0`
  over `train/` and `rollout/` (`perf/` ignored, unclassified metrics fail); plus engine count,
  engine checksums, weights-moved and nonzero-gradient checks — no scenario can quietly drop one.
- **What makes bitwise possible**: deterministic mode + deterministic sglang inference +
  `--debug-deterministic-collective` + pinned NCCL algo; `--sglang-disable-radix-cache` because the
  sides spread requests differently; `--weight-decay 0` so an update depends on gradients alone.

## `scenario_split_deterministic`

```
Type: comparison (baseline=one release, target=one release per deployment), 3 rollouts
Baseline: the whole run in one release
Target, one helm release each, installed in order: TRAINER; INFERENCE e0, e1 (one engine each);
  PRIMARY last (installing it blocks until the run ends). Wiring/addresses from the example's
  address_book; ordering, shared run uuid, install verification and uninstall from
  conftest_deploy/split_deployment.py.
Assertions: zero reconfigure events on both sides; metrics/dumps bitwise; identical engine
  checksums per (rollout, engine); engine count; weights moved; nonzero gradients >= 2 rollouts.
Known bitwise risk: baseline pools 2 engines, target 2x1, so request routing differs - leaning on
  sglang determinism being engine-assignment invariant.
```

## `scenario_hot_restart_deterministic`

```
Type: comparison; both sides run the identical command, target's orchestration script replaced
      twice mid-run. ONE release (ALL), not split. 6 rollouts, --save-interval 1, 2 restarts.
Trigger: relaunch the same release with --hot-restart orchestration,rollout_executor
  (helm-upgrades those two workloads and no others).
Driver: daemon thread polling the dump volume every 5s (last saved iteration, last finished step).
  Gate i: a save S_i then a finished step F_i > S_i; gate 2 also demands S_2 >= F_1 (disjoint redo
  windows). Gate shut 3600s or 5 consecutive poll failures fail the run. On exit: join threads,
  final snapshot, write hot_restart/evidence.json.
Evidence per snapshot: pod name/uid/restartCount; statefulset+leaderworkerset generation and
  restart-at stamps; trainer rpc boot uuid off its health endpoint; read failure counts.
Assertions:
  1. Both restarts happened, no driver failure, cluster seen >= 2 times, reads answered >= half.
  2. k8s layer: no unattributed pod; pods replaced == exactly orchestrator + rollout-executor;
     the same two carry exactly NUM_RESTARTS restart stamps each, nothing else any.
  3. Process layer: one trainer boot uuid across all snapshots, seen after each restart stamp.
  4. Redo, measured off the logs (records only a lower bound): one .trash_* dir per restart
     holding steps 0..F_i with no hole; resume point = byte-identical carried-over prefix, equal
     to the snapshot beside a checkpoint; surviving log has each step exactly once; per-step
     attempts equal the measured windows, all 1 or 2.
  5. Comparison as in scenario_split_deterministic (>= 4 rollouts), target allowed to miss the
     engine checksum of each resume point (see Known limitations).
Why twice: the second take-over hits trainers already taken over once - where leaked state shows.
```

## `scenario_hot_restart_no_checkpoint`

```
Type: same as scenario_hot_restart_deterministic, but ONE restart landed before the run saved
      anything. 6 rollouts, --save-interval 4 (saves after steps 3 and 5; take-over window 0..2).
Gate (NoCheckpointGate): opens on the first finished step while no checkpoint tracker exists; a
  save seen first fails the poll (that take-over is the other scenario) and the failure limit
  fails the run.
Assertions: workloads/process as above with NUM_RESTARTS=1; redone-from-scratch: record carries
  saved_iteration_at_trigger=None; NO .trash_* exists (--load resolves to --ref-load, which holds
  no snapshot of this run, so restore() has nothing to roll back); the one log has steps 0..F
  twice with no hole and nothing thrice; the run still saved after the restart; comparison with NO
  checksum exemption (nothing was rolled back, so nothing is lost).
Why its own scenario: production saves every ~20 steps, so a restart at step 10 is this case;
  load_state finds no tracker, re-seeds, resets the optimizer and returns start rollout 0.
```

## `scenario_split_multi_policy`

```
Type: single run, no baseline (multi trainer is not bitwise-reproducible). 3 rollouts.
Recipe: examples/multi_policy demo; launcher: run_solver_verifier_gsm8k_split.py, one command per
  part. Five releases: TRAINER solver-actor / verifier-actor, INFERENCE solver / verifier
  (2 GPUs each), PRIMARY last.
Assertions: every rank trained with its own policy's args; every policy learned
  (TRAIN_REWARD_BOUNDS); the leader reported every rollout; per policy finite nonzero
  grad_norm/loss >= 2 rollouts; per policy train_rollout_logprob_abs_diff <= 0.5 - the cheapest
  wiring bug is a trainer scoring tokens some other engine generated, and it shows up here.
```

## Known limitations

Read these before treating a red run as a regression.

- **Both hot restart targets fail in the launcher already watching the run**: the take-over takes a
  new state file; the replaced wrapper writes `exited/143` into the old one, so the outer launch
  raises `SystemExit(143)` within a poll while the run trains on. The product settled the new
  launcher's side only; the test does not work around it.
- **Missing engine checksum (deterministic scenario) is exempted, not explained**: the event log
  snapshot is cut between a save and that rollout's weight update, so the resume rollout's checksum
  is rolled back and never rewritten. Fix is to snapshot after the update.
- **The no-checkpoint scenario hits two more product gaps**: `rollout_executor.load(-1,
  require_state=True)` (`resumed` means "trainers initialized", not "checkpoint loaded") dies before
  the train loop on a state file no save wrote; and nothing rolls the event log back, so redone
  rollouts carry duplicate checksum events the comparator refuses. The assertions encode both.
- **The no-checkpoint window is a race**: the gate opens on the first finished step but the
  take-over lands asynchronously; landing after the first save fails loudly on the no-`.trash_*`
  assertion. The slack is unmeasured.
- **Install order leans on `SUBMIT_RETRY_WINDOW_SECONDS` (60s)**: driver last; a slow image pull
  between releases is untested. Registration is not the constraint (reporters wait 3600s for the
  hub; the 240s TTL only expires reporters that reported and went quiet).
- **Both engines of the split scenario report gpu offset 0**: each release numbers its own GPUs;
  nothing reads offsets for placement, but engine order across releases is not offset-ordered.
- **The boot uuid proves one process**: the trainer controller's only; worker or engine process
  restarts inside a surviving pod are covered by pod uid/restartCount, not by it.
- **Thresholds and durations are unmeasured**: `logprob_abs_diff <= 0.5`, every `est_time`, gate
  and join timeouts — first guesses, calibrate on the first real runs.
- **Run uuid**: the product refuses to generate one for a split launch, so each split scenario
  names one itself; the hot restart scenarios instead relaunch an `all` release with none, and the
  take-over reads the uuid back off the installed orchestrator's argv.
