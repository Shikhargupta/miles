# Deployment E2E Tests

## Running

Needs `PYTHONPATH=.` and a miles-workbench pod (`MILES_SCRIPT_*` env preset). Kubernetes only:
entries register via `register_cuda_ci` and fail with a reason on any other backend.

```bash
PYTHONPATH=. python tests/e2e/deploy/test_split_deterministic.py                          # as CI
PYTHONPATH=. python tests/e2e/deploy/conftest_deploy/split/scenario_split_deterministic.py run  # via app
# hot restart is two levels: the mode is a subcommand of its own
PYTHONPATH=. python tests/e2e/deploy/conftest_deploy/hot_restart/scenario_hot_restart_deterministic.py \
    checkpointed run
```

- **Subcommands**: comparison scenarios expose `run` / `baseline` / `target` / `compare` (no GPU) /
  `generate-data`; the multi policy one exposes `run` / `verify`; the realistic soak exposes `run`
  only. The hot restart deterministic app nests those under one subcommand per mode
  (`checkpointed`, `no_checkpoint`).
- **Dump dirs**: `/node_public/dumps/<TEST_NAME>/` (only `run` deletes it). `--dump-dir` overrides
  it for `baseline` / `target` / `compare` only; `run` and the realistic soak always resolve it
  from the test name. Multi policy: `<output_dir>/multi_policy_solver_verifier/<run_id>/`.

## Test Specifications

### `scenario_split_deterministic`

```
Type: comparison (baseline=one release, target=one release per deployment)
Steps: 3 rollouts

1. Baseline: the whole run in one release
2. Target, installed in order: TRAINER; INFERENCE e0, e1 (one engine each); PRIMARY last
   - installing PRIMARY blocks until the run ends
   - addresses from the example's address_book; ordering, shared run uuid and uninstall from
     conftest_deploy/split/split_deployment.py
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
3. Assert per policy: train_rollout_logprob_abs_diff <= 0.1 (the cheapest wiring bug - a
   trainer scoring another engine's tokens - shows up here)
```

### `scenario_hot_restart_deterministic`

```
Type: comparison (baseline=untouched, target=same command, orchestration script replaced mid-run)
Steps: 6 rollouts
Timing: exact - the run sleeps forever at the scheduled step boundary (the fault-injection
        machinery's sleep-forever action) and the driver relaunches the frozen run, so where a
        take-over lands is pinned, not raced. The plan naming that step is delivered through a
        file under the base dump dir (not under either side's, which each run deletes), so the
        relaunch repeats the installed argv byte for byte: every worker pod's command carries
        those arguments, and an in-place relaunch may only rebuild the orchestration side. The
        frozen run writes a sentinel beside that plan when it parks, which is what the driver
        gates on
Modes: checkpointed  - --save-interval 2 (saves after 1, 3, 5), 2 restarts: restart 1 frozen
                       between steps 2 and 3 (resumes save 1), restart 2 frozen between steps
                       4 and 5 (resumes save 3)
       no_checkpoint - --save-interval 4 (saves after 3 and 5), 1 restart frozen between steps
                       1 and 2, before anything was saved
Entries: test_hot_restart_checkpointed.py, test_hot_restart_no_checkpoint.py

1. Relaunch the same command + --hot-restart orchestration,rollout_executor per the mode
2. Assert workloads: only orchestrator + rollout-executor rolled (pod uid / restartCount / stamps)
3. Assert process: one trainer rpc boot uuid throughout, answering the take-over's fresh client
4. Assert redo, measured off the logs, per mode:
   - checkpointed: one .trash_* per restart; resume point == the pinned save (the snapshot
     beside that checkpoint), so the run resumed there, not at step 0; the redone steps are
     exactly the pinned (save, frozen step] windows; per-step attempts all 1 or 2
   - no_checkpoint: record carries no saved iteration; NO .trash_* (the run's --load resolves to
     --ref-load, which holds no snapshot to restore); steps 0..1 appear exactly twice, nothing
     thrice; the run still saves after the restart
5. Compare: bitwise as in scenario_split_deterministic, engine checksums included, no exemption

checkpointed lands every take-over on a non-save step, so unsaved steps are rolled back and
redone; no_checkpoint has nothing to resume from and starts over at rollout 0.
```

### `scenario_hot_restart_realistic_gsm8k`

```
Type: single run, ft's scenario_realistic_gsm8k with hot restarts instead of kills
Steps: as scenario_realistic_gsm8k
Injection: HotRestartFaultForm at random intervals via the ft fault-injection plan, seed logged
Eligibility: none - every draw fires, wherever the run stands. A draw before the first save takes
        over a run holding no checkpoint, which legitimately starts again from --ref-load; that
        is a product path this soak is meant to cover, not skip. A draw lands once the run has
        redone a step it had already trained (a rolled-back log, or step 0 trained twice).
Load-bearing: this scenario adds --save/--load of its own plus --save-interval 10 (a take-over
        resumes from the last checkpoint, so the interval bounds what one costs); mean seconds
        between draws defaults to 1800 (--hot-restart-interval-seconds)

1. Run the realistic gsm8k recipe while the plan injects hot restarts
2. Assert: gsm8k reward improves as in scenario_realistic_gsm8k
3. Assert: at least MIN_HOT_RESTARTS take-overs landed, no injection attempt failed for any
   reason, every relaunch thread finished without raising (that last one is where the run's own
   metric verdict surfaces), every landed take-over stamped the orchestrator and the
   rollout-executor exactly once, no other workload was rolled or lost a pod, one trainer boot
   uuid throughout, and no take-over threw away more than one save interval
4. Artifact: what each take-over cost (index, checkpoint held, step reached) is written to
   <dump_dir>/hot_restart/evidence.json

Hot restart rides the ft injection machinery so a future soak can mix it with pod kills.
```

