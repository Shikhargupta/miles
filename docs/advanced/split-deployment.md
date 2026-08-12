---
title: Deploying the Trainer and the Inference Side Apart
description: Install a run's trainer, its inference side and everything else as separate deployments that address each other statically.
---
A run is normally one deployment: one launch brings up every worker. `--deploy-component` splits it, so
the trainer, the inference side and the orchestration script are installed by separate launches with
separate lifecycles.

<Warning>

**Status.** Under active development. Splitting a run is what lets one orchestration script drive
several trainers or several inference deployments; this page describes the single-instance split, and
[multi instance deployment](/advanced/multi-instance-deployment) the rest.

</Warning>

## The components

| `--deploy-component` | What it deploys |
| --- | --- |
| `all` (default) | everything, as one deployment |
| `trainer` | the trainer controller and its megatron ranks |
| `inference` | the inference controller, its sglang engines and its routers |
| `primary` | everything else: the rollout executor, the orchestration script, the api server, the mini ft controller, the session servers |

- `primary` is `all` minus `trainer` minus `inference`, so the four values partition the run's workers.
- One launch deploys one component. A split run is one launch per component, all with the same base
  arguments, plus the static addresses that describe the components *another* launch deploys - which only
  the `primary` launch has any of.
- `--colocate` is rejected under a split: colocated trainers and engines share gpus, so they are one
  deployment unit.

## Launching a split run

The part a launch deploys is a property of the launch, not of the training arguments: it names every
object the launch installs, so the launcher has to know it before it renders anything. Set
`MILES_SCRIPT_DEPLOY_COMPONENT` (or `ExecuteTrainConfig.deploy_component`), and the launcher passes
`--deploy-component` down to the pods itself. Naming a *different* component in the training arguments
stops the launch.

Deploy the two sides first. Each launch returns once its release is installed and prints, as flags ready to
paste, the addresses to reach it by and - for the launch that carries the orchestration script - the object
store master the other releases have to name. The addresses are derived from the release name, which is
truncated and hashed when it is long, so read them off the launch rather than deriving them:

```bash
python -m miles.utils.external_utils.miles_workbench exec -n "$MILES_NS" -- \
  bash -lc "cd /root/miles && MILES_SCRIPT_DEPLOY_COMPONENT=trainer python scripts/run_qwen3_4b.py train"
```

```bash
python -m miles.utils.external_utils.miles_workbench exec -n "$MILES_NS" -- \
  bash -lc "cd /root/miles && MILES_SCRIPT_DEPLOY_COMPONENT=inference python scripts/run_qwen3_4b.py train"
```

Then run the training itself, naming what it has to reach:

```bash
python -m miles.utils.external_utils.miles_workbench exec -n "$MILES_NS" -- \
  bash -lc "cd /root/miles && MILES_SCRIPT_DEPLOY_COMPONENT=primary python scripts/run_qwen3_4b.py train \
    --trainer-controller-addrs actor=<trainer-controller-host>:8000 \
    --inference-controller-addrs <inference-controller-host>:8000 \
    --inference-router-addrs <router-host>:8000"
```

## Addresses are given, never discovered

A deployment finds its own workers by the names its own release gives them. Nothing derives another
deployment's names, so everything that crosses a deployment boundary is an argument. A static address
always describes a component *some other* launch deploys: passing one for a component this launch
deploys itself is refused while the arguments are validated, so `--trainer-controller-addrs` is refused
on a `trainer` launch, `--inference-controller-addrs` and `--inference-router-addrs` on an `inference`
launch, and all three on an unsplit `all` launch.

- `--trainer-controller-addrs` — `host:port`, or `<role>=host:port` when the run also trains a critic.
- `--inference-controller-addrs` — `host:port`, one entry per inference deployment. A run whose
  inference deployments come and go names none of them and lets them register themselves instead; see
  [multi instance deployment](/advanced/multi-instance-deployment).
- `--inference-router-addrs` — `host:port`, or `<model>=host:port` per model. The routers live with the
  engines they serve, so the rollout executor and the session servers are given their address too.
- The object store master lives with the orchestration script, so the other deployments have to name it:
  pass the primary deployment's mooncake master address in `--mooncake-store-init-kwargs`, together with
  `--object-store-backend mooncake`. The primary launch prints that address, hash truncation and all. A
  launch that carries no orchestration script and names no master is refused while its arguments are
  validated, because a store reference is only redeemable inside the deployment that created it. The master
  comes up with the deployment that carries the orchestration script, so a trainer or inference deployment
  installed before it restarts until that master answers.

All three are required when a launch carries the orchestration script without the component it names,
and the launch fails immediately when one is missing. `--debug-train-only` deploys no engines and no
routers, so it excuses `--inference-router-addrs` only; the inference controller is reached either way.

A given address is waited for rather than assumed: a call to a statically addressed controller waits for it
to answer `/health` before the call goes out, and the routers are waited for off the event loop, so a
deployment installed a moment earlier is not a race.

Each addressed deployment is also asked which run it belongs to before it is used. The trainer controller and
the inference controller answer with their own launch's run, and the inference controller answers with the
routers it serves; a run that does not match, or a router the inference controller does not serve, stops the
launch by naming both sides. Without that check, weight updates could go to one inference deployment while
rollout samples came from another, and the run would merely look like it was not learning.

## Lifecycles are per deployment

- Each launch installs its own helm release, named `miles-run-<run id>-<component>`, and only a whole
  release is installed, upgraded or uninstalled at a time. Every object, hostname and label a deployment
  computes comes from its own release name - the api server's host included - so nothing it derives can
  point into another deployment.
- Only the release that carries the orchestration script has a "training finished" verdict: it writes
  the exit file, the launcher waits for it, and it uninstalls itself when the run ends.
- A trainer or inference release has no training to finish. Its launch returns once the release is
  installed, and the release stays up until you uninstall it:
  `python -m miles.utils.external_utils.miles_workbench stop -n "$MILES_NS" <run id> --deploy-component trainer`.
- Fault tolerance does not cross a release either. The api server and the mini ft controller answer for
  the cells of their own deployment, so a split run refuses them (`--api-server-port 0`) rather than
  reporting every trainer and inference cell as missing. Asking a split deployment for its api server host
  is refused for the same reason: nothing listens there. A trainer deployment keeps the fault tolerance
  of its own ranks, which its own controller drives.
- Failure does not cross a release. An orchestration script that loses a controller fails loud in its
  own release and tears down nothing else, so the trainer and the inference side survive it — which is
  also what [hot restart](/advanced/hot-restart) needs. A weight update opens a numbered window on the inference
  controller, and only an action carrying the open window's number closes it, so a late caller of a window
  that has already been replaced is refused rather than releasing someone else's lock. A failed broadcast is
  aborted only once the trainer has confirmed that it stopped: a failed call means the *client* gave up, and
  resuming health checking while the trainer is still broadcasting would let an engine be restarted
  mid-broadcast. That confirmation is answered by the trainer ranks that run the broadcast — each one marks
  itself in flight for as long as its (uncancellable, synchronous) broadcast body runs, and the trainer
  controller polls that mark — so a controller call that was itself cancelled or abandoned can never report
  a broadcast as finished. Every allocated cell is polled, and every worker of a cell is asked separately, so
  one worker that cannot answer never hides another worker that is still writing. A worker that cannot answer
  counts as broadcasting until its cell's processes have been killed *and* confirmed dead; the errored mark is
  not enough, because a cell is marked errored before that kill even starts, which is exactly the window in
  which a rank's synchronous broadcast body is still running. A cell whose death could not be confirmed when it
  was killed is probed again on every poll, so a rank that was only unreachable at kill time releases the window
  as soon as its death can be confirmed, rather than holding it until the provider reclaims the cell. The budget
  that re-probe is given bounds the probe itself, so a worker that keeps its connection open without answering
  leaves its death unconfirmed at the end of that budget instead of stretching the poll to the probe's own,
  much longer, timeout. A
  confirmation, once taken, is never withdrawn, and a re-probe can only turn unconfirmed into confirmed - the flag is
  written monotonically, so every path that re-probes, including a second kill of an already dead cell, can only add
  a confirmation. Only an
  explicit denial from every worker of a cell releases it; an answer nobody gave counts as broadcasting, and so does
  a liveness read that failed outright instead of answering.
  The whole poll carries a deadline of its own - not one checked between iterations, which a single unanswered
  rpc would sail past - so a caller that gave up does not leave it running against the fleet forever; when it
  expires it answers "still broadcasting" rather than "finished", and names the cells that were still counted
  as broadcasting. When the confirmation does not arrive, the window keeps its lock and
  its paused health checking and the failure says so, rather than the release quietly resuming.
  `start_update_weights` failing anywhere resumes the health checking it had paused and leaves no window open.
- A CI launch cleans up the leftover CI releases of *other* runs before installing its own; the sibling
  releases of the run it is launching are never touched.
- Relaunching a run id only resizes pools, per release, exactly as it does for an unsplit run.

## Arguments stay the user's responsibility

The launcher is the single source of a deployment's arguments: one render produces the command line of
every pod in that release, so a release is internally consistent. Across releases only the run identity is
checked — pass the same base arguments to all of them, and change them in all of them. The static addresses
are not part of that base: they belong to the launch that has to reach what it does not deploy, and neither
are `--deploy-component`, `--api-server-port` and the addresses themselves, which the trainer controller
keeps from its own launch rather than adopting from the arguments the orchestration script hands it.
Checking that the rest of the base arguments agree is a planned hardening, not something that exists today.

## What this is not

- **Not, on its own, several trainers or several engine pools.** A split names one instance of each.
  Installing one release per policy trainer, and registering engine-only deployments into the run's
  one inference controller, is [multi instance deployment](/advanced/multi-instance-deployment).
- **Not, on its own, a hot restart.** The surviving releases are only the precondition;
  [hot restart](/advanced/hot-restart) is what reattaches a new orchestration script to them.
- **Not external rollout.** [External rollout](/advanced/external-rollout) hands Miles engines it does
  not manage. Here Miles manages every worker, in deployments of its own.
