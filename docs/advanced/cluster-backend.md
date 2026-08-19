---
title: Ray and Kubernetes Backend
description: Run the same training script on Ray, or let Kubernetes schedule every worker of the run.
---
Miles runs a job on one of two cluster backends. Ray is the default and needs no configuration.
Kubernetes installs the run as a helm release, so the cluster schedules every worker. The training
script is the same either way.

<Warning>

**Status.** Under active development: flags, chart values and failure semantics still change. Ray
also runs *inside* Kubernetes, and that is the well-trodden path — take this one only if you want
Miles to create the cluster's objects itself.

</Warning>

## Launch

Run everything from the repository root, with `kubectl` and `helm` on your PATH.

Install a workbench — the long-lived pod you launch from — once per namespace, from your
cluster's [`infra.yaml`](#for-cluster-administrator):

```bash
export MILES_NS="miles-$USER"
python -m miles.utils.external_utils.miles_workbench install -n "$MILES_NS" -f infra.yaml
```

It creates the namespace if missing, checks your rights, installs, and waits for Ready. Runs
launched from the pod inherit those values.

Launch from inside it:

```bash
python -m miles.utils.external_utils.miles_workbench exec -n "$MILES_NS" -- \
  bash -lc "cd /root/miles && python scripts/run_qwen3_4b.py train"
```

## Observability

**Built in**

- The launcher follows every pod's logs, prints status changes and warning events, and ends with
  the run's exit code.
- On a failure, collect the namespace's logs, describes and events into one directory — before
  cleaning up, and the whole namespace, because the explanation is usually the pod next to it:
  `python -m miles.utils.external_utils.miles_workbench collect-diagnosis -n "$MILES_NS" --output-dir ~/artifacts/miles`

**External**

- A run's pods are ordinary pods, so whatever the cluster already runs — a metrics stack, a log
  collector, the platform's own dashboards — sees them with no wiring from Miles.
- Prefer it at scale. The built-in following is meant for watching one run, not hundreds of pods.

## Clean up

```bash
python -m miles.utils.external_utils.miles_workbench stop -n "$MILES_NS" 260811-143000-042
python -m miles.utils.external_utils.miles_workbench uninstall -n "$MILES_NS"
```

`stop` removes the run and frees its GPUs; `uninstall` removes the workbench.

## Colocated engines

How `--colocate` places pods:

- **Trainer pods take whole nodes**: every card of the node belongs to one trainer pod.
- **Engine pods request no GPU**: `NVIDIA_VISIBLE_DEVICES: all` is what makes the node's cards
  visible to them anyway.
- **A scheduling gate holds each engine pod**: `miles.radixark.io/colocate-pairing`, so the
  scheduler cannot place it before its trainer has landed.
- **A pairing controller releases it**: one JSON patch pins `kubernetes.io/hostname` to the
  trainer's node and removes the gate.

Engines narrower than a node work — eight `--rollout-num-gpus-per-engine 1` engines share one
8-GPU trainer node:

- **Such an engine sees every card on the node**, so it has to be told which one is its own.
- **The controller computes that card as it seats the pod**, and writes it in the same patch that
  removes the gate, as the annotation `miles.radixark.io/base-gpu-id`.
- **The pod reads it back as `MILES_BASE_GPU_ID`**, a downward-API `fieldRef` on that annotation.
- **Nothing is computed inside the pod**, and `kubectl describe pod` shows the card it was given.

Shared memory:

- **Every pool pod mounts a memory-backed `/dev/shm`**, the same thing the docker quick start asks
  for with `--shm-size`.
- **Kubernetes' own default of 64Mi is not enough**: NCCL wants tens of Mi per peer it cannot reach
  over p2p, so a pool holding part of a node's cards dies at rendezvous
  with `No space left on device`.
- **`run.shmSize` overrides the 32Gi default**: set it in your own helm values file, the launcher
  never writes this key.

Limits:

- **The trainer still takes whole nodes**: a sub-node trainer pod is refused, for two reasons.
  - The device plugin picks the trainer's cards at runtime, so a computed card index is known to be
    the trainer's only when it holds every card of the node.
  - The engine requests no GPU, so the node's GPU accounting rests entirely on the trainer's
    request; a sub-node trainer would leave the cards the engine uses free to be given away.
- **An engine *pod* cannot be wider than a node**: an engine wider than a node is fine — it is split
  into one whole-node pod per node, each paired with an adjacent trainer pod.
- **Losing one of several narrow engines ends the run**: the controller does put the replacement pod
  back on the trainer's node, but the weight update running meanwhile waits on a rank that is not
  coming, and a single-cell trainer has no second cell to retry from.
- **A run of one whole-node engine takes a different path**: losing that engine leaves nothing to
  broadcast to at all.

## Folder convention

A run is many pods on many machines, and they share nothing but the storage `infra.yaml` mounts.
A path that is not on it is the most common way a run fails.

- Every path your script names — `/root/models`, `/root/datasets` — has to be on it.
- Copying a file into a pod is pointless: pods come and go, the mount survives.
- To run your own branch instead of the image's copy, name its sub-path under the storage root:
  `infra.paths.repos.miles: alice/miles`. `megatron` and `sglang` work the same way.

## For cluster administrator

Everything above assumes this was done once.

**Install LWS.** Miles deploys its worker pools as
[LeaderWorkerSets](https://github.com/kubernetes-sigs/lws). Install the CRDs and controller, and
grant users rights over them explicitly: LWS ships no aggregation labels, so a namespace `admin`
role does not include them.

**Give each user a namespace.** The namespace is the real boundary, not the Role: anything that
may create workloads can name another ServiceAccount and read its token. Keep privileged accounts
out of it.

**Write one `infra.yaml`.** The same file drives every Miles chart:

```yaml
infra:
  image:
    repository: radixark/miles
    tag: dev
  sharedStorage:
    type: hostPath
    hostPath: /cluster-storage
    mountPath: /cluster-storage
```

`charts/miles-run/values.yaml` shows the full shape, and each chart's `values.schema.json` is the
authoritative field list.
