---
title: Worker Providers and Injection
description: How a process gets hold of its workers' addresses and handles, and who is allowed to know the backend.
---
Miles runs on two cluster backends (Ray actors, Kubernetes pods). Everything that
addresses workers goes through injected abstractions instead of asking a global.

## The abstractions

- `BaseWorkerProvider` — addresses and handles of workers, plus cell observation
  (`miles/utils/workers/worker_provider/base.py`).
- `ProviderFactory` — the backend-bound facade that hands out providers
  (`miles/utils/workers/worker_provider/factory.py`):
    - `cells(spec_names=...)` — the provider that observes those fleets' cells.
    - `static(worker_name=...)` — the provider that answers for a statically addressed worker.
    - `cell_operations()` — the `BaseCellOperations` this backend suspends/resumes cells with.
- Implementations: `KubernetesProviderFactory`, `RayProviderFactory`, and
  `DeferredProviderFactory`, which builds one of the two on first use.

## Who may know the backend

- Only two places fork on the backend:
    - `create_provider_factory(args)` in `miles/ray/wiring.py`, called in the first lines of
      `train.py` / `train_async.py` / `train_multi_lora_async.py`.
    - the worker-process composition root: `create_worker_provider_factory(worker_argv=...)`,
      assembled into `WorkerCtorContext.providers` where the worker process builds its context —
      `serve_inner` in a pod, `worker_bootstrap.bootstrapped_worker_class` inside a Ray actor.
- Every other module receives a `ProviderFactory`, a `BaseWorkerProvider` or a
  `BaseCellOperations` as a constructor or function argument.
- There is no registry and no service locator. A missing wiring is a `TypeError` at the call
  site, never a silent fallback to another backend.

## Where a worker's ctor kwargs are computed

- Always in the worker's own process, on both backends:
    - Kubernetes: the pod runs `serve_inner`, which calls `compute_ctor_kwargs` and then
      `WorkerClass(**ctor_kwargs)`.
    - Ray: the manager launches a subclass of the spec's worker class whose `__init__` takes
      `(spec_name, worker_argv, cell_index, worker_in_cell_index, gpu_ids)` and calls the same
      `compute_ctor_kwargs` before the wrapped constructor.
- The manager process therefore ships an identity, never a computed kwargs dict: a `ProviderFactory`
  is bound to the process that observes the cluster and cannot be serialised to another one.
- `WorkerLaunchContext` (identity only) is what `env_var` and `launch_command` receive;
  `WorkerCtorContext` adds the required `providers` and is what `ctor_kwargs` receives.

## Two static idioms, deliberately kept apart

| Idiom | Meaning |
| --- | --- |
| `ctx.providers.static(worker_name=...)` | the backend decides how to honour the request (address book under Kubernetes, named actor under Ray) |
| a spec writing `SimpleWorkerProvider(...)` directly | the spec unconditionally fixes the address book, whatever the backend is |

Do not unify them. The first is a question asked of the running cluster; the second is a
decision the spec has already made and wants to hold on both backends.

## Spec-side derivation

- Helpers that derive spec names from `args` (`compute_engine_spec_names`,
  `compute_trainer_spec_name`, …) live in the `miles.ray.specs` package.
- They stay there: `wiring.py` assembles providers, it does not decide what a run's specs are.
