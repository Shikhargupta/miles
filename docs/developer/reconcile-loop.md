---
title: "Reconcile Loop"
description: "A minimal, level-triggered controller runtime for Miles, and how it maps onto the Kubernetes Go stack."
---

## What this is

- A deliberately small port of the [client-go informer stack](https://github.com/kubernetes/client-go/tree/master/tools/cache) plus [controller-runtime](https://github.com/kubernetes-sigs/controller-runtime).
- Tracks fleets of processes — SGLang engine cells, trainer cells — that appear, disappear and restart mid-run.
- The alignment record: **every deviation from Go must appear below with a reason.**

## Modules

All under `miles/utils/workers/reconcile/`.

| Ours | Does | Go |
| --- | --- | --- |
| `k8s_api.py` | LIST/WATCH calls; the only module importing `kubernetes_asyncio` | [typed client `List` / `Watch`](https://github.com/kubernetes/client-go/blob/master/kubernetes/typed/core/v1/pod.go) |
| `k8s_reflector.py` | Cursor bookkeeping, relist on cursor rejection | [`cache.Reflector`](https://github.com/kubernetes/client-go/blob/master/tools/cache/reflector.go) |
| `source_event.py` | Reflector-to-loop wire format | [`watch.Event`](https://github.com/kubernetes/apimachinery/blob/master/pkg/watch/watch.go) + [`DeltaFIFO`'s `Replace` boundary](https://github.com/kubernetes/client-go/blob/master/tools/cache/delta_fifo.go) |
| `object_store.py` | Cache, parent index, segment buffering, replace with deletion synthesis | [`cache.Store`](https://github.com/kubernetes/client-go/blob/master/tools/cache/store.go) + [`DeltaFIFO.Replace()`](https://github.com/kubernetes/client-go/blob/master/tools/cache/delta_fifo.go) |
| `work_queue.py` | Insertion-ordered key dedup with a wakeup | [`workqueue`](https://github.com/kubernetes/client-go/blob/master/util/workqueue/queue.go) |
| `retry_scheduler.py` | Per-key exponential backoff, latest-wins timers | [rate limiter](https://github.com/kubernetes/client-go/blob/master/util/workqueue/default_rate_limiters.go) + [delaying queue](https://github.com/kubernetes/client-go/blob/master/util/workqueue/delaying_queue.go) |
| `source_stream_driver.py` | Open, sync, reopen the stream; pump events into the store | [informer `Run` / `processLoop`](https://github.com/kubernetes/client-go/blob/master/tools/cache/controller.go) |
| `loop.py` | Lifecycle, the single worker, resync | [controller-runtime `Controller`](https://github.com/kubernetes-sigs/controller-runtime/blob/main/pkg/internal/controller/controller.go) |

Four rows are not 1:1. Each is the shadow of a **Dropped** / **Replaced** row below — the class died with its feature, the remainder landed on a neighbor:

- **`ObjectStore` absorbed `DeltaFIFO.Replace()`** — Go's split needs `KnownObjects: indexer`, a pointer back to the store, because deletion synthesis must know what is cached. Splitting recreates that back-pointer to buy a name.
- **No `DeltaFIFO`** — without delta coalescing it has no FIFO, no `Pop`, no delta chain.
- **`RetryScheduler` absorbed both retry pieces** — ours is latest-wins, not a `readyAt` heap, so splitting yields Go names over non-Go semantics.
- **`SourceEvent` has no Go counterpart** — Go pushes into a `Store` by method call. A generator gives teardown and cancellation for free, and lets any non-Kubernetes source be an ordinary generator.

## Decisions per module

### `k8s_reflector.py`

| Upstream | Solves | Decision | Reason |
| --- | --- | --- | --- |
| Reflector | Move remote changes into a local cache reliably | **Kept**, Kubernetes only | Ray / external-URL backends emit in-process: no cursor, no replay window |
| `watchHandler` per-event metadata failure | One malformed frame must not stop the watch | **Kept**: log, skip, advance past it | Tearing down reconnects at the same cursor, replays the same frame, wedges the watch until expiry |
| Bookmarks | Keep an idle cursor fresh | **Kept** | Free server-side, avoids relists |
| `IsTooLargeResourceVersion` | A cursor from the future is never satisfied | **Kept**: 504 joins 410 | Otherwise a rolled-back backend freezes the store forever. A plain gateway timeout costs one LIST |
| `ListAndWatch` → `Run` → LIST again | Refresh after every watch ends | **Dropped**; reopen WATCH from the cursor | A LIST per timeout dominates an idle reflector's cost, and the cursor is still valid |
| `BackoffUntil` | Keep a relist storm off the apiserver | **Replaced** by one flat `retry_delay` | One reflector, small label-scoped LIST |
| LIST pagination | Huge collections | **Dropped** | Thousands of pods at most |

### `object_store.py`

| Upstream | Solves | Decision | Reason |
| --- | --- | --- | --- |
| Store | Read without hitting the apiserver | **Kept**, a plain `dict` | Single-threaded asyncio: no locks |
| `Replace()` on relist | Deletions missed while disconnected | **Kept**, store-side | Ghost cells are forever. Store-side also survives a whole stream reopening, which a reflector-side diff cannot remember across. Costs one event type (`SyncStart`) |
| Indexer | Large-scale reverse lookup | **Dropped**; `dict[key, parent]` scanned | The parent map is already the index |
| `EnqueueRequestForOwner` | Child event to parent key | **Kept** as `key_map`; unmappable objects dropped with an error | Cells are not Kubernetes objects, so the parent comes from labels. One bad pod must not stall the fleet |
| DeltaFIFO | Delta coalescing | **Dropped** | Reconcile reads a snapshot, so the queue needs key dedup, never a delta chain |

### `work_queue.py`

| Upstream | Solves | Decision | Reason |
| --- | --- | --- | --- |
| workqueue | The scheduling core | **Kept** as a dedup set; delayed retry lives in `retry_scheduler.py` | With one worker, the dirty/processing protocol collapses into the set |
| `ShutDown` vs `ShutDownWithDrain` | Finish in-flight work first | **Dropped**; `stop()` runs once, after `start()` has returned: cancel everything, then wait. Awaiting it inside reconcile asserts — use `asyncio.create_task(loop.stop())` — and a hung `start()` is aborted by cancelling its task, not by `stop()` | Drain exists for many Go workers. One worker means one in-flight key, and reconcile is idempotent, so abandoning it costs a re-derivation |

### `retry_scheduler.py`

| Upstream | Solves | Decision | Reason |
| --- | --- | --- | --- |
| Delaying queue on earliest `readyAt` | Rate-limited retry | **Replaced** by latest-wins: a new failure cancels the pending timer | The delay always matches current state, with no deadline comparison |
| Bucket rate limiter | Cap retry pressure | **Dropped** | In-memory accounting behind one worker: no backend to protect |
| `Result{RequeueAfter}` | Timed re-check | **Dropped** | Backoff plus resync covers it |

### `source_stream_driver.py`

| Upstream | Solves | Decision | Reason |
| --- | --- | --- | --- |
| Cache-before-notify ordering | Handlers never read a stale state | **Kept** | Correctness, not volume |
| WaitForCacheSync | Do not decide on a half-filled cache | **Kept**: `run()` + `wait_for_sync()`, the [`SyncingSource`](https://github.com/kubernetes-sigs/controller-runtime/blob/main/pkg/source/source.go) shape; `start()` awaits it | A partial engine list at step 0 would silently shrink the fleet |
| `source.Channel` | External event injection | **Dropped** | Miles-internal events are method calls |

### `loop.py`

| Upstream | Solves | Decision | Reason |
| --- | --- | --- | --- |
| Resync period | Level-triggered backstop | **Kept**, re-enqueues parents that still have members | Go's own limit: a parent that lost its last member is not re-driven |
| `MaxConcurrentReconciles` | Overlap reconcile I/O | **Dropped**; one worker | controller-runtime's default. No I/O to overlap |
| Predicates | Save reconciles at scale | **Dropped** | Reconcile is cheap |

### Dropped wholesale

| Upstream | Reason |
| --- | --- |
| SharedInformer | One or two consumers per object type |
| Manager, leader election, metrics, webhooks, multi-GVK cache, cached client | In-process objects, not a deployed operator |
| kubebuilder / Operator SDK | One resource type, hand-written |
| Thread-safe store / DeltaFIFO / workqueue | Under asyncio state mutates only between `await` points. Removes the data-race class, not the interleaving class: "check, await, mutate" still needs FSM discipline |

Retry lives at two levels:

- `KubernetesReflector.retry_delay` — recovered without ending the stream: dropped watch, failed LIST, expired cursor.
- `ReconcileLoop.source_retry_delay` — the outer net for a stream that dies for good. In-process registries will reach it.

Cleanup raises unless raising would hide a worse error:

- `stop()` → `SourceStreamDriver.aclose()` has nothing in flight, so a stream that fails to close propagates.
- `_aclose_while_unwinding` and the watch's `finally` run while another exception is unwinding. There a close failure would replace the stream error being logged, or swallow the 410 that `watch()` needs in order to relist, so it is logged with a stacktrace instead.

`kubernetes_asyncio` is imported lazily so the loop stays importable without a Kubernetes backend, and declared in `requirements.txt` for the test suites.

## Invariants

1. Reconcile gets a key only and re-derives from the store via `get_by_parent(key)`. It must not block on I/O: one worker serves the fleet.
2. The store is updated before the key is enqueued, and hands out the source's own objects — read-only, as with a client-go cache.
3. No reconcile before the initial list is consumed; `start()` is that barrier.
4. A key is never reconciled concurrently with itself; delivery is at-least-once, so reconcile must be idempotent.
5. Per-key exponential backoff; a later failure replaces the pending timer, a success cancels it.
6. A relist must synthesize deletions, or removed objects drift forever.

## Test layers

A fake encodes what we *believe* a dependency does, so it can never catch a wrong belief. Each real layer exists only for what no cheaper layer can prove. All three run on every PR.

| Layer | Where | Proves |
| --- | --- | --- |
| Fakes + fake clock | `tests/fast/utils/workers/reconcile/` | Control flow, timing, shutdown |
| Real apiserver, no kubelet | `tests/e2e/k8s_apiserver/` | API semantics: cursors, watch timeouts, real 410, relist |
| kind cluster | `tests/e2e/k8s_kind/` | What only a kubelet produces: Running, restarts, graceful deletion, bookmarks |

- Each environment directory imports no Miles code, so it can be verified before anything uses it.
- `pytest tests/e2e/k8s_apiserver tests/e2e/k8s_kind` self-provisions from a Docker daemon: etcd and the apiserver as containers, a pinned kind binary on demand. `MILES_K8S_KEEP=1` leaves the environment up, `MILES_K8S_KUBECONFIG=<path>` reuses a cluster, `MILES_K8S_REQUIRE=1` (default in CI) turns missing Docker into a failure.
- Cursor expiry needs a second apiserver with `--watch-cache=false`: with the cache on, an old `resourceVersion` is still served after etcd is compacted, so nothing can invalidate a live cursor.
