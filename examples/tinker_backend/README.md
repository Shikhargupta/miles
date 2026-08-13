# Tinker-compatible backend

Serve many LoRA training runs on one shared base model through a
[tinker](https://tinker-docs.thinkingmachines.ai/)-style operation API: clients
drive training with explicit `forward_backward` / `optim_step` operations and
sample through the shared engines — no dataset, no reward function, and no
batch schedule on the server.

```
client ──HTTP──> TinkerController (head node)
                   ├─ registration plane   /adapter_runs (the only HTTP routes in v1)
                   ├─ operation ledger     enqueue → claim → complete → ack (Ray actor API;
                   │                       a tinker /api/v1 HTTP frontend is a later PR)
                   └─ serving plane        sglang router (direct)
trainer ranks <──Ray── driver loop (train_tinker_backend.py)
```

## Launch

```bash
python train_tinker_backend.py \
  --tinker-backend \
  --multi-lora-n-adapters 4 \
  --lora-rank 32 --lora-alpha 64 \
  --target-modules all-linear \
  --hf-checkpoint Qwen/Qwen3-0.6B \
  ... # the usual megatron/sglang flags; see run_tinker_backend.py
```

Key flags:

| flag | meaning |
|------|---------|
| `--tinker-backend` | enable the operation backend (requires `--multi-lora-n-adapters > 0`) |
| `--multi-lora-n-adapters N` | fixed slot count; a registration binds a slot for life (queue when full) |
| `--lora-rank` / `--lora-alpha` | deployment-wide ceiling / fixed alpha — clients may lower `rank`, never set `alpha` |
| `--multi-lora-disable-service-mode` | exit once all adapters retire (by default the service keeps serving with zero adapters) |
| `--tinker-max-coalesce-wait-s` | how long one train call coalesces additional ready client batches |
| `--tinker-max-empty-wait-s` | idle-queue yield back to the control phase (keep this small) |

### Activation recompute (memory saving)

`--recompute-granularity selective` is always supported (default
`--recompute-modules core_attn`; add `moe_act` to also recompute the MoE
activation with grouped GEMM). `--recompute-granularity full` — and `moe` in
`--recompute-modules` when expert modules are targeted — additionally
requires a Megatron-Bridge whose PEFT recompute patch recognizes multi-LoRA
`.adapters.<slot>.` params (radixark/Megatron-Bridge#27, branch `bridge` @
`688d34b8`): multi-LoRA trains adapter-only, so those checkpointed regions
replay grad-enabled only because that patch forces TransformerBlock inputs to
require grad. Launch probes the installed bridge and refuses the two shapes
on an unfixed one, where every adapter gradient is silently zero and the job
steps forever at `grad_norm=0.0` without learning (4xH200 GPT-OSS 20B repro,
2026-08-12; full recompute re-validated training real gradients on the fixed
bridge, same config).

## Operation contract

`enqueue_operation(name, operation_id, ordinal, kind, payload)` — ordinals are
consecutive per registration starting at 1; arrival may be out of order
(gap-buffered, and a hole-filling ordinal is always admitted), execution is
strictly ordinal-ordered; retries with the same `operation_id`, same ordinal,
and identical payload return the original operation — anything else is a
typed conflict.

| kind | payload | success result |
|------|---------|----------------|
| `forward_backward` | `{samples: [Datum...], loss: {loss_fn, loss_fn_config?}}` | `{logprobs: [[...]], metrics: {"loss:sum", "unmasked_tokens:sum"}}` |
| `forward` | `{samples: [Datum...]}` | `{logprobs: [[...]]}` (zero gradient, structurally) |
| `optim_step` | `{adam_params: {learning_rate, beta1, beta2, eps, weight_decay, grad_clip_norm}}` | `{grad_norm, learning_rate}` |
| `save_weights_for_sampler` | `{}` | `{serving_version, serving_name}` — completes only after the weights are live |
| `save_state` | `{tag?, ttl_seconds?}` | `{path, step}` (named states are immutable) |
| `load_state` | `{path}` | `{step, path}` (re-publishes on the next push) |

`Datum = {tokens, response_length, loss_mask, loss_weights?, advantages?, rollout_log_probs?}`
— per-token channels align with the response span. Losses reduce as plain
token sums (`Σ(-logp·w)` for `cross_entropy`), so K chunked forward_backward
calls accumulate exactly like one; `loss_weights` own the scale and no server
normalization or scheduler ever touches a tinker slot. Result `metrics` use
the SDK combiner's `name:reduction` keys.

Operation states: `QUEUED → CLAIMED → SUCCEEDED | FAILED(user|server) | CANCELLED`;
poll `get_operation`, then `ack_operation` to release the record. In v1 these
verbs are the controller actor's Ray API (registration/status are the only
HTTP routes); backpressure raises a retryable `OperationBackpressure` — the
future tinker HTTP frontend maps it to 429 + Retry-After, never to a 4xx the
SDK treats as fatal. Deregistering fences every open operation of that
registration as `FAILED(user)`.

## v1 compatibility matrix

Supported: text-only input; the synchronous training loop; 1-D shifted
targets; `loss_fn ∈ {cross_entropy, importance_sampling, ppo}` (per-op clip
config); per-call AdamParams; multi-chunk gradient accumulation with
independent `optim_step`; latest-only sampler weights behind the publish
barrier; named immutable `save_state` / `load_state` (create-from-checkpoint
included, shape-fenced); optional `num_step` auto-retirement.

Explicitly rejected (boundary error, never a silent fallback): multimodal
inputs; nested `(N, K)` top-K targets; other loss functions (CISPO, DRO, ...);
client-set `alpha`; non-finite/out-of-domain AdamParams; a loss's required
per-token channels missing; `response_length == len(tokens)` (targets are
shifted); async/off-policy sampling against pinned snapshots;
cross-world-size state restore; state restore into a slot whose per-rank
optimizer ownership differs from the save (cross-slot restore under DP
sharding — always safe under DP=1); idle slot GC.

## Files

- `run_tinker_backend.py` — disaggregated launch (`prepare` / `serve` / `train`)
- `adapters/example.yaml` — CLI pre-registration example (`--multi-lora-adapter example adapters/example.yaml`)
