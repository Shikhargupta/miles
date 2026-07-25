---
title: Agentic Chat Templates (TITO)
description: How to turn on and verify Token-In-Token-Out (TITO) for multi-turn agentic rollout.
---

# Agentic Chat Templates (TITO)

Multi-turn agentic rollout in Miles runs on **TITO** (Token-In-Token-Out): each turn's token sequence is a bit-perfect prefix of the next, so the trainer sees exactly the tokens the engine produced — no re-tokenization, no drift. The *why* is in the blog ([No Token Left Behind](https://lmsys.org/blog/2026-05-13-no-token-left-behind/)); this page is *how*.

Your harness only ever sends and receives **OpenAI chat messages**, never tokens. Miles keeps the per-trajectory append-only token buffer (ids + logprobs + routed experts) internally and ships it straight to training.

## Prerequisites

Your rollout loop must keep two invariants, or TITO is rejected at runtime:

- **Append-only messages.** Each turn = previous messages + new ones on the tail; past turns are never edited. The only exception is retrying the latest turn — a single-step rollback to the last assistant checkpoint. Diverging earlier, or rolling back more than one turn, is rejected.
- **Appended roles fit the fixed template.** `--tito-model` selects a family whose `FixedTemplate.allowed_append_roles` declares which of `tool`, `user`, `system`, and `assistant` may be appended. The default capability is all four roles; a family narrows it only when its fixed renderer cannot preserve a role append-only. Appending an unsupported role is rejected at runtime.

## Pick your `--tito-model`

No auto-detection — pick the family matching your model. For every family, Miles resolves one `FIXED_TEMPLATE` registration from `--tito-model` alone. The registration owns the bundled Jinja template (or HuggingFace-native template), fixed kwargs, and append-role capability. A non-default family rejects `--chat-template-path` overrides and conflicting fixed kwargs; use `--tito-model default` for a custom renderer.

| Your model | `--tito-model` | `tool` | `user` | `system` | `assistant` |
|---|---|---|---|---|---|
| Qwen3 | `qwen3` | ✅ | ✅ | ✅ | ✅ |
| Qwen3.5 | `qwen35` | ✅ | ✅ | ❌ | ✅ |
| Qwen3-Next | `qwennext` | ✅ | ✅ | ✅ | ✅ |
| GLM-4.7 / GLM-5 | `glm47` | ✅ | ✅ | ✅ | ✅ |
| NVIDIA Nemotron 3 Super / Ultra | `nemotron3` | ✅ | ✅ | ✅ | ✅ |
| Kimi K2.5 / K2.6 | `kimi25` / `kimi26` | ✅ | ✅ | ✅ | ✅ |
| MiniMax M2.5 / M2.7 | `minimax_m25` / `minimax_m27` | ✅ | ✅ | ❌ | ✅ |
| DeepSeek-V3.2 / V4 | `deepseekv32` / `deepseekv4` | ✅ | ✅ | ✅ | ✅ |
| anything else | `default` | ✅ | ✅ | ✅ | ✅ |

`allowed_append_roles` defaults to the maximal four-role surface. That default is a capability claim, not proof that an arbitrary native template is prefix-stable: verify the selected model, and explicitly narrow the family registration when a role is known not to work. More models and verification history live in [issue #712](https://github.com/radixark/miles/issues/712).

## Turn it on

```bash
ROLLOUT_ARGS+=(
   --use-session-server          # entry point for TITO session tracking
   --hf-checkpoint Qwen/Qwen3-4B
   --tito-model qwen3
)
```

## Example

A full multi-turn agentic setup on the session-server TITO path lives in [`examples/experimental/swe-agent-v2`](https://github.com/radixark/miles/tree/main/examples/experimental/swe-agent-v2): its launchers wire `--use-session-server` + `--tito-model glm47` against a real SWE agent.

## Add a new model

Models in the table are verified by Miles maintainers — just pick the family. To support a new model, register a `TITOTokenizer` subclass plus one `FIXED_TEMPLATE` containing its fixed Jinja path or HuggingFace-native kwargs in [`tito_tokenizer.py`](https://github.com/radixark/miles/blob/main/miles/utils/chat_template_utils/tito_tokenizer.py). Its append capability starts with all four roles; narrow `allowed_append_roles` on that same registration only when verification proves a restriction.

The verification design has four layers:

1. An independent test manifest lists the expected capability of every non-default family. The oracle never derives its expected roles from the production `FixedTemplate`.
2. A CPU matrix runs 10 actual appendix shapes — the four single roles plus tool/user/system/assistant combinations — against every family in two render modes. Every cell must be `PASS` or a specific `EXPECTED_REJECT`; marker checks catch templates that silently drop a message. The matrix has 220 attempted cells and zero skips.
3. Production-shape CPU tests exercise TITO decode/merge round trips and explicitly account for invalid request boundaries and role-gate rejections. Session tests verify that client-injected assistant messages are not mistaken for generated rollback checkpoints.
4. GPU tests cover the remaining runtime integration: real model output, parser and stop-token behavior, session HTTP state, rollback, and mismatch classification. They do not duplicate the exhaustive CPU matrix.

For example, the CPU matrix keeps every system case for the restricted families. Qwen3.5 records them as `EXPECTED_REJECT(exception)`, while MiniMax M2.5/M2.7 records them as `EXPECTED_REJECT(dropped-message)`.

Run both scripts below — either an unexpected CPU outcome or a GPU failure blocks the model.

```bash
# CPU / fast — rendered token sequence is append-only
python scripts/tools/verify_chat_template.py \
    --model <hf-id> --tito-model <family>

# GPU / e2e — still holds under real model inference
python scripts/tools/verify_session_tito_tokenizer.py \
    --hf-checkpoint <hf-id> --tito-model <family> \
    --sglang-reasoning-parser <rp> --sglang-tool-call-parser <tcp> --rollout-num-gpus-per-engine 1
```
