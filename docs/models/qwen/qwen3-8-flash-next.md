---
title: Qwen3.8-Flash-Next
description: RL recipe for Qwen3.8-Flash-Next, the GDN + QSA hybrid MoE preview of the Qwen4 architecture.
---

Implementation: [`radixark/miles#2777`](https://github.com/radixark/miles/pull/2777). It goes
with the SGLang
[`sglang-miles-qwen38next`](https://github.com/sgl-project/sglang/tree/sglang-miles-qwen38next)
branch and [`radixark/Megatron-LM#89`](https://github.com/radixark/Megatron-LM/pull/89); the
image in section 3 pins all three.

## 1. Model Introduction

[Qwen3.8-Flash-Next](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next)
is Qwen's **176 B-parameter (6 B active) GDN + QSA hybrid Mixture-of-Experts preview of the
Qwen4 architecture**. Despite the name it is not a variant of the dense Qwen3.8-27B — it
continues the *Next* line, and the architecture is a different one.

- **48 layers, hybrid**: 36 GDN linear-attention layers + 12 QSA sparse full-attention layers.
- **512-expert MoE** at top-10, plus a gated shared expert.
- **Hyper-connections** in place of the block layernorms.
- **PLE**: a frozen, host-resident n-gram embedding table, ~102 GB for the full model.
- Hidden 2560, 24 attention heads / 2 query groups, vocab 248320, rotary base 1e7.
- MTP is not mapped yet; the recipe trains the base stack only.

## 2. Supported Variants

| Variant | `--model-name` | Layers | GPUs |
|---|---|---|---|
| Full | `Qwen3.8-Flash-Next` | 48 | 8 × 4 |
| Smoke slice | `Qwen3.8-Flash-Next-4layer` | 4 | 1 × 4 or 1 × 8 |

## 3. Environment Setup

Use `docker.io/radixark/miles:qwen38next` — the rolling `radixark/miles:dev` image with the
three moving parts checked out at the versions this recipe was built against, multi-arch so
the same tag serves GB300 and x86 nodes.

| Component | Pinned at |
|---|---|
| miles | [`#2777`](https://github.com/radixark/miles/pull/2777) `afd78afd` |
| SGLang | `sglang-miles-qwen38next` `599d7403` |
| Megatron-LM | [`#89`](https://github.com/radixark/Megatron-LM/pull/89) `e8f57451` |

```bash
hf download Qwen/Qwen3.8-Flash-Next --local-dir /root/models/Qwen3.8-Flash-Next
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
```

The reference checkpoint has to be converted first — `--ref-load` resolves to
`<--ckpt-dir>/<megatron_model_type>_torch_dist`:

```bash
cd /root/miles
MODEL_ARGS_LINE="$(python3 miles/utils/external_utils/model_args_utils.py qwen3.8-flash-next)" || exit 1
read -ra MODEL_ARGS <<< "${MODEL_ARGS_LINE}"
CONVERT_KEEP_PP1=1 PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" \
   --hf-checkpoint /root/models/Qwen3.8-Flash-Next \
   --save          /root/ckpt/qwen3.8-flash-next_torch_dist \
   --tensor-model-parallel-size 2 --pipeline-model-parallel-size 1
```

## 4. Launch

Bring up a ray cluster across the nodes, `export MILES_SCRIPT_EXTERNAL_RAY=1`, then on the
head node:

```bash
cd /root/miles
python scripts/run_qwen3_8_next.py train \
  --model-name Qwen3.8-Flash-Next \
  --num-nodes 8 --num-gpus-per-node 4 \
  --num-rollout 5 --rollout-max-response-len 4096
```

Smoke slice on one node:

```bash
python scripts/run_qwen3_8_next.py train \
  --model-name Qwen3.8-Flash-Next-4layer --num-nodes 1 --num-gpus-per-node 8
```

| Shape | TP | PP | EP | Rollout engine |
|---|---|---|---|---|
| 8 × 4 (full) | 2 | 8 | 4 | 8 GPUs, SGLang TP 8 / EP 8 |
| 1 × 4 (4layer) | 2 | 2 | 2 | 4 GPUs, SGLang TP 4 / EP 4 |
| 1 × 8 (4layer) | 2 | 2 | 4 | 4 GPUs, SGLang TP 4 / EP 4 |

GRPO on DAPO-Math-17k in thinking mode, Adam at `lr 1e-6`, `max_tokens_per_gpu 8192`, full
uniform recompute. Rollout is colocated, with the trainer offloaded to disk. The GDN path
uses `flashqla` in the trainer and `flashinfer` prefill in SGLang; `QSA_BACKEND=triton`
selects the sparse-attention kernel.

## 5. What to Check

The 4-layer CI test gates `train/grad_norm`, `train/ppo_kl`,
`train/train_rollout_logprob_abs_diff`, `train/train_rollout_kl` and `rollout/raw_reward`.
On a fresh bring-up read `train/train_rollout_logprob_abs_diff` first — it covers the GDN,
QSA and hyper-connection paths at once.
