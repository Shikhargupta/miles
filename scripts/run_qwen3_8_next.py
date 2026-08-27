"""Qwen3.8-Flash-Next (Qwen4Exp) RL training on the rdx-gb300 slurm cluster.

Deliberately narrower than run_deepseek_v4.py: one model, one cluster, the paths
this project actually uses. The heavy lifting (ray job submit, env plumbing) stays
in miles.utils.external_utils.command_utils.execute_train.

Assumes an ALREADY RUNNING ray cluster (MILES_SCRIPT_EXTERNAL_RAY=1): start a
per-node container on the slurm allocation, `ray start --head` on one node and
`ray start --address` on the rest (fabric IPs via `getent hosts $(hostname)`,
node-local TRITON/INDUCTOR cache dirs), then run this script in the head
container.

Usage (inside the head-node container):
    python scripts/run_qwen3_8_next.py train --num-rollout 5
"""

from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_MODEL_LOCAL = "/scratch/models/Qwen3.8-Flash-Next"  # node-local NVMe, staged on all 8 nodes
_TORCH_DIST = "/data/home/zzeng/ckpt/qwen3.8-flash-next_torch_dist"
_DATA_DIR = "/data/home/zzeng/datasets"
_SAVE_DIR = "/data/home/zzeng/rl_runs"
_MEGATRON_PATH = "/data/home/zzeng/repos/Megatron-hcslot"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    num_nodes: int = 8
    num_gpus_per_node: int = 4
    output_dir: str = "/data/home/zzeng/rl_runs/shared"
    run_id: str = "qwen38next-dapo"
    num_rollout: int = 5
    rollout_max_response_len: int = 4096
    check_weight_update: bool = True
    enable_r3: bool = False
    skip_saving: bool = True
    extra_args: str = ""


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    total_gpus = args.num_nodes * args.num_gpus_per_node
    assert total_gpus == 32, "the parallel config below is shaped for 8 nodes x 4 GPUs"

    ckpt_args = (
        f"--hf-checkpoint {_MODEL_LOCAL} "
        f"--ref-load {_TORCH_DIST} "
    )
    if not args.skip_saving:
        load_save_path = f"{_SAVE_DIR}/{args.run_id}/checkpoints"
        # save-interval 10 < the ~19-cycle TMS disk-resume failure horizon
        # (DiskBackend::open_slot_file_ abort, see progress notes): every crash
        # has a checkpoint behind it and each restart resets the cycle count,
        # so long runs chain through the bug until it is fixed in TMS.
        # Params-only checkpoints: saving optimizer state stages ~80GB/rank of
        # host anon memory and the spike kernel-OOMed c001 twice at the save
        # (runs 100g/100h). Losing Adam moments across a crash-resume is fine
        # for this bring-up (constant lr 1e-6; moments rebuild in a few steps).
        ckpt_args += (
            f"--load {load_save_path} --save {load_save_path} --save-interval 10 "
            "--no-save-optim --no-save-rng --no-load-optim --no-load-rng "
        )

    rollout_args = (
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {args.num_rollout} "
        # 4 prompts x 8 samples = 32 sequences per rollout (down from 32x8=256):
        # faster iterations while bringing the pipeline up.
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 8 "
        "--rollout-temperature 0.8 "
        "--num-steps-per-rollout 1 "
        "--balance-data "
        f"--prompt-data {_DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        '--apply-chat-template-kwargs \'{"thinking_mode":"thinking"}\' '
    )

    # 48 layers / PP8 = 6 per stage, no first/last overrides needed. CP must stay 1:
    # the GDN wrapper only implements context parallelism on the fla backend, and we
    # train on flashqla (fla's chunk_fwd_o is nondeterministic on Blackwell).
    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 8 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 4 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--micro-batch-size 1 "
        # dsv4's proven value on this hardware. 8192 killed a train-step worker
        # (SIGSEGV/OOM-shaped): the torch QSA path materializes [T, S] attention
        # scores (~3.2 GB/layer at 8k), which is exactly the thing the planned
        # triton QSA kernel removes. Revisit after that kernel lands.
        # 2048 was a debugging-era safety margin (torch QSA then materialized
        # [T,S] scores). With the triton QSA kernel and full recompute the
        # activation footprint is small; 8192 packs ~4x more tokens per
        # microbatch, cutting the ~180-microbatch train step to ~45.
        "--max-tokens-per-gpu 8192 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        # GPU Adam on purpose: the CPU-offload optimizer (--optimizer-cpu-offload
        # --use-precision-aware-optimizer --overlap-cpu-optimizer-d2h-h2d) made the
        # step pathologically slow here (long 95%-CPU/0%-GPU phases on Grace), and
        # memory is comfortable without it (train peak ~81GB of 276GB, optimizer
        # states ~+54GB/rank).
    )

    # One sglang engine spans two nodes (tp8 over 4-GPU nodes); GB300 prefers tp8.
    sglang_args = (
        "--rollout-num-gpus-per-engine 8 "
        "--sglang-tp-size 8 "
        "--sglang-dp-size 1 "
        "--sglang-ep-size 8 "
        "--sglang-linear-attn-prefill-backend flashinfer "
        # flashinfer_trtllm (the GB300 default) shuffles expert weights into a
        # blocked 4D layout after loading, which the per-expert weight-update path
        # cannot write into (w13_weight becomes [E, 40, 1280, 64]). The triton
        # runner keeps the standard 3D layout that updates load into directly.
        "--sglang-moe-runner-backend triton "
        "--sglang-chunked-prefill-size 8192 "
        # Hybrid-model radix caching of unfinished reqs (mamba_component
        # prepare_for_caching_req) hit a device-side assert in attempt 28's
        # rollout. Prefix reuse across RL rollouts is ~5% hit rate — not worth
        # that bug class.
        "--sglang-disable-radix-cache "
        "--router-health-success-threshold 1 "
        "--router-health-check-interval-secs 15 "
        "--router-health-failure-threshold 40 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        "--accumulate-allreduce-grads-in-fp32 "
        f"--update-weight-buffer-size {1 * 1024 ** 3} "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--train-memory-margin-bytes 3221225472 "
        # Disk backup is REQUIRED, not optional: with the CPU target, sleep()'s
        # host backup (~80GB/rank x4) on top of c001's ~700GB baseline broke the
        # 944GB physical ceiling and the KERNEL OOM killer SIGKILLed a trainer
        # (run 34, silent death right after sleep; ray's 0.98 monitor was too
        # slow for the seconds-scale spike). The disk path's earlier "slowness"
        # (runs 32/33) was confounded by post-restart cold JIT caches.
        "--offload-train-target disk "
        "--offload-train-disk-dir /tmp/zz_train_offload "
        "--sglang-mem-fraction-static 0.7 "
        "--colocate "
        # hf model_type is qwen4_exp and the bridge registers that alias; the
        # default (hf config class name) would resolve to the aliased
        # Qwen3_5MoeConfig and miss.
        "--model-name qwen4_exp "
        "--qkv-format thd "
        "--linear-attention-backend flashqla "
        # Installs the PLE n-gram side-channel hooks on the stage hosting layer 1;
        # everything else delegates to miles' default provider.
        "--custom-model-provider-path "
        "miles_plugins.models.qwen3_8_next.model_provider.get_qwen3_8_next_model_provider "
        "--rollout-health-check-interval 300 "
        # The first train step compiles triton kernels for many shapes and the
        # pipeline stages drift apart; the default 10-minute NCCL timeout turned
        # that into watchdog SIGABRTs that looked like worker crashes.
        "--distributed-timeout-minutes 60 "
        "--rollout-health-check-timeout 300 "
    )
    if args.check_weight_update:
        # visual.* and the 102 GB frozen n-gram table are never part of an update
        # payload (text-only RL; the table is frozen by design), so the checker
        # must neither reset nor compare them -- reset would garble the real
        # values before the static-state stash snapshots them.
        misc_args += (
            "--check-weight-update-equal "
            # ple_embedding. covers the frozen int64 hash metadata
            # (layer_multipliers, ngram_heads_vocab_sizes/offsets) and the 102 GB
            # table; the PLE's trainable tensors (key_proj etc.) live under ple.*
            # outside ple_embedding and stay checked.
            "--check-weight-update-skip-list visual. ple_embedding. "
        )
    if args.enable_r3:
        misc_args += "--use-rollout-routing-replay "

    train_args = (
        f"{ckpt_args} {rollout_args} {optimizer_args} {grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} {sglang_args} {misc_args} {args.extra_args} "
    )

    extra_env_vars = {
        "SGLANG_SKIP_CHECKPOINT_LOAD_CHECK": "1",
        "SGLANG_HEALTH_CHECK_TIMEOUT": "120",
        "SGLANG_DISABLE_MULTIMEM_AG": "1",
        # rollout engines must import the ported sglang tree, and the training side
        # needs miles_plugins on the path; both are prepended here and merged with
        # execute_train's own entries.
        "PYTHONPATH": "/data/home/zzeng/repos/sglang-B/python",
        # Triton QSA kernel (validated against the torch reference at 1e-7/1e-3):
        # list-semantics selection, flash-style online softmax, no [T, S] score
        # materialization. The torch fallback's masked-SDPA backward is the prime
        # suspect for the native train-step worker deaths.
        "QSA_BACKEND": "triton",
        "TRITON_CACHE_DIR": "/tmp/zz_triton_cache",
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/zz_inductor_cache",
    }

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="qwen3.8-flash-next",
        extra_env_vars=extra_env_vars,
        megatron_path=_MEGATRON_PATH,
    )


if __name__ == "__main__":
    app()
