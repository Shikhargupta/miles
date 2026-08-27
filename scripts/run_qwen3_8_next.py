"""Qwen3.8-Flash-Next (Qwen4Exp) DAPO training.

Assumes an already-running ray cluster (MILES_SCRIPT_EXTERNAL_RAY=1) and a
converted torch_dist reference checkpoint.

Usage (inside the head-node container):
    python scripts/run_qwen3_8_next.py train --num-rollout 5
"""

import os
from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    num_nodes: int = 8
    num_gpus_per_node: int = 4
    run_id: str = "qwen38next-dapo"
    model_dir: str = "/root/models"
    ckpt_dir: str = "/root/ckpt"
    data_dir: str = "/root/datasets"
    save_dir: str = "/root/shared_data"
    megatron_path: str = "/root/Megatron-LM"
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
        f"--hf-checkpoint {args.model_dir}/Qwen3.8-Flash-Next "
        f"--ref-load {args.ckpt_dir}/qwen3.8-flash-next_torch_dist "
    )
    if not args.skip_saving:
        load_save_path = f"{args.save_dir}/{args.run_id}/checkpoints"
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
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 8 "
        "--rollout-temperature 0.8 "
        "--num-steps-per-rollout 1 "
        "--balance-data "
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        '--apply-chat-template-kwargs \'{"thinking_mode":"thinking"}\' '
    )

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
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 8 "
        "--sglang-tp-size 8 "
        "--sglang-dp-size 1 "
        "--sglang-ep-size 8 "
        "--sglang-linear-attn-prefill-backend flashinfer "
        "--sglang-moe-runner-backend triton "
        "--sglang-chunked-prefill-size 8192 "
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
        f"--update-weight-buffer-size {1 * 1024**3} "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--train-memory-margin-bytes 3221225472 "
        "--offload-train-target disk "
        "--offload-train-disk-dir /tmp/train_offload "
        "--sglang-mem-fraction-static 0.7 "
        "--colocate "
        "--model-name qwen4_exp "
        "--qkv-format thd "
        "--linear-attention-backend flashqla "
        "--custom-model-provider-path "
        "miles_plugins.models.qwen3_8_next.model_provider.get_qwen3_8_next_model_provider "
        "--rollout-health-check-interval 300 "
        "--distributed-timeout-minutes 60 "
        "--rollout-health-check-timeout 300 "
    )
    if args.check_weight_update:
        misc_args += "--check-weight-update-equal " "--check-weight-update-skip-list visual. ple_embedding. "
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
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        "QSA_BACKEND": "triton",
        "PYTHONFAULTHANDLER": "1",
        "TRITON_CACHE_DIR": "/tmp/triton_cache",
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/inductor_cache",
    }

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="qwen3.8-flash-next",
        extra_env_vars=extra_env_vars,
        megatron_path=args.megatron_path,
    )


if __name__ == "__main__":
    app()
