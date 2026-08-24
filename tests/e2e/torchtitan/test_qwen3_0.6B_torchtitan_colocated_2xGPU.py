import os

from tests.ci.ci_register import register_cuda_ci

import miles.utils.external_utils.command_utils as U

register_cuda_ci(
    est_time=1200,
    suite="stage-c-2-gpu-h200",
    labels=["torchtitan"],
)

MODEL_NAME = "Qwen3-0.6B"
NUM_GPUS = 2


def prepare():
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def execute():
    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {3000 if U.get_env_enable_infinite_run() else 8} "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 512 "
        "--rollout-temperature 1 "
        "--global-batch-size 32 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-coef 0.00 "
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

    sglang_args = f"--rollout-num-gpus-per-engine {NUM_GPUS} --sglang-decode-log-interval 1000 "

    # torchtitan's only torch-2.11-compatible attention backend is sdpa, which
    # applies a plain causal mask: a microbatch must hold a single document, so
    # this run uses fixed micro-batching instead of dynamic packing.
    titan_args = (
        "--train-backend torchtitan "
        "--titan-model-name qwen3 "
        "--titan-model-flavor 0.6B "
        "--titan-attn-backend sdpa "
        "--titan-seq-len 2048 "
        "--micro-batch-size 1 "
        f"--update-weight-buffer-size {512 * 1024 * 1024} "
    )

    misc_args = f"--hf-checkpoint /root/models/{MODEL_NAME} --actor-num-nodes 1 --actor-num-gpus-per-node {NUM_GPUS} --colocate "

    ci_args = "--ci-test --ci-disable-kl-checker "

    train_args = (
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{sglang_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{titan_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=None,
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
