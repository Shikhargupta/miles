"""Qwen3 dense GRPO LoRA training through the native (raw-mode) LoRA path.

This is the reference / validation recipe for ``miles.backends.megatron_utils.lora_native``:
LoRA is applied directly to the mcore model built by miles' own model provider
(``--megatron-to-hf-mode raw``) instead of going through Megatron-Bridge. Adapters
are exported under HF/PEFT names and shipped to SGLang with the same adapter sync
the bridge path uses, so a run here exercises the whole loop: frozen base +
adapter grads, TP/EP grad summation, adapter-only weight sync, and LoRA serving.

Qwen3-8B / Qwen3-4B / Qwen3-0.6B are dense GQA models with a SwiGLU MLP, which is
exactly the layout the generic implementation covers. TP2 with sequence parallelism
is the interesting configuration: it exercises both the column-parallel (A
replicated, B row-sharded) and row-parallel (A col-sharded, B replicated)
grad-summation paths. Qwen3-0.6B on 2 GPUs is the cheapest end-to-end check.

Note on Qwen3.5: those checkpoints set ``attention_output_gate`` (linear_qkv emits a
4th gate slice) and the 35B-A3B is a GDN hybrid, so neither fits the generic fused-qkv
layout — native LoRA rejects them with a pointer to ``--lora-provider-path``. Use
``scripts/run_qwen3_5_35b_a3b_lora.py`` (bridge mode) for those.

Usage:
  python scripts/run_qwen3_lora_native.py prepare    --model-name Qwen3-8B
  python scripts/run_qwen3_lora_native.py full-train --model-name Qwen3-8B --task gsm8k
  python scripts/run_qwen3_lora_native.py train      --model-name Qwen3-4B --task dapo-math
"""

from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_MEGATRON_MODEL_TYPE = {
    "Qwen3-8B": "qwen3-8B",
    "Qwen3-4B": "qwen3-4B",
    "Qwen3-0.6B": "qwen3-0.6B",  # cheapest config that still exercises the whole loop
}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    model_name: Literal["Qwen3-8B", "Qwen3-4B", "Qwen3-0.6B"] = "Qwen3-8B"
    task: Literal["gsm8k", "dapo-math"] = "gsm8k"

    hf_checkpoint: str | None = None
    torch_dist: str | None = None
    model_dir: str = "/root/models"
    save_dir: str = "/personal/checkpoints"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"

    # performance
    num_gpus_per_node: int = 8
    tensor_model_parallel_size: int = 2

    # LoRA. target modules are HF leaf names; the native path maps them onto the
    # fused megatron modules (q/k/v -> linear_qkv, gate/up -> linear_fc1, ...).
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: str = "all-linear"
    lora_adapter_path: str | None = None
    # Train adapters but keep rollout on the frozen base policy (no LoRA serving).
    lora_train_only: bool = False
    # Verify the megatron->sglang adapter sync with a per-tensor sha256 manifest.
    check_lora_weight_equal: bool = False

    # rollout
    num_rollout: int = 10
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    rollout_max_response_len: int = 0  # 0 => per-task default (gsm8k 512, dapo-math 4096)
    global_batch_size: int = 64

    # rollout engine
    rollout_num_gpus_per_engine: int = 2
    sglang_mem_fraction_static: float = 0.6
    sglang_lora_backend: str = "triton"

    enable_wandb: bool = True
    extra_args: str = ""

    def __post_init__(self):
        if self.hf_checkpoint is None:
            self.hf_checkpoint = f"{self.model_dir}/{self.model_name}"
        if self.torch_dist is None:
            self.torch_dist = f"{self.model_dir}/{self.model_name}_torch_dist"
        if self.rollout_max_response_len == 0:
            self.rollout_max_response_len = 4096 if self.task == "dapo-math" else 512

    @property
    def megatron_model_type(self) -> str:
        return _MEGATRON_MODEL_TYPE[self.model_name]


def _get_parallel_config(args: ScriptArgs) -> str:
    """TP with sequence parallelism, DP over the remaining GPUs.

    TP must stay <= num_query_groups (8 for these checkpoints): below that mcore
    splits a query group across ranks and the local qkv rows stop being a clean
    per-group slice, which native LoRA rejects.
    """
    assert args.tensor_model_parallel_size <= 8, "Qwen3 dense has num_query_groups=8; native LoRA needs TP <= 8"
    perf = (
        f"--tensor-model-parallel-size {args.tensor_model_parallel_size} "
        "--pipeline-model-parallel-size 1 --context-parallel-size 1 "
        "--micro-batch-size 1 --max-tokens-per-gpu 9216 "
    )
    if args.tensor_model_parallel_size > 1:
        perf += "--sequence-parallel "
    return perf


def _download_dataset(args: ScriptArgs):
    match args.task:
        case "gsm8k":
            U.hf_download_dataset("zhuzilin/gsm8k", data_dir=args.data_dir)
        case "dapo-math":
            U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)


def _prepare_download(args: ScriptArgs):
    U.exec_command(f"mkdir -p {args.data_dir} {args.model_dir}")
    U.exec_command(f"hf download Qwen/{args.model_name} --local-dir {args.hf_checkpoint}")
    _download_dataset(args)
    # Native LoRA loads a real megatron dist-checkpoint for the frozen base.
    U.exec_command(
        f"cd {U.repo_base_dir} && PYTHONPATH={args.megatron_path}:{U.repo_base_dir} "
        f"torchrun --nproc-per-node 1 tools/convert_hf_to_torch_dist.py "
        f"--hf-checkpoint {args.hf_checkpoint} --save {args.torch_dist} "
        f"--megatron-to-hf-mode raw --bf16"
    )


def _train(args: ScriptArgs):
    print(
        f"[run] Qwen3 native LoRA: model={args.model_name} "
        f"(megatron_model_type={args.megatron_model_type}), {args.num_gpus_per_node} GPUs, "
        f"TP{args.tensor_model_parallel_size}, rollout tp={args.rollout_num_gpus_per_engine}"
    )
    load_save_path = f"{args.save_dir}/{args.run_id}"

    # raw mode: adapters are attached inside the model provider, not by the bridge.
    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} --load {args.torch_dist} "
        "--megatron-to-hf-mode raw --no-load-optim --no-load-rng --finetune "
    )

    lora_args = (
        f"--lora-rank {args.lora_rank} --lora-alpha {args.lora_alpha} "
        f'--lora-dropout {args.lora_dropout} --target-modules "{args.target_modules}" '
        "--no-gradient-accumulation-fusion "
    )
    if args.lora_adapter_path is not None:
        lora_args += f"--lora-adapter-path {args.lora_adapter_path} "
    if args.lora_train_only:
        lora_args += "--lora-train-only "
    if args.check_lora_weight_equal:
        lora_args += "--check-lora-weight-equal "

    rollout_args = (
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--rollout-temperature 1.0 "
        f"--global-batch-size {args.global_batch_size} "
    )
    match args.task:
        case "gsm8k":  # zhuzilin/gsm8k ships {messages, label} parquet
            rollout_args += f"--prompt-data {args.data_dir}/gsm8k/train.parquet --input-key messages "
        case "dapo-math":  # zhuzilin/dapo-math-17k ships {prompt, label} jsonl (prompt = chat messages)
            rollout_args += f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl --input-key prompt "

    grpo_args = "--advantage-estimator grpo --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "

    optimizer_args = (
        "--optimizer adam --lr 1e-5 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
    )

    perf_args = _get_parallel_config(args)

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        "--sglang-dtype bfloat16 --sglang-decode-log-interval 1000 "
        f"--sglang-max-lora-rank {args.lora_rank} "
        f"--sglang-lora-backend {args.sglang_lora_backend} "
    )

    save_args = f"--save-interval 5 --save {load_save_path} "

    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--update-weight-buffer-size 536870912 "
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {args.num_gpus_per_node} --colocate "
    )

    wandb_args = U.get_default_wandb_args(__file__, run_id=args.run_id) if args.enable_wandb else ""

    train_args = (
        f"{ckpt_args} {lora_args} {rollout_args} {optimizer_args} {grpo_args} "
        f"{wandb_args} {perf_args} {sglang_args} {save_args} {misc_args} {args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs):
    """Download the checkpoint + dataset and convert the base to a torch-dist checkpoint."""
    _prepare_download(args)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """Run GRPO LoRA training through the native path (assumes prepare already ran)."""
    _train(args)


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs):
    """Prepare, then run GRPO LoRA training through the native path."""
    _prepare_download(args)
    _train(args)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
