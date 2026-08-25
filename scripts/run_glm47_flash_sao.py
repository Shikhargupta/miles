from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

# GLM-4.7-Flash with SAO (Single-rollout Asynchronous Optimization, arXiv:2607.07508).
#
# Differences from scripts/run_glm47_flash.py, which is synchronous GRPO:
#   - actor-critic PPO with GAE instead of a group baseline
#   - ONE rollout per prompt (--n-samples-per-prompt 1); the critic supplies the baseline
#   - async driver (train_async.py)
#   - DIS: the ratio is measured against the rollout policy (--use-rollout-logprobs) and
#     out-of-trust-region tokens are gated to zero rather than clamped (--eps-clip-mode mask)
#   - TTUR: K critic updates per policy update (--critic-updates-per-policy-update)
#
# python scripts/run_glm47_flash_sao.py --run-id <YYMMDD-hash>


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = U.create_run_id()
    model_org: str = "zai-org"
    model_name: str = "GLM-4.7-Flash"
    megatron_model_type: str = "glm4.7-flash"
    num_gpus_per_node: int = 8
    rollout_num_gpus_per_engine: int = 1  # 20 attention heads, so rollout TP must divide 20
    enable_eval: bool = False
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    # traces must live on node-local /scratch: /personal is a shared quota and blew up a run
    traces_dir: str = ""


def prepare(args: ScriptArgs):
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(
        f"hf download {args.model_org}/{args.model_name} --local-dir {args.model_dir}/{args.model_name}"
    )
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=f"{args.model_dir}/{args.model_name}",
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs):
    ref_load_path = f"{args.model_dir}/{args.model_name}_torch_dist"
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"
    traces_dir = args.traces_dir or f"/scratch/{args.run_id}/traces"

    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name} "
        f"--ref-load {ref_load_path} "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        # --critic-load/--critic-lr fall back to --load/--lr; --critic-save derives as <save>_critic
        f"--save-interval {2 if args.mode == 'debug_minimal' else 20} "
        f"--save-retain-interval {2 if args.mode == 'debug_minimal' else 20} "
    )

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        f"--num-rollout {4 if args.mode == 'debug_minimal' else 300} "
        # SAO: one rollout per prompt. The batch is made of distinct prompts, not a group.
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 1 "
        f"--rollout-max-response-len {100 if args.mode == 'debug_minimal' else 8192} "
        "--rollout-temperature 1 "
        "--global-batch-size 32 "
        "--max-seq-len 65536 "
    )

    perf_args = (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        # EP=8 shards the 64 experts across all 8 GPUs; smaller EP quadruples per-GPU expert
        # weight and has OOM'd this model before.
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        # halved vs the GRPO launcher: PPO carries a second full model, and
        # --observe-training-entropy roughly doubles logits memory for the step
        "--max-tokens-per-gpu 16384 "
    )

    sao_args = (
        # actor-critic; this is what builds the value model and switches advantages to GAE
        "--advantage-estimator ppo "
        "--critic-lr 5e-6 "
        "--critic-lr-warmup-iters 10 "
        "--num-critic-only-steps 1 "
        "--critic-updates-per-policy-update 2 "
        "--normalize-advantages "
        # DIS: ratio against the rollout policy, gated on both sides instead of clamped.
        # Bounds are the paper's math/TIR setting (eps_low 0.3, eps_high 5.0).
        "--use-rollout-logprobs "
        "--eps-clip-mode mask "
        "--eps-clip 0.3 "
        "--eps-clip-high 5.0 "
        # reward-level KL is rejected under ppo: the critic never sees ref log probs
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
        # lowered from 0.7: the critic is a second resident model on the same GPUs
        "--sglang-mem-fraction-static 0.5 "
        "--sglang-speculative-algorithm EAGLE "
        "--sglang-speculative-num-steps 2 "
        "--sglang-speculative-eagle-topk 1 "
        "--sglang-speculative-num-draft-tokens 3 "
        "--use-rollout-routing-replay "
    )

    # standing rules: dashboard + both entropy flags travel together, traces on /scratch
    observability_args = (
        f"--dump-details {traces_dir} "
        "--use-miles-dashboard "
        "--observe-training-entropy "
        "--use-rollout-entropy "
        "--use-prometheus "
        f"--prometheus-run-name {args.run_id} "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{sao_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{observability_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        train_script="train_async.py",
        megatron_path=args.megatron_path,
        extra_env_vars={"PYTHONPATH": args.megatron_path},
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
