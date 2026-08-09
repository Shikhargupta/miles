"""SWE-Agent launcher (GLM-4.7-Flash): Miles <-> Harbor orchestration.

Supports any task type (SWE-bench, Terminal-Bench, custom) via Harbor.

Usage:
    python run.py
    python run.py --mode normal
    python run.py --base-dir /my/models --prompt-data /my/data.jsonl
"""

import os
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_rollout_only"] = "normal"
    run_id: str = U.create_run_id()
    megatron_model_type: str = "glm4.7-flash"
    num_gpus_per_node: int = 8
    megatron_path: str = "/root/Megatron-LM"

    # Paths
    skip_prepare: bool = False
    base_dir: str = "/root"
    model_name: str = "GLM-4.7-Flash"
    hf_checkpoint: str = "zai-org/GLM-4.7-Flash"
    ref_load: str = "/root/GLM-4.7-Flash_torch_dist"
    save_dir: str = "/root/GLM-4.7-Flash_agent_v2/"
    prompt_data: str = "/root/swe_train.jsonl"

    # Training settings
    max_seq_len: int = 65536
    rollout_max_response_len: int = 8192
    num_rollout: int = 3000
    rollout_batch_size: int = 4
    n_samples_per_prompt: int = 8
    global_batch_size: int = 32
    save_interval: int = 100
    save_traces_dir: str = ""
    use_miles_dashboard: bool = False
    observe_training_entropy: bool = False
    use_rollout_entropy: bool = False
    enable_mtp: bool = False

    # Agent settings
    agent_server_url: str = os.environ.get("AGENT_SERVER_URL", os.environ.get("SWE_AGENT_URL", "http://agent_env:11000"))
    agent_model_name: str = os.environ.get("AGENT_MODEL_NAME", "model")
    harbor_tasks_dir: str = os.environ.get("HARBOR_TASKS_DIR", "/root/harbor_tasks")
    router_external_host: str = os.environ.get("MILES_ROUTER_EXTERNAL_HOST", socket.gethostname())  # public IP
    session_external_base_url: str = os.environ.get("MILES_SESSION_EXTERNAL_BASE_URL", "")
    agent_model_api_key: str = os.environ.get("AGENT_MODEL_API_KEY", "")
    miles_host_ip: str = os.environ.get("MILES_HOST_IP", "")  # optional cluster/pod IP override

    # W&B settings
    wandb_key: str = os.environ.get("WANDB_KEY", os.environ.get("WANDB_API_KEY", ""))
    wandb_project: str = os.environ.get("WANDB_PROJECT", "my-wandb-project")
    wandb_team: str = os.environ.get("WANDB_TEAM", "")
    wandb_run_name: str = "glm47-flash-swe-tito"

    # Prometheus settings
    use_prometheus: bool = True
    prometheus_port: int = 9090
    prometheus_run_name: str = "glm47-flash-swe-tito"


def cleanup():
    """Kill old Ray jobs and stale processes to free GPU resources."""
    my_pid = os.getpid()
    ppid = os.getppid()
    print(f"Cleanup starting (pid={my_pid}, ppid={ppid})")
    targets = ["sglang", "train.py", "MegatronTrain"]
    exclude = f"grep -v '^{my_pid}$' | grep -v '^{ppid}$'"
    for t in targets:
        subprocess.run(
            f"pgrep -f '{t}' | {exclude} | xargs -r kill 2>/dev/null || true",
            shell=True,
        )
    time.sleep(5)
    print(f"Cleanup complete (pid={my_pid}) — old processes killed.")


def prepare(args: ScriptArgs):
    """Convert HF checkpoint to torch_dist format if not already done."""
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.base_dir,
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs):
    ckpt_args = f"--hf-checkpoint {args.hf_checkpoint} --ref-load {args.ref_load} --save {args.save_dir} --save-interval {args.save_interval} "

    rollout_args = (
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        "--metadata-key metadata "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        "--rollout-temperature 0.8 "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        f"--max-seq-len {args.max_seq_len} "
        f"--global-batch-size {args.global_batch_size} "
        "--balance-data "
    )

    perf_args = (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    grpo_args = "--advantage-estimator grpo --use-kl-loss --kl-loss-coef 0.01 --kl-loss-type low_var_kl --entropy-coef 0.0 --eps-clip 0.2 --eps-clip-high 0.28 "

    optimizer_args = "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "

    sglang_args = "--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.7 --sglang-tool-call-parser glm47 --sglang-reasoning-parser glm45 --sglang-router-port 31000 "
    if args.enable_mtp:
        sglang_args += "--sglang-speculative-algorithm EAGLE --sglang-speculative-num-steps 2 --sglang-speculative-eagle-topk 1 --sglang-speculative-num-draft-tokens 3 "

    agent_args = (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path swe_agent_function.run "
        "--custom-rm-path generate.reward_func "
        "--rollout-function-path generate.RolloutFn "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        "--tito-model glm47 "
        "--use-session-server "
        "--session-server-port 30000 "
    )

    misc_args = f"--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash --colocate --actor-num-nodes {args.num_nodes} --actor-num-gpus-per-node {args.num_gpus_per_node} --rollout-num-gpus {args.num_gpus_per_node} "

    debug_args = "--debug-rollout-only " if args.mode == "debug_rollout_only" else ""

    trace_args = ""
    if args.save_traces_dir:
        trace_args = f"--dump-details {args.save_traces_dir} "
    if args.use_miles_dashboard:
        if not args.save_traces_dir:
            raise ValueError("--use-miles-dashboard requires --save-traces-dir")
        trace_args += "--use-miles-dashboard "

    entropy_args = ""
    if args.observe_training_entropy:
        entropy_args += "--observe-training-entropy "
    if args.use_rollout_entropy:
        entropy_args += "--use-rollout-entropy "

    wandb_args = ""
    if args.wandb_key:
        wandb_args = f"--use-wandb --wandb-project {args.wandb_project} --wandb-group {args.wandb_run_name} --wandb-key {args.wandb_key} "
        if args.wandb_team:
            wandb_args += f"--wandb-team {args.wandb_team} "

    prometheus_args = ""
    if args.use_prometheus:
        prometheus_args = f"--use-prometheus --prometheus-port {args.prometheus_port} --prometheus-run-name {args.prometheus_run_name} "

    train_args = f"{ckpt_args}{rollout_args}{optimizer_args}{grpo_args}{wandb_args}{prometheus_args}{trace_args}{entropy_args}{perf_args}{sglang_args}{agent_args}{misc_args}{debug_args}"

    miles_root = U.repo_base_dir

    extra_env_vars = {
        "PYTHONPATH": f"{args.megatron_path}:{SCRIPT_DIR}:{miles_root}",
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        "AGENT_SERVER_URL": args.agent_server_url,
        "AGENT_MODEL_NAME": args.agent_model_name,
        "MILES_ROUTER_EXTERNAL_HOST": args.router_external_host,
        "HARBOR_TASKS_DIR": args.harbor_tasks_dir,
    }
    if args.session_external_base_url:
        extra_env_vars["MILES_SESSION_EXTERNAL_BASE_URL"] = args.session_external_base_url
    if args.agent_model_api_key:
        extra_env_vars["AGENT_MODEL_API_KEY"] = args.agent_model_api_key
    if args.miles_host_ip:
        extra_env_vars["MILES_HOST_IP"] = args.miles_host_ip

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        extra_env_vars=extra_env_vars,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    cleanup()
    if not args.skip_prepare:
        prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
