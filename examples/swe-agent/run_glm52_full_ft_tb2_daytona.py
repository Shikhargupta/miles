"""GLM-5.2 744B-A40B **full-parameter** agentic launcher: Terminal-Bench-2 on Daytona.

Grafts two paths that did not previously meet:
  * the agentic Harbor path (session server + TITO + terminus-2) from
    ``run_glm52_lora_tb2_daytona.py``, and
  * the full-parameter GB300 parallelism from ``scripts/run_glm5_2_744b_a40b.py``.

The LoRA sibling trains ~0.1% of the weights, so it can afford ``PP=1`` with ``EP``
spanning the world. Full-parameter cannot: at 744B the mixed-precision optimizer state
is ``744e9 * 16 B ~= 11.9 TB`` (bf16 weights + bf16 grads + fp32 master + Adam m/v),
which does not fit under that layout. This script therefore adopts the pipeline-parallel
GB300 config from the full-FT script instead:

    TP=8  PP=4  DP=2  EP=16  ETP=1  CP=1   ==  64 GPUs  ==  16 nodes x 4

``EP=16`` (not the world) is the value that fits the fp32 optimizer states on 277 GiB
GB300 parts. ``--decoder-first-pipeline-num-layers 18`` / ``--decoder-last-pipeline-num-layers
20`` place every pipeline-stage boundary on a DSA *computing* layer (stage starts land on
global layers 1, 19, 39, 59); GLM-5.2 shares the DSA top-k across layers, so a stage that
began on a skip layer would read a stale anchor.

Attention backend: ``tilelang`` (thd), never ``megatron`` (bshd). ``megatron`` is a dense
O(S**2) reference that materialises the full fp32 score matrix and caps out near S=4096 --
unusable at the 65536 sequence budget agentic TB2 needs. thd also carries
``packed_seq_params``, which is what lets Megatron's checkpoint ``custom_forward``
closure-capture the anchor top-k and thereby permits activation recompute. The tilelang
backward NaN seen in earlier LoRA runs was two kernel defects in
``miles_plugins/models/glm5/ops/``; both are fixed.

Differences from the LoRA sibling, beyond parallelism:
  * no ``--lora-*`` / ``--target-modules`` / ``--sglang-max-lora-rank`` /
    ``--sglang-lora-backend``; ``megatron_model_type`` drops the ``_lora`` suffix.
  * **every** parameter changes each step, so the colocated fp8 rollout needs a full
    744B weight re-quantise + sync per step rather than a small adapter push. This is
    the dominant new cost versus LoRA and the first thing to profile.
  * ``--optimizer-cpu-offload`` is mandatory, not optional: the 8.9 TB of fp32 master +
    Adam moments lives in host RAM (16 nodes x 920 GB = 14.7 TB).

Usage (16 nodes x 4 GB300, Ray already up across the allocation):
  MILES_SCRIPT_EXTERNAL_RAY=1 python run_glm52_full_ft_tb2_daytona.py \\
      --num-nodes 16 --num-gpus-per-node 4 --prompt-data /data/.../tb2_train.jsonl
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
    # No _lora suffix: this is the dense full-parameter model type.
    megatron_model_type: str = "glm5.2-744B-A40B"
    num_gpus_per_node: int = 4
    megatron_path: str = "/data/home/sdong/Megatron-LM"

    # Paths (this cluster: /data is the shared VAST mount, visible on every node)
    hf_checkpoint: str = "/data/models/GLM-5.2"
    # Rollout-side fp8 checkpoint; the trainer stays bf16.
    fp8_rollout_checkpoint: str = "/data/models/GLM-5.2_fp8"
    # Reference checkpoint for KL. Only needed when a KL term is actually on; the GRPO
    # recipe below runs kl-coef 0, so this stays empty by default and no second 744B
    # copy has to be converted or held.
    ref_load: str = ""
    # Must be shared across nodes: every rank writes its dist-checkpoint shard here and
    # the generated sglang config is read by engine actors on every node.
    save_dir: str = "/data/home/sdong/runs/260805-f86465b2"
    save_traces_dir: str = ""
    prompt_data: str = "/data/home/sdong/datasets/tb2/tb2_train_glm52.jsonl"

    # Sequence budget: --max-seq-len caps the whole session (prompt + every completion +
    # every env response); --rollout-max-response-len caps a single turn.
    max_seq_len: int = 65536
    rollout_max_response_len: int = 8192
    # Serving window, deliberately independent of max_seq_len: the agent needs the full
    # context or its very first request is rejected.
    sglang_context_length: int = 65536

    # Training settings
    num_rollout: int = 200
    rollout_batch_size: int = 4
    n_samples_per_prompt: int = 8
    global_batch_size: int = 32
    save_interval: int = 10
    lr: str = "1e-6"  # full-FT: two orders below the LoRA sibling's 3e-5

    # GLM-5.2 specifics. megatron (bshd) is dense O(S**2) and caps near S=4096, so the
    # 65536 agentic budget requires tilelang. See the module docstring.
    dsa_attention_backend: Literal["megatron", "tilelang"] = "tilelang"
    use_r3: bool = True

    # Dynamic batching is available on thd (unlike the bshd LoRA path, which is pinned to
    # --micro-batch-size 1), and is the main activation-memory lever at long sequence.
    max_tokens_per_gpu: int = 8192

    # Rollout engine
    fp8_rollout_gpus_per_engine: int = 16
    sglang_mem_fraction_static: float = 0.85
    # Node-local, must NOT be tmpfs. / has ~99 GB free on these nodes.
    offload_train_disk_dir: str = "/var/tmp/miles_train_offload_260805"

    # Agent settings
    agent_server_url: str = os.environ.get(
        "AGENT_SERVER_URL", os.environ.get("SWE_AGENT_URL", "http://127.0.0.1:11001")
    )
    agent_model_name: str = os.environ.get("AGENT_MODEL_NAME", "model")
    harbor_tasks_dir: str = os.environ.get(
        "HARBOR_TASKS_DIR", "/data/home/sdong/datasets/tb2/terminal-bench"
    )
    # sgl-router binds with a Rust SocketAddr parse, so this MUST be a numeric IP.
    router_external_host: str = os.environ.get("MILES_ROUTER_EXTERNAL_HOST", socket.gethostname())
    miles_host_ip: str = os.environ.get("MILES_HOST_IP", "")
    # Daytona is SaaS; the sandboxes are remote, the compute nodes only need egress.
    harbor_env_type: str = os.environ.get("HARBOR_ENV_TYPE", "daytona")
    daytona_api_key: str = os.environ.get("DAYTONA_API_KEY", "")
    harbor_daytona_disk_gb: str = os.environ.get("HARBOR_DAYTONA_DISK_GB", "10")

    # W&B settings
    wandb_key: str = os.environ.get("WANDB_KEY", os.environ.get("WANDB_API_KEY", ""))
    wandb_project: str = os.environ.get("WANDB_PROJECT", "glm-gb300")
    wandb_team: str = os.environ.get("WANDB_TEAM", "eigent_radixark_training")
    wandb_run_name: str = "260805-f86465b2-glm52-full-ft-tb2-daytona"

    # Prometheus: off by default here. Grafana scraping on this cluster relies on k8s pod
    # annotations that Slurm does not provide, and --use-prometheus is what triggers the
    # prometheus_client DuplicateTimeseries crash that stops an sglang engine from
    # recovering after a restart.
    use_prometheus: bool = False
    prometheus_port: int = 9091


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


def _parallel_args(args: ScriptArgs) -> str:
    """Full-parameter GB300 layout: TP=8 PP=4 DP=2 EP=16 ETP=1 CP=1 over 64 GPUs.

    Lifted from scripts/run_glm5_2_744b_a40b.py's ``num_gpus_per_node == 4`` branch.
    TP=8 spans two nodes, which is free here because the rack is a single NVLink
    (MNNVL/IMEX) domain rather than IB-connected.
    """
    world_size = args.num_nodes * args.num_gpus_per_node
    if not (args.num_nodes >= 16 and args.num_gpus_per_node == 4):
        raise NotImplementedError(
            "Full-parameter GLM-5.2 has only one validated layout on GB300: "
            f">=16 nodes x 4 GPUs. Got {args.num_nodes} x {args.num_gpus_per_node} "
            f"(world={world_size})."
        )

    # tilelang => thd, which carries packed_seq_params and therefore permits both
    # activation recompute and dynamic batching.
    if args.dsa_attention_backend != "tilelang":
        raise NotImplementedError(
            "megatron/bshd materialises the full fp32 [b,np,S,S] score matrix and caps "
            f"near S=4096; this run needs S={args.max_seq_len}. Use tilelang."
        )

    return (
        "--tensor-model-parallel-size 8 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 4 "
        "--decoder-first-pipeline-num-layers 18 "
        "--decoder-last-pipeline-num-layers 20 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 16 "
        "--expert-tensor-parallel-size 1 "
        "--qkv-format thd "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.max_tokens_per_gpu} "
        "--data-pad-size-multiplier 1024 "
        "--log-probs-chunk-size 16384 "
        # Mandatory at full-parameter scale: 8.9 TB of fp32 master + Adam moments
        # lives in host RAM (16 x 920 GB = 14.7 TB), not HBM.
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )


def _sglang_args(args: ScriptArgs) -> str:
    world_size = args.num_nodes * args.num_gpus_per_node
    engine = min(args.fp8_rollout_gpus_per_engine, world_size)
    max_bs = 64
    return (
        f"--rollout-num-gpus-per-engine {engine} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        # dp-attention stays off: its per-iteration collectives desync against colocate
        # weight syncs and deadlock the engines.
        f"--sglang-ep-size {engine} "
        "--sglang-attention-backend nsa "
        "--sglang-nsa-decode-backend flashmla_kv "
        "--sglang-nsa-prefill-backend flashmla_sparse "
        "--sglang-page-size 64 "
        "--sglang-kv-cache-dtype fp8_e4m3 "
        f"--sglang-context-length {args.sglang_context_length} "
        f"--sglang-cuda-graph-max-bs {max_bs} --sglang-max-running-requests {max_bs} "
        f"--sglang-chunked-prefill-size {min(8192, 2048 * engine)} "
        "--sglang-watchdog-timeout 3600 "
        "--sglang-moe-runner-backend triton --sglang-disable-shared-experts-fusion "
        "--sglang-tool-call-parser glm47 "
        "--sglang-reasoning-parser glm45 "
        # Port only; this does NOT select the miles router. The sgl-router is used.
        "--sglang-router-port 31001 "
    )


def _write_sglang_fp8_config(args: ScriptArgs) -> str:
    """Serve the fp8 checkpoint while the trainer stays bf16.

    Unlike the LoRA sibling, update_weights here pushes the *whole* re-quantised 744B
    model every step, because full-parameter training changes every tensor.
    """
    path = f"{args.save_dir.rstrip('/')}/sglang_fp8_rollout.yaml"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(
            "sglang:\n"
            "  - name: default\n"
            f"    model_path: {args.fp8_rollout_checkpoint}\n"
            "    update_weights: true\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            f"        num_gpus: {args.num_nodes * args.num_gpus_per_node}\n"
        )
    return path


def execute(args: ScriptArgs):
    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        "--megatron-to-hf-mode bridge "
        f"--dsa-attention-backend {args.dsa_attention_backend} "
        f"--save {args.save_dir} "
        f"--save-interval {args.save_interval} "
    )
    if args.ref_load:
        ckpt_args += f"--ref-load {args.ref_load} "

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

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        f"--lr {args.lr} "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    r3_args = "--use-rollout-routing-replay " if args.use_r3 else ""

    agent_args = (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path swe_agent_function.run "
        "--custom-rm-path generate.reward_func "
        "--rollout-function-path generate.RolloutFn "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        "--tito-model glm47 "
        "--use-session-server "
        "--session-server-port 30001 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--calculate-per-token-loss "
        "--colocate "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--offload-train-target disk "
        f"--offload-train-disk-dir {args.offload_train_disk_dir} "
        "--offload-train-disk-chunk-mb 256 "
    )

    traces_dir = args.save_traces_dir or f"{args.save_dir.rstrip('/')}/traces"
    if traces_dir != "disabled":
        misc_args += f"--dump-details {traces_dir} --use-miles-dashboard "

    # Standing requirement: entropy must be instrumented on every run. A falling entropy
    # is the earliest warning of policy collapse on a long agentic run, and full-parameter
    # training can collapse far faster than LoRA.
    misc_args += "--observe-training-entropy --use-rollout-entropy "

    # bf16 has no grad scaler, so a non-finite grad norm would otherwise reach the step
    # and clipping would write NaN into every tensor. This routes through miles' own
    # guard, which skips the step and leaves weights intact.
    misc_args += "--no-check-for-nan-in-loss-and-grad "

    debug_args = "--debug-rollout-only " if args.mode == "debug_rollout_only" else ""

    wandb_args = ""
    if args.wandb_key:
        wandb_args = (
            "--use-wandb "
            f"--wandb-project {args.wandb_project} "
            f"--wandb-group {args.wandb_run_name} "
            f"--wandb-key {args.wandb_key} "
        )
        if args.wandb_team:
            wandb_args += f"--wandb-team {args.wandb_team} "

    prometheus_args = ""
    if args.use_prometheus:
        prometheus_args = (
            "--use-prometheus "
            f"--prometheus-port {args.prometheus_port} "
            f"--prometheus-run-name {args.wandb_run_name} "
        )

    sglang_args = _sglang_args(args) + f"--sglang-config {_write_sglang_fp8_config(args)} "

    train_args = (
        f"{ckpt_args}"
        f"{rollout_args}"
        f"{optimizer_args}"
        f"{grpo_args}"
        f"{r3_args}"
        f"{wandb_args}"
        f"{prometheus_args}"
        f"{_parallel_args(args)}"
        f"{sglang_args}"
        f"{agent_args}"
        f"{misc_args}"
        f"{debug_args}"
    )

    miles_root = U.repo_base_dir

    extra_env_vars = {
        "PYTHONPATH": f"{args.megatron_path}:{SCRIPT_DIR}:{miles_root}",
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        "AGENT_SERVER_URL": args.agent_server_url,
        "AGENT_MODEL_NAME": args.agent_model_name,
        "MILES_ROUTER_EXTERNAL_HOST": args.router_external_host,
        "HARBOR_TASKS_DIR": args.harbor_tasks_dir,
        "HARBOR_ENV_TYPE": args.harbor_env_type,
        "HARBOR_DAYTONA_DISK_GB": args.harbor_daytona_disk_gb,
        # GLM-5 DSA indexer uses interleaved RoPE; a mismatch garbles long sequences.
        "INDEXER_ROPE_NEOX_STYLE": "0",
        "SGLANG_NSA_FORCE_MLA": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.8,max_split_size_mb:512",
        # Re-derived for GB300's 277 GiB (the LoRA sibling's 0.72 was sized against a
        # 139.8 GiB H200). Caps torch near 222 GiB, leaving ~55 GiB for the sglang
        # residual, the CUDA context and NCCL's out-of-pool allocations.
        "MILES_TRAIN_MEMORY_FRACTION": "0.80",
    }
    if args.miles_host_ip:
        extra_env_vars["MILES_HOST_IP"] = args.miles_host_ip
    if args.daytona_api_key:
        extra_env_vars["DAYTONA_API_KEY"] = args.daytona_api_key

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
    execute(args)


if __name__ == "__main__":
    typer.run(main)
