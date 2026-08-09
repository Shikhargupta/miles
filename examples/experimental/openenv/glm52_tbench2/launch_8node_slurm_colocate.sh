#!/bin/bash
# Site adapter for PR-2220's run_glm5_2_744b_a40b_daytona.py on rdx-gb300 (Slurm).
#
# Based on the PR's launch_16node_slurm.sh, with this cluster's specifics folded in:
#   * --container-name=ray so every srun step on a node reuses ONE container (their
#     idea, and better than the step-merging workaround I had used: Ray's CoreWorker
#     needs the raylet's unix socket under /tmp/ray, invisible across containers).
#   * --mem=0 to get all 920 GB; without it Slurm capped the allocation at 880 GB,
#     which is what several OOM kills were measured against.
#   * FABRIC_PREFIX IP selection -- the management subnet is not routable between
#     nodes, and `hostname -I` ordering is not stable.
#   * Node-local JIT caches. $HOME is on the VAST NFS mount, so Triton's default
#     $HOME/.triton is shared by all ranks and concurrent MoE kernel compilation dies
#     with "OSError: [Errno 116] Stale file handle". Exported BEFORE `ray start`
#     because Ray actors inherit the raylet's environment, not the driver's.
#   * PYTHONPATH carrying the openenv venv, the OpenEnv checkout (tbench2_env is a
#     flat package, and an editable .pth is not honoured via PYTHONPATH) and miles.
#
#SBATCH --job-name=glm52-tb2-colocate
#SBATCH --account=customer
#SBATCH --partition=batch_1
#SBATCH --nodes=8
# c001-c008 were wedged for ~8h; they released their Slurm allocation but leftover
# processes still hold their RAM (c001 had 33 GB free vs 804 GB on c009). With --mem=0
# any job landing there dies instantly with no log (jobs 1948, 1949).
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=/data/home/sdong/logs/%x-%j.log
set -uo pipefail

MILES_ROOT=/data/home/sdong/worktrees/pr2220/miles
MEGATRON=/data/home/sdong/worktrees/megatron-pr63/Megatron-LM
OPENENV=/data/home/sdong/OpenEnv
VENV_SP=/data/home/sdong/openenv-venv/lib/python3.12/site-packages
IMAGE=/data/home/sdong/images/miles-dev-arm64.sqsh
TB2_TASKS=/data/home/sdong/terminal-bench-2
PROMPT_DATA=/data/home/sdong/datasets/tbench2_train.jsonl
DAYTONA_ENV_FILE=/data/home/sdong/.secrets_260805.env
RUN_ID=260809-4caad5db

RECIPE=$MILES_ROOT/examples/experimental/openenv/glm52_tbench2/run_glm5_2_744b_a40b_daytona.py
C="--container-image=$IMAGE --container-mounts=/data:/data --container-name=ray"

PP="$VENV_SP:$OPENENV/envs:$MILES_ROOT:$MILES_ROOT/examples/experimental/openenv"
COMMON="export PYTHONNOUSERSITE=1 RAY_memory_monitor_refresh_ms=0 PYTHONPATH=$PP
        export TRITON_CACHE_DIR=/var/tmp/jit-\$SLURM_JOB_ID/triton
        export TORCHINDUCTOR_CACHE_DIR=/var/tmp/jit-\$SLURM_JOB_ID/inductor
        export CUDA_CACHE_PATH=/var/tmp/jit-\$SLURM_JOB_ID/nv
        ulimit -n 1048576 || true
        mkdir -p \$TRITON_CACHE_DIR \$TORCHINDUCTOR_CACHE_DIR \$CUDA_CACHE_PATH /var/tmp/opt_state"

# Compute fabric on this cluster is 10.4.90.x; the 10.0.1.x management subnet is
# NOT routable between nodes. The upstream default prefix "10." picks the wrong one
# and Ray workers then fail with "Failed to connect to GCS at 10.0.1.2:6379" (job 1926).
FABRIC_PREFIX="${FABRIC_PREFIX:-10.4.}"
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_ip=$(srun --nodes=1 --ntasks=1 -w "${nodes[0]}" hostname -I | tr ' ' '\n' | grep -E "^${FABRIC_PREFIX:-10.}" | head -1)
ngpu_total=$(( ${#nodes[@]} * 4 ))
echo "head=${nodes[0]} ($head_ip)  nodes=${#nodes[@]}  gpus=$ngpu_total  COLOCATE"

srun --overlap --nodes=1 --ntasks=1 --gpus-per-node=4 -w "${nodes[0]}" $C bash -c "
  $COMMON
  ray start --head --node-ip-address=$head_ip --port=6379 --num-gpus=4 --disable-usage-stats
  for t in \$(seq 1 90); do
    ray status 2>/dev/null | grep -q '/$ngpu_total\.0 GPU' && break
    echo \"waiting for ray nodes... (\$t/90)\"; sleep 10
  done
  source $DAYTONA_ENV_FILE
  export MILES_SCRIPT_EXTERNAL_RAY=1 MASTER_ADDR=$head_ip OPENENV_RUN_ID=$RUN_ID
  export OPENENV_TB2_TASKS_DIR=$TB2_TASKS OPENENV_LAUNCHER=sdong
  # NOTE: everything from line 62 to the closing quote is ONE double-quoted bash -c
  # argument. Never put a double quote or an unescaped backtick in these comments --
  # it closes the string early and the rest of the file is re-parsed by the outer
  # shell. That is what killed job 2032. Run bash -n on this file before every sbatch.
  #
  # Keep the reference episode budget. An earlier 1200s/20-turn cap was tried on the
  # theory that dead Daytona sandboxes were blocking the sync batch; that diagnosis was
  # wrong -- the sandbox errors were fallout from the /data NFS outage. What the short
  # cap actually did was guillotine slow-but-correct trajectories: job 2030 logged 421
  # episode-exceeded-1200s terminations forced to reward 0, and raw_reward sat at
  # 0.30-0.63. At 3600s/64 turns, job 2031 step 0 had ZERO timeouts and raw_reward
  # 0.607 on the same model and task set. Do not lower these.
  export OPENENV_MAX_ROLLOUT_TIME_SECONDS=3600 OPENENV_MAX_TURNS=40
  # 8, not 4: the 4 was collateral from the wrong Daytona-is-the-limiter diagnosis
  # and it is half what the recipe itself defaults to. Measured ramp-to-peak
  # concurrency at 4 was 126-146s per rollout; the stagger lands on the critical
  # path because each episode's 3600s clock starts when IT starts. Do not push far
  # past 8 without evidence: Daytona rate-limits creation (ThrottlerException) and
  # the leg then retries with backoff up to 30s, which lengthens the stagger.
  export OPENENV_DAYTONA_CREATE_CONCURRENCY=8
  export WANDB_PROJECT=glm-gb300 WANDB_TEAM=eigent_radixark_training
  cd $MILES_ROOT
  # PR 2220 fidelity: the recipe body is unmodified from 2220, so carrying NO sglang
  # overrides in --extra-args restores its engine config verbatim --
  #   sglang_world_size = 8  ->  --rollout-num-gpus-per-engine 8
  #                              --sglang-ep-size 8
  #                              --sglang-mem-fraction-static 0.85
  #                              --sglang-max-running-requests 512
  # ep-size and engine width are THE SAME VARIABLE in 2220 and must never be set
  # independently. EP=8 against a 16-GPU engine mis-maps the MoE expert shards: the
  # model emits garbage and NOTHING CRASHES to tell you -- rollout/entropy 4.62 vs
  # 0.19, truncation 0.98, reward 0.0, grad_norm exactly 0.0 (run 260808-ea68b06f).
  # Only the colocate-required offload flags stay in --extra-args.
  # Comments must live HERE, never between the python3 continuation lines below: a
  # bare # line there ends the command and turns --extra-args into its own command.
  python3 $RECIPE train \
    --num-nodes ${#nodes[@]} \
    --num-gpus-per-node 4 \
    --colocate \
    --run-id $RUN_ID \
    --model-dir /data/models \
    --model-local-dir /data/models \
    --data-dir /data/home/sdong/datasets \
    --prompt-data $PROMPT_DATA \
    --megatron-path $MEGATRON \
    --offload-train-disk-dir /var/tmp/opt_state \
    --no-enable-mtp \
    --bf16-grads \
    --eval-interval 1000 \
    --output-dir /data/home/sdong/runs \
    --save-interval 3 \
    --rollout-batch-size 8 \
    --n-samples-per-prompt 8 \
    --global-batch-size 64 \
    --sglang-config low-latency \
    --extra-args '--offload-train --offload-rollout --offload-train-target disk --wandb-team eigent_radixark_training --wandb-project glm-gb300' \
    $*
" &
HEAD_PID=$!

sleep 30
for ((i=1; i<${#nodes[@]}; i++)); do
  srun --overlap --nodes=1 --ntasks=1 --gpus-per-node=4 -w "${nodes[$i]}" $C bash -c "
    $COMMON
    until ray start --address=$head_ip:6379 --num-gpus=4 --disable-usage-stats --block; do
      echo 'worker retry ray join...'; sleep 10
    done" &
done

wait $HEAD_PID
echo "=== driver exited; stopping job ==="
scancel "$SLURM_JOB_ID"
