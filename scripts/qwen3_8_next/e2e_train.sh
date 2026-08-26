#!/bin/bash
# Runs inside the HEAD node's container after all workers joined.
set -x
export HOME=/data/home/zzeng
export HF_HOME=/data/home/zzeng/.cache/huggingface
export PYTHONUNBUFFERED=1
export MILES_SCRIPT_EXTERNAL_RAY=1
export MASTER_ADDR=$(getent hosts $(hostname) | awk 'NR==1{print $1}')
export WANDB_ENTITY=miles_training
export WANDB_API_KEY=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
export PYTHONPATH=/data/home/zzeng/repos/miles-qwen4exp:/data/home/zzeng/repos/Megatron-hcslot:/data/home/zzeng/repos/sglang-B/python

cd /data/home/zzeng/repos/miles-qwen4exp

# All 8 nodes must be in before the job is submitted.
for i in $(seq 1 120); do
  n=$(ray status 2>/dev/null | grep -c "node_")
  echo "ray nodes visible: $n"
  [ "$n" -ge 8 ] && break
  sleep 5
done
ray status | head -20

# typer with a single command treats it as the main command: no subcommand name
python3 scripts/run_qwen3_8_next.py ${TRAIN_EXTRA:-}
echo "E2E_TRAIN_EXIT=$?"
