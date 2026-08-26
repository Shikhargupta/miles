#!/bin/bash
# HF -> Megatron torch_dist for Qwen3.8-Flash-Next, TP4 on one node.
#
# Reads the checkpoint from node-local NVMe (/scratch), which is ~9x faster than the
# shared VAST mount at the same read concurrency. Writes the result to /data so all
# 8 nodes see it.
set -x
REPO=/data/home/zzeng/repos/miles-qwen4exp
MEGATRON=/data/home/zzeng/repos/Megatron-hcslot
MODEL=/scratch/models/Qwen3.8-Flash-Next
[ -d "$MODEL" ] || MODEL=/data/models/Qwen3.8-Flash-Next
OUT=/data/home/zzeng/ckpt/qwen3.8-flash-next_torch_dist

mkdir -p "$(dirname "$OUT")"
cd "$REPO"

MODEL_ARGS=$(PYTHONPATH="$REPO:$REPO/miles/utils/external_utils" python3 -c "
import sys
from miles.utils.external_utils.model_args_utils import load_model_args
print(load_model_args('qwen3.8-flash-next'))
")
echo "MODEL_ARGS=$MODEL_ARGS"

# The converter defaults to sharding by pipeline: when PP is 1 and world_size > 1 it
# sets PP = world_size and leaves TP alone, so asking for TP4 on 4 GPUs made it
# compute total_model_size = TP4 x PP4 = 16 and refuse. CONVERT_KEEP_PP1 is the
# switch it provides for exactly this. (torch_dist is a resharding-friendly format,
# so the parallelism used to write it does not have to match the training layout --
# TP4 here is to match what training will use, not a requirement of the format.)
# torchrun defaults OMP_NUM_THREADS to 1 "to avoid your system being overloaded",
# which makes every CPU tensor op in the bridge single-threaded. That matters here
# because mbridge does its reshape/concat/TP-slice work on the CPU, and this model's
# fused MoE experts are ~2.5 B elements per layer -- 123 B elements in total, all
# through one core of 144. Measured: 3.4 min/layer, i.e. ~2.8 h for 48 layers, with
# exactly one thread in state R out of 168.
export OMP_NUM_THREADS=32

# Secondary: torchrun defaults OMP_NUM_THREADS to 1, which serialises whatever CPU
# tensor work is left (the safetensors read itself). The bridge's heavy reshape and
# split work is on the GPU now -- see Qwen38NextBridge._weight_to_mcore_format.
export OMP_NUM_THREADS=32

export CONVERT_KEEP_PP1=1
export PYTHONPATH="$MEGATRON:$REPO"
export HOME=/data/home/zzeng
export CUDA_DEVICE_MAX_CONNECTIONS=1

torchrun --nproc-per-node 4 --master-port 29577 \
  "$REPO/tools/convert_hf_to_torch_dist.py" \
  $MODEL_ARGS \
  --tensor-model-parallel-size 4 \
  --expert-model-parallel-size 1 \
  --hf-checkpoint "$MODEL" \
  --save "$OUT"
rc=$?
echo "CONVERT_EXIT=$rc"
ls -la "$OUT" 2>/dev/null | head
du -sh "$OUT" 2>/dev/null
