#!/bin/bash
# Megatron-side per-token logprobs for the parity comparison against sglang.
set -x
REPO=/data/home/zzeng/repos/miles-qwen4exp
MEGATRON=/data/home/zzeng/repos/Megatron-hcslot
MODEL=/scratch/models/Qwen3.8-Flash-Next
[ -d "$MODEL" ] || MODEL=/data/models/Qwen3.8-Flash-Next
CKPT=/data/home/zzeng/ckpt/qwen3.8-flash-next_torch_dist
OUT=/data/home/zzeng/parity

cd "$REPO"
MODEL_ARGS=$(PYTHONPATH="$REPO" python3 -c "
from miles.utils.external_utils.model_args_utils import load_model_args
print(load_model_args('qwen3.8-flash-next'))
")

export PYTHONPATH="$MEGATRON:$REPO"
export HOME=/data/home/zzeng
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=32
export CUDA_LAUNCH_BLOCKING=1
export QSA_BACKEND=triton
export TRITON_CACHE_DIR=/tmp/zz_triton

torchrun --nproc-per-node 4 --master-port 29651 \
  "$REPO/scripts/qwen3_8_next/megatron_logprobs.py" \
  $MODEL_ARGS \
  --tensor-model-parallel-size 4 \
  --expert-model-parallel-size 1 \
  --hf-checkpoint "$MODEL" \
  --load "$CKPT" \
  --tokens "$OUT/tokens4k.pt" \
  --parity-out "$OUT"
echo "PARITY_MEGATRON_EXIT=$?"
ls -l "$OUT"
