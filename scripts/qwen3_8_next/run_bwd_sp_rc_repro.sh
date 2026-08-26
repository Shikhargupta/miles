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
# Compile caches must be node-local: HOME is on NFS, and two nodes sharing one
# inductor/triton cache produce "OSError: Stale file handle" mid-compile.
export TORCHINDUCTOR_CACHE_DIR=/tmp/zz_inductor_cache
export TRITON_CACHE_DIR=/tmp/zz_triton_cache
# Gradient dumps: run-to-run parity of the SAME code, so non-intrusive naming is
# fine (module paths only have to match themselves). BWD_TAG picks the exp dir.
export DUMPER_ENABLE=1
export DUMPER_ENABLE_GRAD=1
export DUMPER_DIR=/data/home/zzeng/parity/bwd
export DUMPER_EXP_NAME="${BWD_TAG:-bwd1}"
export DUMPER_CLEANUP_PREVIOUS=0

torchrun --nproc-per-node 4 --master-port 29643 \
  "$REPO/scripts/qwen3_8_next/megatron_logprobs.py" \
  $MODEL_ARGS \
  --tensor-model-parallel-size 4 --sequence-parallel --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --expert-model-parallel-size 1 \
  --hf-checkpoint "$MODEL" \
  --load "$CKPT" \
  --tokens "$OUT/tokens4k.pt" \
  --no-gradient-accumulation-fusion \
  --parity-out "$OUT" --split-every 700 --backward
echo "PARITY_BWD_EXIT=$?"
ls -l "$OUT"
