#!/bin/bash
# Offline PP=1 repro hunt for the e2e train-step SIGSEGV: loop fwd+bwd over
# randomly packed THD batches on the e2e's per-node topology (TP2+SP, EP2 here
# vs EP4 across nodes), triton QSA + flashqla GDN, recompute full.
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
export TORCHINDUCTOR_CACHE_DIR=/tmp/zz_inductor_cache
export TRITON_CACHE_DIR=/tmp/zz_triton_cache
export GDN_BACKEND=${GDN_BACKEND:-flashqla}
export QSA_BACKEND=${QSA_BACKEND:-triton}
export PYTHONFAULTHANDLER=1

torchrun --nproc-per-node 4 --master-port 29661 \
  "$REPO/scripts/qwen3_8_next/debug/megatron_logprobs.py" \
  $MODEL_ARGS \
  --tensor-model-parallel-size 2 --sequence-parallel \
  --expert-model-parallel-size 2 \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --hf-checkpoint "$MODEL" \
  --load "$CKPT" \
  --tokens "$OUT/tokens4k.pt" \
  --no-gradient-accumulation-fusion \
  --parity-out "$OUT" --bwd-stress ${STRESS_ITERS:-200}
echo "BWD_STRESS_EXIT=$?"
