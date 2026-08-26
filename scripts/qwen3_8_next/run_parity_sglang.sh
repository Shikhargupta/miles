#!/bin/bash
# sglang half of the fwd parity: launch the ported server, score the same tokens,
# shut it down. Speculative decoding is deliberately left off -- we only score the
# prompt, so the draft model cannot affect the numbers, and dropping it removes a
# moving part and a few minutes of startup.
# NOTE: never edit this file while an instance is running; bash reads scripts
# incrementally.
set -x
# sglang-B is the cluster-side copy of the local zhichen/qwen4exp-v0518 worktree
# (same two commits on the same base) and is the tree that actually served this
# model: SERVER_READY 254s, RESULT=Bnew:OK. The sglang-qwen4exp directory here is a
# partial sync -- 425 of 3353 files, missing sglang.srt.environ -- so it cannot run.
SGLANG=/data/home/zzeng/repos/sglang-B
REPO=/data/home/zzeng/repos/miles-qwen4exp
MODEL=/scratch/models/Qwen3.8-Flash-Next
[ -d "$MODEL" ] || MODEL=/data/models/Qwen3.8-Flash-Next
OUT=/data/home/zzeng/parity
PORT=30001
LOG=/data/home/zzeng/logs/server-parity.log

export PYTHONPATH="$SGLANG/python"
export HOME=/data/home/zzeng
export HF_HOME=/data/home/zzeng/.cache/huggingface

# Dumper, enabled over HTTP rather than at startup, for two reasons. The dumper
# copies tensors to host inside the forward, which makes CUDA graph capture fail
# ("operation failed due to a previous error during capture"), and the server's
# warmup passes would otherwise fill the dump directory with tensors for tokens we
# never asked about. So: patch the source at import time, keep dumping off through
# startup, and turn it on for exactly the one scoring request.
if [ -n "$DUMP_DIR" ]; then
  export DUMPER_ENABLE=0
  export DUMPER_DIR="$DUMP_DIR"
  export DUMPER_EXP_NAME=sglang
  export DUMPER_NON_INTRUSIVE_MODE=core
  export DUMPER_SOURCE_PATCHER_CONFIG="$REPO/scripts/qwen3_8_next/dump_patch_sglang.yaml"
  export DUMPER_SERVER_PORT=reuse
  export DUMPER_CLEANUP_PREVIOUS=0
  EXTRA_ARGS="--disable-cuda-graph"
fi

T0=$(date +%s)
nohup python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp 4 --host 127.0.0.1 --port $PORT \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 8192 \
  --linear-attn-prefill-backend flashinfer \
  --max-running-requests 8 \
  --disable-radix-cache \
  $EXTRA_ARGS \
  > "$LOG" 2>&1 &
SRV=$!
echo "server pid=$SRV"

READY=0
for i in $(seq 1 360); do
  if ! kill -0 $SRV 2>/dev/null; then echo "SERVER_DIED after $(( $(date +%s) - T0 ))s"; break; fi
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:$PORT/health" 2>/dev/null)
  if [ "$code" = "200" ]; then READY=1; echo "SERVER_READY after $(( $(date +%s) - T0 ))s"; break; fi
  sleep 10
done
if [ "$READY" != "1" ]; then tail -60 "$LOG"; echo "PARITY_SGLANG_EXIT=1"; exit 1; fi

if [ -n "$DUMP_DIR" ]; then
  curl -s -X POST "http://127.0.0.1:$PORT/dumper/configure" \
    -H 'Content-Type: application/json' -d '{"enable": true}' | head -c 400; echo
fi

python3 "$REPO/scripts/qwen3_8_next/sglang_logprobs.py" \
  --url "http://127.0.0.1:$PORT" \
  --tokens "$OUT/tokens.pt" \
  --out "$OUT/sglang_logprobs.pt"
RC=$?

# Second scoring pass against the same server: with the radix cache disabled the
# prefill recomputes, so any difference between the two files is sglang's own
# run-to-run nondeterminism -- the floor below which no cross-framework number
# can be trusted.
python3 "$REPO/scripts/qwen3_8_next/sglang_logprobs.py" \
  --url "http://127.0.0.1:$PORT" \
  --tokens "$OUT/tokens.pt" \
  --out "$OUT/sglang_logprobs_2.pt"

if [ -n "$DUMP_DIR" ]; then
  curl -s -X POST "http://127.0.0.1:$PORT/dumper/configure" \
    -H 'Content-Type: application/json' -d '{"enable": false}' | head -c 200; echo
fi

kill -9 $SRV 2>/dev/null
sleep 5
echo "PARITY_SGLANG_EXIT=$RC"
