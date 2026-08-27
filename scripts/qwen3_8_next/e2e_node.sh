#!/bin/bash
# Per-node container entry for the e2e RL run: joins (or heads) the ray cluster and
# stays alive. Role and head address come from env (ROLE=head|worker, HEAD_IP).
set -x
export HOME=/data/home/zzeng
export HF_HOME=/data/home/zzeng/.cache/huggingface
export PYTHONUNBUFFERED=1
# Compile caches must be node-local: HOME is on NFS, and two nodes sharing one
# inductor/triton cache produce "OSError: Stale file handle" mid-compile.
export TORCHINDUCTOR_CACHE_DIR=/tmp/zz_inductor_cache
export TRITON_CACHE_DIR=/tmp/zz_triton_cache
# Seed the node-local triton cache from the persistent copy: fresh containers
# start cold, and cold bwd kernels JIT inside the 1F1B pipeline (serialized
# across stages). Refresh the seed from a warm node after kernel changes:
#   rsync -a /tmp/zz_triton_cache/ /data/home/zzeng/cache/triton_seed/
SEED=/data/home/zzeng/cache/triton_seed
[ -d "$SEED" ] && mkdir -p "$TRITON_CACHE_DIR" && cp -rn "$SEED/." "$TRITON_CACHE_DIR/" 2>/dev/null

# hostname -I's first entry is 10.0.1.2 on EVERY node (a shared address); the
# per-node fabric IP is the one DNS resolves for the hostname (10.4.90.x).
IP=$(getent hosts $(hostname) | awk 'NR==1{print $1}')
echo "NODE_IP=$IP ROLE=$ROLE HEAD_IP=$HEAD_IP"

# The memory monitor killed two R3 runs at 95% while the true node peak (898GB
# of 944GB, sleep()'s TMS CPU backup on top of the head node's extra services)
# fits physically. The env must be on the RAYLET process, i.e. set before
# `ray start`, not on the driver.
export RAY_memory_usage_threshold=0.98

ray stop --force 2>/dev/null; pkill -9 -f "ray::" 2>/dev/null; sleep 2

if [ "$ROLE" = "head" ]; then
  ray start --head --node-ip-address "$IP" --num-gpus 4 --disable-usage-stats
else
  # wait for the head to come up
  for i in $(seq 1 60); do
    ray start --address="$HEAD_IP:6379" --node-ip-address "$IP" --num-gpus 4 --disable-usage-stats && break
    sleep 5
  done
fi
echo "RAY_STARTED role=$ROLE"
sleep infinity
