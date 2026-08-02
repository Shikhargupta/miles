#!/bin/bash
# Required pre-step for GLM-5.2 training with --dsa-attention-backend tilelang on sm90 (H200).
#
# The trainer does NOT run miles_plugins/models/glm5/ops/ -- the megatron-bridge provider builds
# its own layer spec, so the kernels that execute come from the installed megatron-bridge wheel.
# That copy enables TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE, a tilelang pass that aliases
# still-live shared-memory buffers and miscompiles the sparse-MLA backward: the forward stays
# clean but dQ/dKV come back NaN on ordinary valid causal indices, so training reports a finite
# loss alongside train/grad_norm = nan.
#
# Run this on EVERY node before launching, then verify with a standalone SparseMLA backward.
set -euo pipefail

BRIDGE_GLM5=/usr/local/lib/python3.12/dist-packages/megatron/bridge/models/glm5
FLAG=TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE

echo "### $(hostname) $(date -u '+%F %T') UTC"

for f in $(grep -rl "$FLAG: True" "$BRIDGE_GLM5" 2>/dev/null | grep -v __pycache__); do
    [ -f "$f.smem_merge.bak" ] || cp "$f" "$f.smem_merge.bak"
    sed -i "s/$FLAG: True/$FLAG: False/" "$f"
    echo "patched $f"
done

grep -rn "$FLAG" "$BRIDGE_GLM5" 2>/dev/null | grep -v __pycache__

if grep -rq "$FLAG: True" "$BRIDGE_GLM5" 2>/dev/null; then
    echo "FAILED: the flag is still True somewhere"
    exit 1
fi

python3 -c "
import ast, glob
for f in glob.glob('$BRIDGE_GLM5/tilelang/*.py'):
    ast.parse(open(f).read())
print('PARSE_OK')
"

# A miscompiled kernel is cached as a compiled binary, so the edit alone changes nothing.
rm -rf /root/.tilelang/cache
find "$BRIDGE_GLM5" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

echo PATCH_DONE
