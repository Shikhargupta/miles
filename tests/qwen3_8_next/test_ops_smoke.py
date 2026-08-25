"""Import + semantics smoke test for the Qwen3.8-Next ops.

Cheap checks that catch the mistakes that do not raise: a compression that
silently dilutes its last block with padding, a causal mask off by one block, a
conv that reaches into the future.
"""
import torch

from miles_plugins.models.qwen3_8_next.ops.hc import grouped_gemma_rmsnorm, hc_mix
from miles_plugins.models.qwen3_8_next.ops.ple_hash import ngram_hash_ids
from miles_plugins.models.qwen3_8_next.ops.ple import Qwen38NextPLE, causal_depthwise_conv
from miles_plugins.models.qwen3_8_next.ops.ple_embedding import Qwen38NextFrozenNGramEmbedding
from miles_plugins.models.qwen3_8_next.ops.qsa_indexer import (
    Qwen38NextQSAIndexer,
    block_causal_mask,
    compress_keys_by_mean,
    gemma_rmsnorm_last_dim,
)
from miles_plugins.models.qwen3_8_next.hyper_connection import (
    Qwen38NextHCHeadContraction,
    Qwen38NextHyperConnection,
)
from miles_plugins.models.qwen3_8_next.qwen3_8_next import get_qwen3_8_next_spec
from miles_plugins.mbridge.qwen3_8_next import Qwen38NextBridge

ok = True
print("all Qwen3.8-Next modules import OK")

print("=== compress_keys_by_mean ===")
tk = torch.arange(24, dtype=torch.float32).reshape(12, 2)
c = compress_keys_by_mean(tk, 4)
want0 = tk[:4].mean(0)
good = torch.allclose(c[0], want0)
ok &= good
print(f"  [12,2] r=4 -> {tuple(c.shape)}, block0 == mean(rows 0..3): {good}")
c2 = compress_keys_by_mean(tk[:10], 4)          # trailing partial block of 2
want_last = tk[8:10].mean(0)
good = c2.shape[0] == 3 and torch.allclose(c2[-1], want_last)
ok &= good
print(f"  partial tail [10,2] r=4 -> {tuple(c2.shape)}, last block == mean(rows 8..9) "
      f"(not diluted by padding): {good}")

print("=== block_causal_mask ===")
pos = torch.tensor([0, 3, 4, 7, 8])
m = block_causal_mask(pos, num_blocks=3, compress_ratio=4)
for i, p in enumerate(pos.tolist()):
    vis = m[i].nonzero().flatten().tolist()
    expect = list(range(min((p + 1) // 4, 3)))
    good = vis == expect
    ok &= good
    print(f"  pos {p}: visible {vis}, expected {expect}: {good}")

print("=== gemma_rmsnorm_last_dim ===")
x = torch.randn(6, 8)
n = gemma_rmsnorm_last_dim(x, torch.zeros(8), 1e-6)
rms = n.pow(2).mean(-1).sqrt()
good = torch.allclose(rms, torch.ones_like(rms), atol=1e-5)
ok &= good
print(f"  zero weight -> unit RMS per row: {good} (mean {rms.mean():.6f})")

print("=== causal_depthwise_conv ===")
# A filter that is 1 only at the newest tap must reproduce its input exactly;
# any leakage from future positions would show up here.
cw = torch.zeros(8, 1, 4)
cw[:, 0, -1] = 1.0
xx = torch.arange(5 * 8, dtype=torch.float32).reshape(5, 8)
good = torch.equal(causal_depthwise_conv(xx, cw, dilation=3), xx)
ok &= good
print(f"  identity-at-newest-tap reproduces input (no future leakage): {good}")
# With cu_seqlens the conv must not carry state across a document boundary.
cw2 = torch.zeros(8, 1, 4)
cw2[:, 0, -2] = 1.0                              # one step back, dilation 1
cu = torch.tensor([0, 2, 5])
out = causal_depthwise_conv(xx, cw2, dilation=1, cu_seqlens=cu)
good = torch.equal(out[2], torch.zeros(8))       # first row of segment 2 has no history
ok &= good
print(f"  cu_seqlens resets history at a document boundary: {good}")

print()
print("VERDICT=" + ("PASS" if ok else "FAIL"))
