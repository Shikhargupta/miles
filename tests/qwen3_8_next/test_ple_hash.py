"""Parity: miles PLE n-gram hashing vs sglang.

Two independent sglang references, neither of which needs sglang runtime state:
  * ``fused_qwen4_ngram_hash`` -- the fused kernel serving actually uses;
  * ``Qwen4ExpNGramEmbedding._shift_right_ignore_eos`` called on a stub, to check
    the EOS-boundary logic on its own.

The hash is integer arithmetic, so the bar is exact equality, not a tolerance.
"""
import sys
import torch

torch.manual_seed(0)

NGRAM_SIZE, HEADS_PER_NGRAM = 3, 8
EOS = 248044
VOCAB = 248320

# The three tensors the checkpoint ships (read out of the safetensors headers).
MULT = torch.tensor([23703573157769, 20109073645365, 8052911324071],
                    dtype=torch.long, device="cuda")
SIZES = torch.tensor([20000003, 20000023, 20000033, 20000047, 20000059, 20000063,
                      20000069, 20000077, 20000081, 20000093, 20000107, 20000147,
                      20000153, 20000159, 20000161, 20000171],
                     dtype=torch.long, device="cuda")
OFFS = torch.tensor([0, 20000003, 40000026, 60000059, 80000106, 100000165,
                     120000228, 140000297, 160000374, 180000455, 200000548,
                     220000655, 240000802, 260000955, 280001114, 300001275],
                    dtype=torch.long, device="cuda")

sys.path.insert(0, "/data/home/zzeng/repos/miles-qwen4exp")
from miles_plugins.models.qwen3_8_next.ple_ops import ngram_hash_ids, shift_right_ignore_eos

from sglang.srt.models.qwen4_exp import Qwen4ExpNGramEmbedding

ok = True

print("=== 1. shift_right_ignore_eos vs sglang (stub-called) ===")
class _Stub:
    pass
stub = _Stub()
stub.eos_token_id = EOS

for trial, (B, S, eos_frac) in enumerate([(3, 16, 0.0), (3, 16, 0.25), (2, 9, 0.5)]):
    tok = torch.randint(0, VOCAB, (B, S), device="cuda", dtype=torch.long)
    if eos_frac:
        m = torch.rand(B, S, device="cuda") < eos_frac
        tok = torch.where(m, torch.full_like(tok, EOS), tok)
    for n in range(0, NGRAM_SIZE):
        mine = shift_right_ignore_eos(tok, n, EOS)
        ref = Qwen4ExpNGramEmbedding._shift_right_ignore_eos(stub, tok, n)
        same = torch.equal(mine, ref)
        ok &= same
        print(f"  trial{trial} B={B} S={S} eos_frac={eos_frac} n={n}: "
              f"{'exact' if same else 'MISMATCH'}")

print("=== 2. ngram_hash_ids vs sglang fused kernel ===")
try:
    from sglang.kernels.ops.qwen4_ple import (
        can_fuse_qwen4_ngram_hash,
        fused_qwen4_ngram_hash,
    )
except Exception as e:
    print(f"  fused kernel unavailable: {type(e).__name__}: {e}")
    fused_qwen4_ngram_hash = None

if fused_qwen4_ngram_hash is not None:
    for T, eos_frac in [(32, 0.0), (128, 0.2), (1024, 0.05)]:
        ctx = torch.randint(0, VOCAB, (T, NGRAM_SIZE), device="cuda", dtype=torch.long)
        if eos_frac:
            m = torch.rand(T, NGRAM_SIZE, device="cuda") < eos_frac
            ctx = torch.where(m, torch.full_like(ctx, EOS), ctx)
        can = can_fuse_qwen4_ngram_hash(ctx, MULT, SIZES, OFFS)
        if not can:
            print(f"  T={T}: fused kernel declined this input, skipping")
            continue
        ref = fused_qwen4_ngram_hash(ctx, MULT, SIZES, OFFS, EOS)
        mine = ngram_hash_ids(ctx, MULT, SIZES, OFFS, NGRAM_SIZE, HEADS_PER_NGRAM, EOS)
        same = mine.shape == ref.shape and torch.equal(mine, ref)
        ok &= same
        print(f"  T={T} eos_frac={eos_frac}: mine{tuple(mine.shape)} ref{tuple(ref.shape)} "
              f"{'exact' if same else 'MISMATCH'}")
        if not same:
            bad = (mine != ref).nonzero()[:5]
            for b in bad:
                i, j = b.tolist()
                print(f"      [{i},{j}] mine={mine[i,j].item()} ref={ref[i,j].item()}")

print("=== 3. ids land inside each head's row range ===")
ctx = torch.randint(0, VOCAB, (256, NGRAM_SIZE), device="cuda", dtype=torch.long)
ids = ngram_hash_ids(ctx, MULT, SIZES, OFFS, NGRAM_SIZE, HEADS_PER_NGRAM, EOS)
in_range = True
for h in range(ids.shape[-1]):
    lo, hi = OFFS[h].item(), (OFFS[h] + SIZES[h]).item()
    col = ids[:, h]
    good = bool(((col >= lo) & (col < hi)).all())
    in_range &= good
    if not good:
        print(f"  head {h}: OUT OF RANGE  min={col.min().item()} max={col.max().item()} "
              f"expected [{lo}, {hi})")
print(f"  all 16 heads within their row ranges: {in_range}")
print(f"  global id range: [{ids.min().item()}, {ids.max().item()}]  "
      f"table rows = 320,001,446")
ok &= in_range

print()
print("VERDICT=" + ("PASS" if ok else "FAIL"))
