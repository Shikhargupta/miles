"""Qwen3.8-Next hyper-connection verification, three layers:

  1. FWD vs sglang (bf16, the JIT kernels serving actually uses) -- expect
     agreement to ~1 ulp of bf16, not bit-exactness, since the reduction order
     differs.
  2. FWD vs an independent float64 reference -- decides whether any bf16 gap is
     just rounding or a real formula error, by checking which side is closer to
     the exact answer.
  3. BWD via gradcheck in float64. sglang's JIT kernels carry no autograd
     (it is an inference engine: sg_out.requires_grad is False), so sglang
     cannot serve as a backward reference at all; the exact-math reference is
     the only sound one.

Imports only miles_plugins.models.qwen3_8_next.ops (pure torch): no megatron.core,
so Transformer Engine's ABI problem is out of the picture.
"""
import torch

torch.manual_seed(0)

HC, HIDDEN, LOWRANK, EPS = 4, 2560, 320, 1e-6
S, B = 7, 2
TOKENS = S * B

from reference_ops import (
    grouped_gemma_rmsnorm, hc_combine, hc_inject_gate, hc_mix,
)

# ---------------------------------------------------------------- reference
def ref_mix(x, wn, wd, wu, n, hidden, eps):
    """Independent transcription of sglang GatedResidual.mix, dtype-generic."""
    xг = x.unflatten(-1, (n, hidden))
    var = xг.pow(2).mean(-1, keepdim=True)
    normed = ((xг * torch.rsqrt(var + eps)).flatten(-2)) * (1.0 + wn)
    g = torch.nn.functional.silu(normed @ wd.T / n)
    g = torch.sigmoid(g @ wu.T)
    agg = (g.unflatten(-1, (n, hidden)) * normed.unflatten(-1, (n, hidden))).mean(-2)
    return agg, normed

def ref_combine(x, normed, y, wi, n, hidden):
    a = 2 * torch.sigmoid(normed @ wi.T / n)
    return (x.unflatten(-1, (n, hidden)) + y.unsqueeze(-2) * a.unsqueeze(-1)).flatten(-2)

# ---------------------------------------------------------------- weights
from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig
sg = GatedResidual(HyperConnectionConfig(
    hc_count=HC, hidden_size=HIDDEN, params_dtype=torch.bfloat16,
    hc_lowrank=LOWRANK, rms_norm_eps=EPS, hc_per_branch_norm=True,
)).cuda()
# weight is built as torch.zeros(hidden) with no dtype -> fp32; at runtime the
# loader hands it the checkpoint's bf16, and the JIT kernel requires bf16.
sg.hc_norm.weight.data = sg.hc_norm.weight.data.to(torch.bfloat16)
with torch.no_grad():
    for p in (sg.hc_norm.weight, sg.input_mix_weight_down.weight,
              sg.input_mix_weight_up.weight, sg.block_inject_weight.weight):
        p.normal_(0, 0.02)

WN = sg.hc_norm.weight.detach().clone()
WD = sg.input_mix_weight_down.weight.detach().clone()
WU = sg.input_mix_weight_up.weight.detach().clone()
WI = sg.block_inject_weight.weight.detach().clone()

x = torch.randn(TOKENS, HC * HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.5
y = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.5

def stats(a, b):
    a, b = a.double(), b.double()
    d = (a - b).abs()
    return d.max().item(), d.mean().item()

# ---------------------------------------------------------------- 1 + 2
print("=== exact (float64) reference ===")
x64, y64 = x.double(), y.double()
r_agg, r_normed = ref_mix(x64, WN.double(), WD.double(), WU.double(), HC, HIDDEN, EPS)
r_out = ref_combine(x64, r_normed, y64, WI.double(), HC, HIDDEN)
scale_mix = r_agg.abs().max().item()
scale_comb = r_out.abs().max().item()
print(f"  |mix|max={scale_mix:.4f}  |combine|max={scale_comb:.4f}")
print(f"  1 ulp of bf16 at these scales: mix={scale_mix*2**-8:.3e}  combine={scale_comb*2**-8:.3e}")

print("=== ours (bf16) vs sglang (bf16) ===")
sg_agg, sg_res = sg.mix(x)
o_normed = grouped_gemma_rmsnorm(x, WN, HC, EPS)
o_agg = hc_mix(o_normed, WD, WU, HC, HIDDEN, out_dtype=torch.bfloat16)
sg_out = sg.combine(y, sg_res)
o_post = hc_inject_gate(o_normed, WI, HC)
o_out = hc_combine(x, y, o_post, HC, HIDDEN)
m, mm = stats(o_agg, sg_agg);   print(f"  mix      max_abs={m:.3e} mean_abs={mm:.3e}")
c, cm = stats(o_out, sg_out);   print(f"  combine  max_abs={c:.3e} mean_abs={cm:.3e}")

print("=== irreducible floor: rounding the exact answer to bf16 ===")
floor_mix = (r_agg.to(torch.bfloat16).double() - r_agg).abs().max().item()
floor_comb = (r_out.to(torch.bfloat16).double() - r_out).abs().max().item()
print(f"  mix={floor_mix:.3e}  combine={floor_comb:.3e}")
print("=== distance to exact reference (lower is better) ===")
om, _ = stats(o_agg, r_agg);  sm, _ = stats(sg_agg, r_agg)
oc, _ = stats(o_out, r_out);  sc, _ = stats(sg_out, r_out)
print(f"  mix      ours={om:.3e}   sglang={sm:.3e}   ratio={om/max(sm,1e-30):.2f}x")
print(f"  combine  ours={oc:.3e}   sglang={sc:.3e}   ratio={oc/max(sc,1e-30):.2f}x")
print(f"  ours/floor:  mix={om/max(floor_mix,1e-30):.2f}x   combine={oc/max(floor_comb,1e-30):.2f}x")
print(f"  sglang/floor: mix={sm/max(floor_mix,1e-30):.2f}x   combine={sc/max(floor_comb,1e-30):.2f}x")

fwd_ok = (om <= 4 * scale_mix * 2**-8) and (oc <= 4 * scale_comb * 2**-8)
comparable = (om <= 3 * max(sm, 1e-30)) and (oc <= 3 * max(sc, 1e-30))

# ---------------------------------------------------------------- 3
print("=== backward: gradcheck in float64 (small dims) ===")
sn, sh, sl, shc = 3, 8, 4, 4
gwn = torch.randn(shc * sh, device="cuda", dtype=torch.float64) * 0.02
gwd = torch.randn(sl, shc * sh, device="cuda", dtype=torch.float64) * 0.02
gwu = torch.randn(shc * sh, sl, device="cuda", dtype=torch.float64) * 0.02
gwi = torch.randn(shc, shc * sh, device="cuda", dtype=torch.float64) * 0.02
gx = torch.randn(sn, shc * sh, device="cuda", dtype=torch.float64)
gy = torch.randn(sn, sh, device="cuda", dtype=torch.float64)

def f_mix(x_, wn_, wd_, wu_):
    nd = grouped_gemma_rmsnorm(x_, wn_, shc, EPS)
    return hc_mix(nd, wd_, wu_, shc, sh, out_dtype=x_.dtype)

def f_comb(x_, wn_, wi_, y_):
    nd = grouped_gemma_rmsnorm(x_, wn_, shc, EPS)
    return hc_combine(x_, y_, hc_inject_gate(nd, wi_, shc), shc, sh)

gc1 = torch.autograd.gradcheck(
    f_mix, (gx.clone().requires_grad_(), gwn.clone().requires_grad_(),
            gwd.clone().requires_grad_(), gwu.clone().requires_grad_()),
    eps=1e-6, atol=1e-7, rtol=1e-5, raise_exception=False)
gc2 = torch.autograd.gradcheck(
    f_comb, (gx.clone().requires_grad_(), gwn.clone().requires_grad_(),
             gwi.clone().requires_grad_(), gy.clone().requires_grad_()),
    eps=1e-6, atol=1e-7, rtol=1e-5, raise_exception=False)
print(f"  gradcheck(mix)     = {gc1}")
print(f"  gradcheck(combine) = {gc2}")
print(f"  sglang provides grads? {sg_out.requires_grad}  (inference engine: expected False)")

print("=== layout invariance: [s,b,n*C] vs flat [tokens,n*C] ===")
x3 = x.view(S, B, HC * HIDDEN).contiguous(); y3 = y.view(S, B, HIDDEN).contiguous()
n3 = grouped_gemma_rmsnorm(x3, WN, HC, EPS)
d1 = (hc_mix(n3, WD, WU, HC, HIDDEN, out_dtype=torch.bfloat16).reshape(TOKENS, HIDDEN).double() - o_agg.double()).abs().max().item()
d2 = (hc_combine(x3, y3, hc_inject_gate(n3, WI, HC), HC, HIDDEN).reshape(TOKENS, HC*HIDDEN).double() - o_out.double()).abs().max().item()
print(f"  mix={d1:.3e}  combine={d2:.3e}  (expect exactly 0)")
layout_ok = (d1 == 0.0 and d2 == 0.0)

print()
print(f"FWD_WITHIN_BF16_ULP={fwd_ok}")
print(f"ACCURACY_COMPARABLE_TO_SGLANG={comparable}")
print(f"GRADCHECK={gc1 and gc2}")
print(f"LAYOUT_INVARIANT={layout_ok}")
print("VERDICT=" + ("PASS" if (fwd_ok and comparable and gc1 and gc2 and layout_ok) else "FAIL"))
