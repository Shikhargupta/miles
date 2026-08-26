"""Numerical validation: triton PLE gate+conv vs the torch reference chain.

Reference = the exact ops/ple.py math (grouped_gemma_rmsnorm + dot gate +
causal_depthwise_conv + SiLU + residual), autograd for gradients. Covers
multi-segment cu_seqlens (the training THD case), tiny segments shorter than
the conv receptive field, and the real widths.
"""

import math
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/data/home/zzeng/repos/miles-qwen4exp")

from miles_plugins.models.qwen3_8_next.ops.hc import grouped_gemma_rmsnorm
from miles_plugins.models.qwen3_8_next.ops.ple import causal_depthwise_conv
from miles_plugins.models.qwen3_8_next.ops.kernel.ple_triton import ple_gate_conv_triton


def torch_reference(hc, key, value, wk, wq, wc, convw, n, eps, dil, cu):
    T = hc.shape[0]
    C = hc.shape[1] // n
    kn = grouped_gemma_rmsnorm(key, wk, n, eps).reshape(T, n, C)
    qn = grouped_gemma_rmsnorm(hc, wq, n, eps).reshape(T, n, C)
    score = (kn * qn).sum(dim=-1, keepdim=True) / math.sqrt(C)
    gate = torch.sigmoid(score.abs().clamp_min(1e-6).sqrt() * score.sign())
    gated = (gate * value.unsqueeze(-2)).flatten(-2)
    gn = grouped_gemma_rmsnorm(gated, wc, n, eps)
    conv = causal_depthwise_conv(gn.to(convw.dtype), convw, dil, cu)
    conv = F.silu(conv)
    return (gated.to(conv.dtype) + conv).to(hc.dtype)


def rel_err(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item()


def run_case(T, C, n, K, dil, dtype, seed, tol, segs):
    g = torch.Generator(device="cuda").manual_seed(seed)
    W = n * C
    eps = 1e-6
    mk = lambda *s: torch.randn(*s, device="cuda", dtype=dtype, generator=g)
    hc = mk(T, W).requires_grad_()
    key = mk(T, W).requires_grad_()
    value = mk(T, C).requires_grad_()
    wk = (0.05 * mk(W)).requires_grad_()
    wq = (0.05 * mk(W)).requires_grad_()
    wc = (0.05 * mk(W)).requires_grad_()
    convw = (mk(W, 1, K) / K).requires_grad_()
    cu = torch.tensor(segs, dtype=torch.int32, device="cuda") if segs else None
    dout = mk(T, W)

    params = (hc, key, value, wk, wq, wc, convw)
    ref = torch_reference(hc, key, value, wk, wq, wc, convw, n, eps, dil, cu)
    ref.backward(dout)
    ref_grads = [p.grad.clone() for p in params]
    for p in params:
        p.grad = None

    tri = ple_gate_conv_triton(hc, key, value, wk, wq, wc, convw, n, eps, dil, cu)
    tri.backward(dout)
    tri_grads = [p.grad.clone() for p in params]

    errs = {"out": rel_err(tri, ref)}
    names = ["dhc", "dkey", "dvalue", "dwk", "dwq", "dwc", "dconvw"]
    for name, r, t in zip(names, ref_grads, tri_grads):
        errs[name] = rel_err(t, r)
    bad = {k: v for k, v in errs.items() if v > tol}
    status = "OK " if not bad else "FAIL"
    print(f"[ple {status}] T={T} C={C} n={n} K={K} d={dil} segs={bool(segs)} {dtype}: "
          + " ".join(f"{k}={v:.2e}" for k, v in errs.items()))
    return not bad


def main():
    ok = True
    # fp32 tol is looser than the HC/QSA tests: the gate is sigmoid(sign*sqrt(|s|))
    # and d/ds = 1/(2*sqrt(|s|)) blows up toward the 1e-6 clamp knee, so per-token
    # scores that land near the knee amplify fp32 summation-order differences
    # between tl.sum and torch's reduction into ~1e-4 relative on the gate-side
    # grads (dhc/dkey/dwk/dwq). Away from the knee the same shapes sit at ~2e-5.
    for dtype, tol in [(torch.float32, 5e-4), (torch.bfloat16, 4e-2)]:
        ok &= run_case(37, 64, 4, 4, 3, dtype, 1, tol, segs=None)
        ok &= run_case(64, 64, 4, 4, 3, dtype, 2, tol, segs=[0, 5, 6, 30, 64])
        ok &= run_case(128, 2560, 4, 4, 3, dtype, 3, tol, segs=[0, 1, 3, 70, 128])
        ok &= run_case(621, 2560, 4, 4, 3, dtype, 4, tol, segs=[0, 200, 621])
    print("ALL_OK" if ok else "SOME_FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
