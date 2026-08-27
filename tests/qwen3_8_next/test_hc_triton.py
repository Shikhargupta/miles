"""Numerical validation: triton HC kernels vs the torch reference in ops/hc.py.

The torch chain (grouped_gemma_rmsnorm -> hc_mix / hc_inject_gate / hc_combine)
is the parity-verified reference; autograd through it defines the expected
gradients. The triton path must match fwd and every grad to fp32 tolerance in
fp32 and to the bf16 floor in bf16.
"""

import os
import sys

import torch

sys.path.insert(0, "/data/home/zzeng/repos/miles-qwen4exp")
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))

from reference_ops import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_inject_gate,
    hc_mix,
)
from miles_plugins.models.qwen3_8_next.ops.kernel.hc_triton import (
    hc_combine_triton,
    hc_mix_inject_triton,
)


def make_params(W, R, n, dtype, g):
    weight = (0.05 * torch.randn(W, device="cuda", dtype=dtype, generator=g)).requires_grad_()
    w_down = (torch.randn(R, W, device="cuda", dtype=dtype, generator=g) / W**0.5).requires_grad_()
    w_up = (torch.randn(W, R, device="cuda", dtype=dtype, generator=g) / R**0.5).requires_grad_()
    w_inj = (torch.randn(n, W, device="cuda", dtype=dtype, generator=g) / W**0.5).requires_grad_()
    return weight, w_down, w_up, w_inj


def rel_err(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item()


def run_mix_case(T, C, n, R, dtype, seed, tol):
    g = torch.Generator(device="cuda").manual_seed(seed)
    W = n * C
    eps = 1e-6
    x = torch.randn(T, W, device="cuda", dtype=dtype, generator=g).requires_grad_()
    weight, w_down, w_up, w_inj = make_params(W, R, n, dtype, g)

    dmix = torch.randn(T, C, device="cuda", dtype=dtype, generator=g)
    dhp = torch.randn(T, n, device="cuda", dtype=torch.float32, generator=g)

    # reference
    normed = grouped_gemma_rmsnorm(x, weight, n, eps)
    mix_ref = hc_mix(normed, w_down, w_up, n, C, out_dtype=x.dtype)
    hp_ref = hc_inject_gate(normed, w_inj, n)
    torch.autograd.backward([mix_ref, hp_ref], [dmix, dhp])
    ref_grads = [t.grad.clone() for t in (x, weight, w_down, w_up, w_inj)]
    for t in (x, weight, w_down, w_up, w_inj):
        t.grad = None

    # triton
    mix_tri, hp_tri = hc_mix_inject_triton(x, weight, w_down, w_up, w_inj, n, eps)
    torch.autograd.backward([mix_tri, hp_tri], [dmix, dhp])
    tri_grads = [t.grad.clone() for t in (x, weight, w_down, w_up, w_inj)]

    errs = {
        "mix": rel_err(mix_tri, mix_ref),
        "hpost": rel_err(hp_tri, hp_ref),
    }
    for name, r, t in zip(["dx", "dw_norm", "dw_down", "dw_up", "dw_inj"], ref_grads, tri_grads):
        errs[name] = rel_err(t, r)
    bad = {k: v for k, v in errs.items() if v > tol}
    status = "OK " if not bad else "FAIL"
    print(f"[mix {status}] T={T} C={C} n={n} R={R} {dtype}: " + " ".join(f"{k}={v:.2e}" for k, v in errs.items()))
    return not bad


def run_headmix_case(T, C, n, R, dtype, seed, tol):
    g = torch.Generator(device="cuda").manual_seed(seed)
    W = n * C
    eps = 1e-6
    x = torch.randn(T, W, device="cuda", dtype=dtype, generator=g).requires_grad_()
    weight, w_down, w_up, _ = make_params(W, R, n, dtype, g)
    dmix = torch.randn(T, C, device="cuda", dtype=dtype, generator=g)

    normed = grouped_gemma_rmsnorm(x, weight, n, eps)
    mix_ref = hc_mix(normed, w_down, w_up, n, C, out_dtype=x.dtype)
    mix_ref.backward(dmix)
    ref_grads = [t.grad.clone() for t in (x, weight, w_down, w_up)]
    for t in (x, weight, w_down, w_up):
        t.grad = None

    mix_tri, _ = hc_mix_inject_triton(x, weight, w_down, w_up, None, n, eps)
    mix_tri.backward(dmix)
    tri_grads = [t.grad.clone() for t in (x, weight, w_down, w_up)]

    errs = {"mix": rel_err(mix_tri, mix_ref)}
    for name, r, t in zip(["dx", "dw_norm", "dw_down", "dw_up"], ref_grads, tri_grads):
        errs[name] = rel_err(t, r)
    bad = {k: v for k, v in errs.items() if v > tol}
    status = "OK " if not bad else "FAIL"
    print(f"[head {status}] T={T} C={C} n={n} R={R} {dtype}: " + " ".join(f"{k}={v:.2e}" for k, v in errs.items()))
    return not bad


def run_combine_case(T, C, n, dtype, seed, tol):
    g = torch.Generator(device="cuda").manual_seed(seed)
    W = n * C
    res = torch.randn(T, W, device="cuda", dtype=dtype, generator=g).requires_grad_()
    y = torch.randn(T, C, device="cuda", dtype=dtype, generator=g).requires_grad_()
    hp = torch.rand(T, n, device="cuda", dtype=torch.float32, generator=g).mul(2).requires_grad_()
    dout = torch.randn(T, W, device="cuda", dtype=dtype, generator=g)

    out_ref = hc_combine(res, y, hp, n, C)
    out_ref.backward(dout)
    ref_grads = [t.grad.clone() for t in (res, y, hp)]
    for t in (res, y, hp):
        t.grad = None

    out_tri = hc_combine_triton(res, y, hp, n)
    out_tri.backward(dout)
    tri_grads = [t.grad.clone() for t in (res, y, hp)]

    errs = {"out": rel_err(out_tri, out_ref)}
    for name, r, t in zip(["dres", "dy", "dhpost"], ref_grads, tri_grads):
        errs[name] = rel_err(t, r)
    bad = {k: v for k, v in errs.items() if v > tol}
    status = "OK " if not bad else "FAIL"
    print(f"[comb {status}] T={T} C={C} n={n} {dtype}: " + " ".join(f"{k}={v:.2e}" for k, v in errs.items()))
    return not bad


def main():
    torch.manual_seed(0)
    ok = True
    # real model shape is C=2560, n=4, R=320; include small/odd shapes too
    for T, C, n, R in [(7, 64, 4, 16), (33, 256, 4, 32), (128, 2560, 4, 320), (621, 2560, 4, 320)]:
        for dtype, tol_mix, tol_comb in [
            (torch.float32, 2e-5, 1e-5),
            (torch.bfloat16, 3e-2, 1e-2),
        ]:
            ok &= run_mix_case(T, C, n, R, dtype, seed=T + 1, tol=tol_mix)
            ok &= run_headmix_case(T, C, n, R, dtype, seed=T + 2, tol=tol_mix)
            ok &= run_combine_case(T, C, n, dtype, seed=T + 3, tol=tol_comb)
    # 3D leading shape as used in megatron ([s, b, W])
    g = torch.Generator(device="cuda").manual_seed(99)
    x3 = torch.randn(17, 2, 4 * 64, device="cuda", dtype=torch.float32, generator=g)
    weight, w_down, w_up, w_inj = make_params(4 * 64, 16, 4, torch.float32, g)
    m3, hp3 = hc_mix_inject_triton(x3, weight, w_down, w_up, w_inj, 4, 1e-6)
    assert m3.shape == (17, 2, 64) and hp3.shape == (17, 2, 4), (m3.shape, hp3.shape)
    print("[3d OK ] shapes", tuple(m3.shape), tuple(hp3.shape))
    print("ALL_OK" if ok else "SOME_FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
