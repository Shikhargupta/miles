"""Numerical validation: triton QSA sparse attention vs the torch reference.

Runs fwd + bwd on random data across shapes that cover GQA grouping, -1 padding,
odd sequence lengths, and selections with duplicated coverage patterns. The torch
reference is the mask-based path that the fwd parity runs verified against sglang.
"""

import sys

import torch

sys.path.insert(0, "/data/home/zzeng/repos/miles-qwen4exp")

from miles_plugins.models.qwen3_8_next.ops.kernel.qsa_sparse_attn import (
    qsa_sparse_attention_triton,
)


def torch_reference(q, k, v, indices, scale):
    T, Hq, D = q.shape
    S, Hkv, _ = k.shape
    rep = Hq // Hkv
    mask = torch.zeros(T, S, dtype=torch.bool, device=q.device)
    valid = indices >= 0
    rows = torch.arange(T, device=q.device).unsqueeze(-1).expand_as(indices)
    mask[rows[valid], indices[valid].long()] = True
    qh = q.transpose(0, 1).float()
    kh = k.transpose(0, 1).repeat_interleave(rep, dim=0).float()
    vh = v.transpose(0, 1).repeat_interleave(rep, dim=0).float()
    scores = torch.einsum("htd,hsd->hts", qh, kh) * scale
    scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    p = torch.softmax(scores, dim=-1)
    p = torch.nan_to_num(p, 0.0)  # rows with no valid index
    return torch.einsum("hts,hsd->htd", p, vh).transpose(0, 1)


def run_case(T, S, Hq, Hkv, D, K, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(T, Hq, D, device="cuda", dtype=dtype, generator=g, requires_grad=True)
    k = torch.randn(S, Hkv, D, device="cuda", dtype=dtype, generator=g, requires_grad=True)
    v = torch.randn(S, Hkv, D, device="cuda", dtype=dtype, generator=g, requires_grad=True)
    # Unique indices per row: production selections are unique by construction
    # (top-k block expansion is disjoint from the incomplete tail block), and the
    # kernel is list-semantics -- a duplicated index would be counted twice, which
    # the mask-based reference cannot represent.
    scores_rand = torch.rand(T, S, device="cuda", generator=g)
    idx = scores_rand.topk(K, dim=-1).indices.to(torch.int32)
    pos = torch.arange(T, device="cuda").unsqueeze(1)
    keep = torch.rand(T, K, device="cuda", generator=g) > 0.3
    keep[:, 0] = True
    idx = torch.where(keep, idx, torch.full_like(idx, -1))

    scale = D ** -0.5
    out_t = qsa_sparse_attention_triton(q, k, v, idx, scale)
    gout = torch.randn_like(out_t)
    out_t.backward(gout)
    gq_t, gk_t, gv_t = q.grad.clone(), k.grad.clone(), v.grad.clone()
    q.grad = k.grad = v.grad = None

    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    out_r = torch_reference(q2, k2, v2, idx, scale).to(dtype)
    out_r.backward(gout)

    def rel(a, b):
        return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()

    r_o = rel(out_t, out_r)
    r_q = rel(gq_t, q2.grad)
    r_k = rel(gk_t, k2.grad)
    r_v = rel(gv_t, v2.grad)
    tol = 2e-2 if dtype == torch.bfloat16 else 2e-4
    ok = all(r < tol for r in (r_o, r_q, r_k, r_v))
    print(f"T={T} S={S} Hq={Hq}/{Hkv} D={D} K={K} {str(dtype):14} "
          f"out={r_o:.2e} dq={r_q:.2e} dk={r_k:.2e} dv={r_v:.2e} {'OK' if ok else 'FAIL'}")
    return ok


def main():
    cases = [
        (128, 128, 4, 2, 64, 32, torch.float32, 0),
        (257, 257, 6, 2, 128, 64, torch.float32, 1),
        (515, 515, 24, 2, 128, 96, torch.bfloat16, 2),
        (1024, 1024, 24, 2, 128, 512, torch.bfloat16, 3),
        (700, 700, 8, 8, 64, 50, torch.float32, 4),  # no GQA grouping
    ]
    results = [run_case(*c) for c in cases]
    print("ALL_OK" if all(results) else "SOME_FAILED")


if __name__ == "__main__":
    main()
