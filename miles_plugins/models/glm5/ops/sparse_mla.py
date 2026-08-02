import logging
import os

import torch

from .tilelang_sparse_mla_bwd import sparse_mla_bwd
from .tilelang_sparse_mla_fwd import sparse_mla_fwd_interface

logger = logging.getLogger(__name__)

# A non-finite adapter gradient is reported as a bare ``train/grad_norm = nan`` next to a
# perfectly finite loss, which does not say whether this kernel produced the NaN or merely
# received one. Each check is a device-side reduce plus a sync, so spend a small fixed
# budget: once it is gone the guard short-circuits and the hot path is untouched.
_probe_budget = 3

# The kernels are provably clean on random data at the exact production shapes (H=8,
# topk=2048, ragged S, -1 padding, all--1 rows), so whatever breaks them is a property of
# the real activations. Persist one failing input set and the search moves offline, where
# an iteration costs seconds instead of a 40-minute rollout.
_dump_budget = 1
_DUMP_DIR = os.environ.get("DSA_NAN_DUMP_DIR", "/scratch/dsa_nan_dump")


def _census(**tensors: torch.Tensor) -> str:
    out = []
    for name, t in tensors.items():
        f = t.float()
        finite = torch.isfinite(f)
        absmax = f[finite].abs().max().item() if bool(finite.any()) else float("nan")
        out.append(
            f"{name}(nan={int(torch.isnan(f).sum())},inf={int(torch.isinf(f).sum())},"
            f"absmax={absmax:.3e},numel={t.numel()})"
        )
    return " ".join(out)


def _row_report(name: str, t: torch.Tensor, indices: torch.Tensor) -> str:
    """Locate non-finite rows and say whether they are the rows with no valid key.

    A NaN confined to rows the loss masks out would explain the central puzzle: a finite
    loss alongside a backward that scatters NaN into the shared dKV and so poisons every
    layer at once.
    """
    bad = ~torch.isfinite(t.float()).reshape(t.shape[0], -1).all(dim=1)
    n_bad = int(bad.sum())
    if n_bad == 0:
        return f"{name}: all rows finite"
    empty = (indices == -1).all(dim=-1).reshape(indices.shape[0], -1).all(dim=1)
    return (
        f"{name}: {n_bad}/{t.shape[0]} rows non-finite, "
        f"{int((bad & empty).sum())} of them empty-index, "
        f"{int((bad & ~empty).sum())} with >=1 valid key, "
        f"first_rows={torch.nonzero(bad).flatten()[:8].tolist()}"
    )


class SparseMLA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, indices, scaling):
        """
        Args:
            q: Query tensor (seq_len, heads, dim_plus_tail_dim)
            kv: Key-Value tensor (seq_len_kv, kv_group, dim_plus_tail_dim)
            indices: Sparse indices tensor (seq_len, kv_group, topk)

        Returns:
            out: Output tensor (seq_len, heads, dim)
        """
        global _probe_budget

        indices = indices.contiguous()
        q, kv = q.contiguous(), kv.contiguous()
        ctx.scaling = scaling
        tl_out, tl_lse = sparse_mla_fwd_interface(q, kv, indices, sm_scale=scaling)

        if _probe_budget > 0 and not (torch.isfinite(tl_out).all() and torch.isfinite(tl_lse).all()):
            _probe_budget -= 1
            logger.error(
                "DSA nan probe [forward]: %s | %s | %s | empty_index_rows=%d scaling=%s",
                _census(q=q, kv=kv, out=tl_out, lse=tl_lse),
                _row_report("out", tl_out, indices),
                _row_report("lse", tl_lse, indices),
                int((indices == -1).all(dim=-1).sum()),
                scaling,
            )

        # Save tensors for backward pass
        ctx.save_for_backward(q, kv, indices, tl_out, tl_lse)

        return tl_out, tl_lse

    @staticmethod
    def backward(ctx, grad_output, grad_lse):
        """
        Args:
            grad_output: Gradient of the loss with respect to output

        Returns:
            Gradients for q, kv, and indices (None for indices)
        """
        global _probe_budget, _dump_budget

        q, kv, indices, tl_out, tl_lse = ctx.saved_tensors
        scaling = ctx.scaling
        do = grad_output.contiguous()

        tl_dq, tl_dkv = sparse_mla_bwd(q, kv, tl_out, do, indices, tl_lse, sm_scale=scaling)

        if _probe_budget > 0 and not (torch.isfinite(tl_dq).all() and torch.isfinite(tl_dkv).all()):
            _probe_budget -= 1
            delta = (tl_out.float() * do.float()).sum(-1)
            logger.error(
                "DSA nan probe [backward]: %s | %s | %s | %s | empty_index_rows=%d "
                "lse_min=%s idx_min=%d idx_max=%d kv_len=%d n_out_of_range=%d scaling=%s",
                _census(q=q, kv=kv, out=tl_out, lse=tl_lse, do=do, delta=delta, dq=tl_dq, dkv=tl_dkv),
                _row_report("out", tl_out, indices),
                _row_report("do", do, indices),
                _row_report("dq", tl_dq, indices),
                int((indices == -1).all(dim=-1).sum()),
                float(tl_lse.min()),
                int(indices.min()),
                int(indices.max()),
                kv.shape[0],
                int(((indices != -1) & (indices >= kv.shape[0])).sum()),
                scaling,
            )

            if _dump_budget > 0:
                _dump_budget -= 1
                os.makedirs(_DUMP_DIR, exist_ok=True)
                path = f"{_DUMP_DIR}/dsa_nan_rank{int(os.environ.get('RANK', -1))}.pt"
                torch.save(
                    {
                        "q": q.cpu(),
                        "kv": kv.cpu(),
                        "indices": indices.cpu(),
                        "out": tl_out.cpu(),
                        "lse": tl_lse.cpu(),
                        "do": do.cpu(),
                        "scaling": scaling,
                    },
                    path,
                )
                logger.error("DSA nan probe [backward]: dumped failing inputs to %s", path)

        # Return gradients for each input (None for indices as it's not differentiable)
        return tl_dq, tl_dkv, None, None
