import logging

import torch

from .tilelang_sparse_mla_bwd import sparse_mla_bwd
from .tilelang_sparse_mla_fwd import sparse_mla_fwd_interface

logger = logging.getLogger(__name__)

# A non-finite adapter gradient is reported as a bare ``train/grad_norm = nan`` next to a
# perfectly finite loss, which does not say whether this kernel produced the NaN or merely
# received one. Each check is a device-side reduce plus a sync, so spend a small fixed
# budget: once it is gone the guard short-circuits and the hot path is untouched.
_probe_budget = 3


def _census(**tensors: torch.Tensor) -> str:
    return " ".join(
        f"{name}(nan={int(torch.isnan(t).sum())},inf={int(torch.isinf(t).sum())},numel={t.numel()})"
        for name, t in tensors.items()
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

        if _probe_budget > 0 and not torch.isfinite(tl_out).all():
            _probe_budget -= 1
            logger.error(
                "DSA nan probe [forward]: %s empty_index_rows=%d",
                _census(q=q, kv=kv, out=tl_out, lse=tl_lse),
                int((indices == -1).all(dim=-1).sum()),
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
        global _probe_budget

        q, kv, indices, tl_out, tl_lse = ctx.saved_tensors
        scaling = ctx.scaling

        tl_dq, tl_dkv = sparse_mla_bwd(q, kv, tl_out, grad_output.contiguous(), indices, tl_lse, sm_scale=scaling)

        if _probe_budget > 0 and not (torch.isfinite(tl_dq).all() and torch.isfinite(tl_dkv).all()):
            _probe_budget -= 1
            logger.error(
                "DSA nan probe [backward]: %s empty_index_rows=%d lse_min=%s "
                "idx_min=%d idx_max=%d kv_len=%d n_out_of_range=%d",
                _census(q=q, kv=kv, out=tl_out, lse=tl_lse, do=grad_output, dq=tl_dq, dkv=tl_dkv),
                int((indices == -1).all(dim=-1).sum()),
                float(tl_lse.min()),
                int(indices.min()),
                int(indices.max()),
                kv.shape[0],
                int(((indices != -1) & (indices >= kv.shape[0])).sum()),
            )

        # Return gradients for each input (None for indices as it's not differentiable)
        return tl_dq, tl_dkv, None, None
