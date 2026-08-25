"""Qwen3.8-Next (Qwen4Exp) hyper-connection math -- pure torch, no Megatron.

Kept free of Megatron imports on purpose: it lets these functions be unit-tested
against sglang directly, without dragging in megatron.core (whose import chain
pulls Transformer Engine).

The authority for every formula is sglang
``python/sglang/srt/layers/hyperconnection.py``: ``GroupedGemmaRMSNorm`` plus
``GatedResidual._mix_compute`` / ``_combine_compute``. sglang uses that one class
for the attention HC, the MLP HC, and the model-level final mixer, so this one
set of functions covers all three sites.

**Precision policy.** Every reduction and every elementwise step runs in fp32
(fp64 if the caller is in fp64), and only the returned tensor is cast back to
the input dtype. That matters for two reasons:

  * the hidden state is bf16, so the output rounding is an irreducible ~0.5 ulp
    floor -- but everything *before* it is avoidable error, and in bf16 the
    low-rank projections plus the gate multiply were contributing more than the
    floor itself;
  * the target is train/inference consistency, and sglang's JIT kernels already
    accumulate in fp32. Staying in bf16 here would leave us further from the
    exact answer than sglang is, so the train-vs-infer gap would be the *sum*
    of two errors instead of just sglang's.

The extra cost is small: the projections are ``n*C <-> lowrank`` (10240 <-> 320),
tiny next to attention and the MoE MLP.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def _acc_dtype(x: Tensor) -> torch.dtype:
    """Accumulate in fp32, but never downcast an fp64 caller (gradcheck)."""
    return x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32


def grouped_gemma_rmsnorm(x: Tensor, weight: Tensor, n: int, eps: float) -> Tensor:
    """Per-stream RMSNorm over ``[..., n*C]``, Gemma-style scale.

    Two details that are easy to get wrong and both change the numbers:

    * the variance is reduced **within each stream** (``group_size=hidden``),
      not over the whole ``n*C`` vector -- the latter is what Megatron's
      ``learned_output_contract`` does, so the two are not interchangeable;
    * the learned scale enters as ``1 + weight``, because the checkpoint stores
      it as a delta from unity (initialised to zeros). Feeding it to a plain
      Megatron RMSNorm, which multiplies by ``weight`` directly, silently
      scales every stream by ~0.

    Returns the normed tensor in the accumulation dtype, not the input dtype:
    callers feed it straight into the gate projections, and rounding it to bf16
    in between is pure loss. Cast at the end of the chain instead.
    """
    acc = _acc_dtype(x)
    x_grouped = x.to(acc).unflatten(-1, (n, x.shape[-1] // n))
    variance = x_grouped.pow(2).mean(dim=-1, keepdim=True)
    x_norm = (x_grouped * torch.rsqrt(variance + eps)).flatten(-2)
    return x_norm * (1.0 + weight.to(acc))


def hc_mix(
    normed: Tensor, w_down: Tensor, w_up: Tensor, n: int, hidden: int, out_dtype: torch.dtype
) -> Tensor:
    """Low-rank gated read: ``[..., n*C] -> [..., C]``.

    ``mean_c sigmoid(W_up SiLU(W_down N / n))_c * N_c``.

    The ``/ n`` sits between the down projection and the SiLU; moving it past
    the nonlinearity changes the result. The gate multiplies the *normed*
    streams, not the raw residual. And the reduction is a mean over streams --
    Megatron's mHC sums instead.
    """
    acc = normed.dtype
    gate = F.silu(F.linear(normed, w_down.to(acc)) / n)
    gate = torch.sigmoid(F.linear(gate, w_up.to(acc)))
    mixed = (gate.unflatten(-1, (n, hidden)) * normed.unflatten(-1, (n, hidden))).mean(dim=-2)
    return mixed.to(out_dtype)


def hc_inject_gate(normed: Tensor, w_inject: Tensor, n: int) -> Tensor:
    """Per-stream write gate ``a = 2 * sigmoid(W_inject N / n)`` -> ``[..., n]``.

    Left in the accumulation dtype; ``hc_combine`` consumes it directly.
    """
    acc = normed.dtype
    return 2 * torch.sigmoid(F.linear(normed, w_inject.to(acc)) / n)


def hc_combine(
    residual: Tensor, block_output: Tensor, h_post: Tensor, n: int, hidden: int
) -> Tensor:
    """``X'_c = X_c + a_c * y``, flattened back to ``[..., n*C]``.

    Qwen3.8-Next has no residual mixing matrix, so unlike Megatron's mHC there
    is no ``bmm(h_res^T, residual)`` term to compute -- the identity collapses
    it to the residual itself.

    The product and the add happen in the accumulation dtype with a single cast
    at the end, so the only rounding is the unavoidable one onto the bf16 hidden
    state. Doing it in bf16 instead rounds twice and lands ~2x further out.
    """
    out_dtype = residual.dtype
    acc = _acc_dtype(h_post) if h_post.dtype not in (torch.float32, torch.float64) else h_post.dtype
    R = residual.to(acc).unflatten(-1, (n, hidden))
    injection = block_output.to(acc).unsqueeze(-2) * h_post.to(acc).unsqueeze(-1)
    return (R + injection).flatten(-2).to(out_dtype)
