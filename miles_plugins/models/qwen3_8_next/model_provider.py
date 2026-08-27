"""Custom miles model provider for Qwen3.8-Flash-Next.

Delegates the entire model construction to miles' default provider (the spec,
config derivation, MoE wiring -- everything) and adds exactly one thing on top:
forward hooks that publish the PLE n-gram side channel. The training loop calls
``model(input_ids=..., packed_seq_params=...)`` with no model-specific plumbing,
and the PLE layer deliberately refuses to run without published ids, so someone
who can see both the input tokens and the model has to bridge the two -- this is
that someone. Wired up via ``--custom-model-provider-path``, miles' sanctioned
extension point, rather than a patch on the training loop.
"""

import logging

from megatron.core.models.gpt import GPTModel

from miles_plugins.models.qwen3_8_next.ops.backend import warm_kernels
from miles_plugins.models.qwen3_8_next.ops.ple_embedding import Qwen38NextFrozenNGramEmbedding
from miles_plugins.models.qwen3_8_next.ops.ple_context import (
    clear_ple_batch,
    publish_ple_batch,
)
from miles_plugins.models.qwen3_8_next.ops.ple_hash import build_ngram_contexts_packed

logger = logging.getLogger(__name__)


def _find_ple_embedding(model):
    for _, module in model.named_modules():
        if isinstance(module, Qwen38NextFrozenNGramEmbedding):
            return module
    return None


def _install_ple_context_hooks(model: GPTModel) -> None:
    ple_embedding = _find_ple_embedding(model)
    if ple_embedding is None:
        # This pipeline stage does not host the PLE layer; nothing to publish.
        return

    def pre_hook(_module, args, kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None:
            return
        packed = kwargs.get("packed_seq_params")
        cu_seqlens = getattr(packed, "cu_seqlens_q", None) if packed is not None else None
        flat = input_ids.reshape(-1)
        contexts = build_ngram_contexts_packed(
            flat, cu_seqlens, ple_embedding.ngram_size, ple_embedding.eos_token_id
        )
        ngram_ids = ple_embedding.compute_ngram_ids(contexts)
        publish_ple_batch(ngram_ids, cu_seqlens)

    def post_hook(_module, _args, _output):
        clear_ple_batch()

    model.register_forward_pre_hook(pre_hook, with_kwargs=True)
    model.register_forward_hook(post_hook)
    logger.info("PLE n-gram context hooks installed on the stage hosting the PLE layer")


def get_qwen3_8_next_model_provider(pre_process: bool = True, post_process: bool = True, vp_stage=None):
    from megatron.training import get_args

    from miles.backends.megatron_utils.model_provider import get_model_provider_func

    args = get_args()
    # Null the custom path while resolving the base provider, or we recurse into
    # ourselves; miles' wrapper re-reads it afterwards, so restore it.
    saved = args.custom_model_provider_path
    args.custom_model_provider_path = None
    try:
        base_provider = get_model_provider_func(args)
    finally:
        args.custom_model_provider_path = saved

    model = base_provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
    _install_ple_context_hooks(model)
    _warm_triton_kernels()
    return model


def _warm_triton_kernels() -> None:
    """Pre-compile every enabled triton kernel family, all ranks in parallel.

    Left to first use, bwd kernels JIT during the first BACKWARD -- inside the
    1F1B pipeline, where stages compile serially while the rest wait (runs
    32/33 spent 25-40 min there and were killed as presumed hangs). The warmup
    closures live next to the kernels they compile (ops/kernel/*, registered
    via ops.backend.register_warmup); this passes the model's real constexpr
    widths and gets out of the way. The node-local compile cache seeded by
    e2e_node.sh makes this near-instant after any prior run of a kernel version.
    """
    from megatron.training import get_args

    a = get_args()
    warm_kernels(
        hidden_size=a.hidden_size,
        hc_count=getattr(a, "num_residual_streams", 4),
        hc_lowrank=getattr(a, "qwen3_8_next_hc_lowrank", 320),
        ple_conv_kernel=getattr(a, "qwen3_8_next_ple_conv_kernel_size", 4),
        ple_conv_dilation=getattr(a, "qwen3_8_next_ple_conv_dilation", 3),
        qsa_head_dim=getattr(a, "kv_channels", 256),
        # The QSA kernels specialize on GROUP (= per-rank q heads / kv heads)
        # and D only, so per-rank counts must match production: heads are split
        # by TP.
        qsa_q_heads=a.num_attention_heads // a.tensor_model_parallel_size,
        qsa_kv_heads=max(1, getattr(a, "num_query_groups", 2) // a.tensor_model_parallel_size),
    )
    logger.info("triton kernel warmup complete (families per ops/backend.py)")
