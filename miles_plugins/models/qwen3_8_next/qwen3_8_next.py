"""Qwen3.8-Next transformer block spec for Megatron.

Usage: ``--spec miles_plugins.models.qwen3_8_next.qwen3_8_next get_qwen3_8_next_spec``

Built on ``get_qwen3_5_spec``'s approach, because sglang's model is literally
``Qwen4ExpModel(Qwen3_5ForCausalLM)``: a **uniform GPT decoder block** where the
``linear_attention`` layers get their ``self_attention`` submodule swapped for a
gated-delta-net wrapper. Notably *not* ``megatron.core.models.hybrid``, whose
``HyperConnectionHybridLayer`` hardcodes ``HyperConnectionModule`` and would give
us no way in.

Three edits on top of what Qwen3.5 builds:

1. ``config.enable_hyper_connections`` (set by the bridge) already makes
   ``get_gpt_decoder_block_spec`` emit ``HyperConnectionTransformerLayer`` and
   populate the two HC ModuleSpec slots with ``HyperConnectionModule``. Swap both
   for ``Qwen38NextHyperConnection``, whose gating is the low-rank per-feature
   read gate and identity residual mixing that this model actually uses.

2. Drop every block layernorm. Qwen3.8-Next's checkpoint has **zero**
   ``input_layernorm`` and **zero** ``post_attention_layernorm`` tensors, and no
   final norm either -- each HC's ``hc_norm`` is the pre-block norm, and the
   final mixer's is the final norm. Qwen3.5 leaves TE's fused
   ``LayerNormColumnParallelLinear`` in ``linear_qkv`` (which is where its
   ``linear_qkv.layer_norm_weight`` comes from) and gives MoE layers a real
   ``pre_mlp_layernorm``; both would end up with no source tensor to load, and a
   layernorm sitting at its init value in front of an already-normed input is a
   silent correctness bug rather than a load error.

3. Fill ``TransformerBlockSubmodules.hc_head_contraction`` with
   ``Qwen38NextHCHeadContraction``. ``TransformerBlock``'s built-in contraction is
   DeepSeek-V4's -- one projection to a per-stream scalar, a sum over streams, and
   a single RMS across the whole ``n*C`` vector -- whereas Qwen3.8-Next's is the
   same low-rank gated mean as its per-layer HC with a per-stream RMS. That slot
   is new (radixark/Megatron-LM, "mhc: allow a model to supply its own output
   contraction") and is opt-in, so leaving it unset keeps DeepSeek-V4 on the old
   path with its parameter names intact.

Everything runs on Transformer Engine (``use_transformer_engine=True``), so the
attention and MLP linears stay TE modules and fp8 remains reachable.
"""

import copy

from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from transformers import AutoConfig

from miles_plugins.models.qwen3_5 import Attention as Qwen35LinearAttention
from miles_plugins.models.qwen3_5 import _get_text_config
from miles_plugins.models.qwen3_8_next.hyper_connection import (
    Qwen38NextHCHeadContraction,
    Qwen38NextHyperConnection,
)


def _layer_types(text_config):
    """Per-layer ``linear_attention`` / ``full_attention`` labels.

    Mirrors Qwen3.5's fallback: some config classes do not expose ``layer_types``,
    in which case every ``full_attention_interval``-th layer is full attention.
    For Qwen3.8-Flash-Next the released config does expose it, and it agrees --
    48 layers, 36 linear + 12 full.
    """
    if hasattr(text_config, "layer_types") and text_config.layer_types:
        return list(text_config.layer_types)
    interval = getattr(text_config, "full_attention_interval", 4)
    n = text_config.num_hidden_layers
    return ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(n)]


def _hc_spec(config):
    return ModuleSpec(module=Qwen38NextHyperConnection)


def _strip_block_layernorms(layer_spec, config):
    """Replace the fused-layernorm qkv with a plain linear, and drop pre_mlp_layernorm.

    ``backend.column_parallel_layer_norm_linear()`` is what Qwen3.5 puts in
    ``linear_qkv``; swapping it for ``TEColumnParallelLinear`` removes the
    ``layer_norm_weight`` parameter without giving up Transformer Engine.
    """
    submodules = layer_spec.submodules
    attn = submodules.self_attention
    if getattr(attn, "submodules", None) is not None and hasattr(attn.submodules, "linear_qkv"):
        attn.submodules.linear_qkv = TEColumnParallelLinear
    submodules.input_layernorm = IdentityOp
    submodules.pre_mlp_layernorm = IdentityOp


def get_qwen3_8_next_spec(args, config, vp_stage=None):
    """Transformer block spec for Qwen3.8-Next."""
    assert config.enable_hyper_connections, (
        "Qwen3.8-Next needs enable_hyper_connections=True; the bridge's _build_config "
        "sets it, so this means the model was not loaded through Qwen38NextBridge."
    )
    assert getattr(config, "qwen3_8_next_hc_lowrank", None), (
        "config.qwen3_8_next_hc_lowrank is unset; load through Qwen38NextBridge."
    )

    # Always take the MoE path for MoE checkpoints, matching Qwen3.5.
    if not args.num_experts:
        config.moe_layer_freq = [0] * config.num_layers

    kwargs = {"use_transformer_engine": True}
    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage
    transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)

    assert config.pipeline_model_parallel_layout is None, "not support this at the moment"

    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)

    hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    text_config = _get_text_config(hf_config)
    layer_types = _layer_types(text_config)

    for layer_id in range(num_layers_to_build):
        global_layer_id = layer_id + offset
        layer_spec = copy.deepcopy(transformer_layer_spec.layer_specs[layer_id])

        # Qwen3.8-Next's gating, in the slots enable_hyper_connections opened.
        layer_spec.submodules.self_attention_hyper_connection = _hc_spec(config)
        layer_spec.submodules.mlp_hyper_connection = _hc_spec(config)

        if layer_types[global_layer_id] == "linear_attention":
            layer_spec.submodules.self_attention = ModuleSpec(
                module=Qwen35LinearAttention,
                params={"args": args},
            )

        _strip_block_layernorms(layer_spec, config)
        transformer_layer_spec.layer_specs[layer_id] = layer_spec

    transformer_layer_spec.hc_head_contraction = ModuleSpec(module=Qwen38NextHCHeadContraction)

    # No final norm either: the checkpoint has no model.language_model.norm.weight,
    # because the contraction's own hc_norm is it. IdentityOp is deliberate rather
    # than post_layer_norm=False -- has_final_layernorm_in_this_stage() gates the
    # contraction on `submodules.layer_norm and post_process and post_layer_norm`,
    # so switching the norm off would switch the contraction off with it. IdentityOp
    # keeps the gate truthy while allocating nothing.
    transformer_layer_spec.layer_norm = IdentityOp

    return transformer_layer_spec
