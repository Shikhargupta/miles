"""Megatron-Bridge adapter support for GLM-5.3-Flash (``glm5_next``).

The upstream GLM bridge only handles the older ``GlmMoeDsaForCausalLM``.
GLM-5.3 is a composite conditional-generation model whose language model adds
KDA layers and mHC. Miles already implements the model and HF conversion in its
``mbridge`` plugin; this small bridge makes the same model buildable by the
Megatron-Bridge LoRA path and gives adapter export the nested HF parameter names
expected by SGLang.
"""

from __future__ import annotations

from types import SimpleNamespace


def _text_config(hf_pretrained):
    config = hf_pretrained.config
    return getattr(config, "text_config", None) or config


def _nest_hf_param(value):
    def nest(name: str) -> str:
        if name.startswith("model.") and not name.startswith("model.language_model."):
            return "model.language_model." + name[len("model.") :]
        return name

    if isinstance(value, str):
        return nest(value)
    if isinstance(value, dict):
        return {key: nest(name) for key, name in value.items()}
    return value


def _build_glm5_next_bridge():
    from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
    from megatron.bridge.models.glm5.glm5_bridge import GLM5Bridge
    from megatron.core.models.gpt import GPTModel

    @MegatronModelBridge.register_bridge(source="Glm5NextForConditionalGeneration", target=GPTModel)
    class MilesGLM5NextBridge(GLM5Bridge):
        """GLM-5.3 model construction plus LoRA adapter-name conversion."""

        def provider_bridge(self, hf_pretrained):
            text_config = _text_config(hf_pretrained)
            # GLM5Bridge consumes a language-model config rather than the outer
            # conditional-generation config. It only reads ``.config`` here.
            provider = super().provider_bridge(SimpleNamespace(config=text_config))

            from miles_plugins.models.glm5_next.glm5_next import _apply_glm5_next_config

            _apply_glm5_next_config(provider, text_config)
            # GLM-5.3 encodes ``head_dim=0`` to denote NoPE, while the actual
            # attention channel width lives in qk_nope_head_dim (256). The
            # regular mbridge launcher supplies this as --kv-channels; mirror
            # that resolution here so TE never receives a zero head width.
            provider.kv_channels = int(text_config.qk_nope_head_dim)
            # The parent GLM-5 bridge selects Megatron's global experimental
            # DSA dispatcher. GLM-5.3 is hybrid KDA/DSA and its custom spec
            # replaces attention per layer, so the global dispatcher must stay
            # disabled or get_gpt_decoder_block_spec rejects the configuration.
            provider.experimental_attention_variant = None

            def glm5_next_layer_spec(config, vp_stage=None):
                from megatron.training.global_vars import get_args

                from miles_plugins.models.glm5_next.glm5_next import get_glm5_next_spec

                return get_glm5_next_spec(get_args(), config, vp_stage=vp_stage)

            provider.transformer_layer_spec = glm5_next_layer_spec
            # GLM-5.3's published checkpoint has no trainable MTP head in this
            # RL recipe. Avoid constructing the older GLM bridge's MTP path.
            provider.mtp_num_layers = None
            return provider

        def mapping_registry(self):
            registry = super().mapping_registry()
            mappings = registry.mappings if hasattr(registry, "mappings") else registry._mappings
            # The language model is nested under model.language_model in GLM-5.3.
            # LoRA synchronization only streams adapter tensors, but nesting all
            # inherited mappings keeps naming internally consistent.
            for mapping in mappings:
                mapping.hf_param = _nest_hf_param(mapping.hf_param)
            return registry

    return MilesGLM5NextBridge


_build_glm5_next_bridge()
