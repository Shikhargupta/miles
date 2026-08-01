"""Compatibility shim for the pre-plugin native-LoRA provider path.

Native LoRA moved to :mod:`miles_plugins.lora` in #2017.  Keep the old dotted
provider path importable so existing launch scripts and saved configurations do
not fail at startup; all behavior lives in the plugin implementation.
"""

from miles_plugins.lora import (
    apply_native_lora,
    export_lora_hf_named,
    export_lora_sglang_named,
    load_lora_adapter_hf,
    wrap_model_provider_with_lora,
)

__all__ = [
    "apply_native_lora",
    "export_lora_hf_named",
    "export_lora_sglang_named",
    "load_lora_adapter_hf",
    "wrap_model_provider_with_lora",
]
