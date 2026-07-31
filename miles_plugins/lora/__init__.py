"""Miles-native LoRA plugin.

The package implements the native-LoRA provider protocol directly, so
``--lora-provider-path miles_plugins.lora`` (the default) resolves here. The
pre-#2017 module path ``miles_plugins.lora.lora`` keeps working for explicit
pins. Core Miles utilities are imported lazily at call time, never at module
level, so importing this package stays cycle-free and light.
"""

from miles_plugins.lora.lora import (
    apply_native_lora,
    export_lora_hf_named,
    load_lora_adapter_hf,
    wrap_model_provider_with_lora,
)

__all__ = [
    "apply_native_lora",
    "export_lora_hf_named",
    "load_lora_adapter_hf",
    "wrap_model_provider_with_lora",
]
