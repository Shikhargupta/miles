import torch
from mbridge.models.ext.deepseek_v3 import dequant_fp8_safetensor_io


def _patch_fp8_safe_open_device() -> None:
    """Load FP8 shards on each distributed rank's current CUDA device.

    mbridge's FP8 safetensor loader passes the unqualified device string
    ``cuda``.  safetensors resolves that to cuda:0 even after a worker selects
    a different local device, then the Triton dequantizer rejects the foreign
    pointer on ranks 1+.  Keep the upstream loader intact and qualify only that
    ambiguous CUDA argument at the Miles integration boundary.
    """

    safe_open = dequant_fp8_safetensor_io.safe_open
    if getattr(safe_open, "_miles_current_cuda_device", False):
        return

    def safe_open_on_current_device(*args, **kwargs):
        if kwargs.get("device") == "cuda" and torch.cuda.is_available():
            kwargs["device"] = f"cuda:{torch.cuda.current_device()}"
        return safe_open(*args, **kwargs)

    safe_open_on_current_device._miles_current_cuda_device = True
    dequant_fp8_safetensor_io.safe_open = safe_open_on_current_device


_patch_fp8_safe_open_device()

from .deepseek_v32 import DeepseekV32Bridge
from .deepseekv4 import DeepseekV4Bridge
from .glm4 import GLM4Bridge
from .glm4moe import GLM4MoEBridge
from .glm4moe_lite import GLM4MoELiteBridge
from .glm5_next import Glm5NextBridge
from .inkling import InklingBridge
from .joyai_llm_flash import JoyAILLMFlashBridge
from .mimo import MimoBridge
from .qwen3_5 import Qwen3_5Bridge
from .qwen3_next import Qwen3NextBridge

__all__ = [
    "GLM4Bridge",
    "GLM4MoEBridge",
    "GLM4MoELiteBridge",
    "Qwen3NextBridge",
    "Qwen3_5Bridge",
    "MimoBridge",
    "DeepseekV32Bridge",
    "Glm5NextBridge",
    "DeepseekV4Bridge",
    "JoyAILLMFlashBridge",
    "InklingBridge",
]
