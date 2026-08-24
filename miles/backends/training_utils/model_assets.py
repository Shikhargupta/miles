"""Load the tokenizer/config assets every training backend needs.

Loading is serialized across ranks: the HF loaders write into a shared cache
directory, and concurrent writers corrupt it. Each backend had its own copy of
this rank-by-rank loop.
"""

import logging
from argparse import Namespace
from dataclasses import dataclass
from typing import Any

import torch.distributed as dist

from miles.utils.distributed_utils import get_gloo_group
from miles.utils.hf_config import load_hf_config
from miles.utils.processing_utils import load_processor, load_tokenizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelAssets:
    hf_config: Any
    tokenizer: Any
    processor: Any | None = None


def load_model_assets(args: Namespace, *, with_processor: bool = False) -> ModelAssets:
    """Load config/tokenizer (and optionally a multimodal processor) one rank at a time.

    ``with_processor`` is opt-in rather than inferred from the config so a backend
    that does not consume a processor never pays for loading one.
    """
    hf_config = tokenizer = processor = None
    for i in range(dist.get_world_size()):
        if i == dist.get_rank():
            hf_config = load_hf_config(args.hf_checkpoint)
            tokenizer = load_tokenizer(
                args.hf_checkpoint,
                chat_template_path=args.chat_template_path,
                trust_remote_code=True,
            )
            if with_processor and hasattr(hf_config, "vision_config"):
                processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        dist.barrier(group=get_gloo_group())

    return ModelAssets(hf_config=hf_config, tokenizer=tokenizer, processor=processor)
