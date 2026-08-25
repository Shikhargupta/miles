"""Backend-neutral API for streaming training-side weights as HF-named tensors.

Every weight consumer (colocated IPC, NCCL broadcast, P2P/RDT direct writes,
disk-delta, HF export) drives one of these iterators; every training backend
(megatron raw/bridge today, FSDP-family next) implements one. The API speaks HF
names and ``WeightUpdatePlacement`` only — no backend types cross it.
"""

import dataclasses
from abc import ABC, abstractmethod
from argparse import Namespace
from collections.abc import Iterator, Mapping
from typing import ClassVar

import torch

from miles.utils.lora import is_lora_weight_name


@dataclasses.dataclass(frozen=True)
class WeightUpdatePlacement:
    """Which training-side parallel dims the iterator gathers before yielding.

    A gathered dim: every yielded tensor is full along that dim, identically on
    every rank of that dim's group. A non-gathered dim: each rank yields its own
    shard of the param set (e.g. its PP slice). Backends ignore dims they don't
    have (an FSDP-family backend treats every placement as fully gathered).
    """

    gather_pp: bool
    # Always gathered today; explicit so a future protocol can relax them.
    gather_tp: bool = True
    gather_ep: bool = True


def resolve_placement(required: WeightUpdatePlacement, forced: WeightUpdatePlacement | None) -> WeightUpdatePlacement:
    """Join of the protocol's required placement and the iterator's forced one:
    a dim is gathered if either side gathers it. Protocols must accept a
    placement that gathers more dims than they asked for and derive their
    source-rank set from the resolved placement, not from their own assumption.
    """
    if forced is None:
        return required
    return WeightUpdatePlacement(
        gather_pp=required.gather_pp or forced.gather_pp,
        gather_tp=required.gather_tp or forced.gather_tp,
        gather_ep=required.gather_ep or forced.gather_ep,
    )


class HfWeightIteratorBase(ABC):
    """Mental model: ``training_model.to_hf(placement).named_parameters()``,
    streamed as size-bounded buckets.

    Collective contract: every training rank must drive the returned iterators
    to exhaustion in lockstep (collectives run inside ``next()``). Bucket
    structure is identical across ranks that share the same coordinates along
    all non-gathered dims. Yielded tensors are freshly allocated per bucket and
    stay valid for as long as the caller holds a reference.
    """

    # Placement this implementation can produce; None means any.
    forced_placement: ClassVar[WeightUpdatePlacement | None] = None

    def __init__(
        self,
        args: Namespace,
        model,
        *,
        placement: WeightUpdatePlacement,
        model_name: str,
        quantization_config: dict | None,
    ) -> None:
        self.args = args
        self.model = model
        self.placement = placement
        self.model_name = model_name
        self.quantization_config = quantization_config

    @abstractmethod
    def iter_hf_base_weights(
        self,
        weights: Mapping[str, torch.Tensor] | None,
        *,
        materialize: bool = True,
    ) -> Iterator[list[tuple[str, torch.Tensor]]]:
        """Base model weights as HF-named GPU tensors, one size-bounded bucket
        per ``next()``; atomic update groups are never split across buckets.

        ``weights``: backend-native named weights to read (e.g. a snapshot from
        the weights backuper); None reads the live model parameters.
        ``materialize=False`` joins every collective but skips conversion and
        yields nothing — for ranks the transfer protocol never reads from.
        """

    def get_hf_lora_weights(self, adapter=None) -> list[tuple[str, torch.Tensor]]:
        """The complete adapter in HF PEFT naming (lora_A/lora_B), as one list.

        Always fully gathered regardless of ``self.placement`` — adapters are
        small and engines load the whole adapter in one call. ``adapter=None``
        exports the single-LoRA adapter; otherwise the multi-LoRA adapter to
        export. Collective: every rank must call this together.
        """
        named_tensors = self._export_lora_named_tensors(adapter)
        if not named_tensors:
            raise RuntimeError(
                f"LoRA weight sync failed: the weight iterator produced zero chunks"
                f"{f' for adapter {adapter!r}' if adapter is not None else ''}. "
                "No adapter weights were sent to the rollout engine. This usually means "
                "the Megatron-Bridge or SGLang version is incompatible."
            )
        if not any(is_lora_weight_name(name) for name, _tensor in named_tensors):
            raise RuntimeError(
                "LoRA weight sync failed: chunk contains no LoRA weights "
                "(no lora_A/lora_B names found). Check weight iterator configuration."
            )
        return named_tensors

    @abstractmethod
    def _export_lora_named_tensors(self, adapter) -> list[tuple[str, torch.Tensor]]:
        """Backend hook: the complete adapter as HF-named tensors, fully
        gathered along every parallel dim — the backend owns its parallel
        groups, so any PP assembly happens inside the hook."""
