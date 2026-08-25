"""Backend-neutral API for streaming training-side weights as HF-named tensors.

Every weight consumer (colocated IPC, NCCL broadcast, P2P/RDT direct writes,
disk-delta, HF export) drives one of these iterators; every training backend
(megatron raw/bridge today, FSDP-family next) implements one. The API speaks HF
names and ``WeightUpdatePlacement`` only — no backend types cross it.
"""

import dataclasses
import math
from abc import ABC, abstractmethod
from argparse import Namespace
from collections.abc import Iterator, Mapping, Sequence
from typing import ClassVar

import torch
import torch.distributed as dist

from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.lora import is_lora_weight_name


@dataclasses.dataclass(frozen=True)
class WeightUpdatePlacement:
    """Which training-side parallel dims the iterator gathers before yielding.

    A gathered dim: every yielded tensor is full along that dim, identically on
    every rank of that dim's group. A non-gathered dim: each rank yields its own
    shard of the param set (e.g. its PP slice). Backends ignore dims they don't
    have (an FSDP-family backend treats every placement as FULL).
    """

    gather_pp: bool
    # Always gathered today; explicit so a future protocol can relax them.
    gather_tp: bool = True
    gather_ep: bool = True

    FULL: ClassVar["WeightUpdatePlacement"]
    KEEP_PP: ClassVar["WeightUpdatePlacement"]


WeightUpdatePlacement.FULL = WeightUpdatePlacement(gather_pp=True)
WeightUpdatePlacement.KEEP_PP = WeightUpdatePlacement(gather_pp=False)


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
        small and engines load the whole adapter in one call, so PP assembly
        happens here rather than at call sites. ``adapter=None`` exports the
        single-LoRA adapter; otherwise the multi-LoRA adapter to export.
        Collective: every rank must call this together.
        """
        named_tensors = self._export_lora_named_tensors(adapter)
        if not named_tensors:
            raise RuntimeError(
                f"LoRA weight sync failed: the weight iterator produced zero chunks"
                f"{f' for adapter {adapter!r}' if adapter is not None else ''}. "
                "No adapter weights were sent to the rollout engine. This usually means "
                "the Megatron-Bridge or SGLang version is incompatible."
            )
        named_tensors = _assemble_pp_full_adapter(named_tensors)
        if not any(is_lora_weight_name(name) for name, _tensor in named_tensors):
            raise RuntimeError(
                "LoRA weight sync failed: chunk contains no LoRA weights "
                "(no lora_A/lora_B names found). Check weight iterator configuration."
            )
        return named_tensors

    @abstractmethod
    def _export_lora_named_tensors(self, adapter) -> list[tuple[str, torch.Tensor]]:
        """Backend hook: the adapter's HF-named tensors, TP/EP gathered;
        PP-local is fine (``get_hf_lora_weights`` assembles PP)."""


def _assemble_pp_full_adapter(
    hf_named_tensors: Sequence[tuple[str, torch.Tensor]],
) -> list[tuple[str, torch.Tensor]]:
    """Assemble the complete adapter on every PP rank (backend exporters gather
    TP/EP but not PP)."""
    pp = get_parallel_state().pp
    if pp.size == 1:
        return list(hf_named_tensors)
    pp_rank = pp.rank
    global_ranks = dist.get_process_group_ranks(pp.group)
    device = torch.cuda.current_device()

    local_meta = [(n, tuple(t.shape), t.dtype) for n, t in hf_named_tensors]
    all_meta: list = [None] * pp.size
    dist.all_gather_object(all_meta, local_meta, group=pp.group)

    local_by_name = {n: t for n, t in hf_named_tensors}
    merged: dict[str, torch.Tensor] = {}
    for src_pp, meta in enumerate(all_meta):
        by_dtype: dict = {}
        for n, shape, dtype in meta:
            by_dtype.setdefault(dtype, []).append((n, shape))
        for dtype, entries in by_dtype.items():
            numel = sum(math.prod(shape) for _, shape in entries)
            flat = torch.empty(numel, dtype=dtype, device=device)
            if src_pp == pp_rank:
                off = 0
                for n, shape in entries:
                    k = math.prod(shape)
                    flat[off : off + k].copy_(local_by_name[n].reshape(-1))
                    off += k
            dist.broadcast(flat, src=global_ranks[src_pp], group=pp.group)
            off = 0
            for n, shape in entries:
                k = math.prod(shape)
                merged[n] = flat[off : off + k].view(shape)
                off += k
    return sorted(merged.items())
