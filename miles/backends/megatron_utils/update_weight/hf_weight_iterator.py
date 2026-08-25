"""Megatron implementations' shared base and factory for the backend-neutral
HF weight iterator API."""

from abc import abstractmethod
from argparse import Namespace
from collections.abc import Sequence

import torch

from miles.backends.training_utils.parallel import get_parallel_state
from miles.backends.training_utils.weight_update.atomic_groups import get_hf_atomic_update_groups
from miles.backends.training_utils.weight_update.gather import broadcast_from_owners
from miles.backends.training_utils.weight_update.hf_weight_iterator import (
    HfWeightIteratorBase,
    WeightUpdatePlacement,
    resolve_placement,
)


class MegatronHfWeightIteratorBase(HfWeightIteratorBase):
    forced_placement = WeightUpdatePlacement(gather_pp=True)

    def _hf_atomic_update_groups(self):
        return get_hf_atomic_update_groups(self.model_name, q_lora_rank=self.args.q_lora_rank)

    def _export_lora_named_tensors(self, adapter):
        # Both megatron exporters gather TP/EP but not PP.
        named_tensors = self._export_pp_local_lora(adapter)
        pp = get_parallel_state().pp
        if pp.size == 1:
            return named_tensors
        return broadcast_from_owners(named_tensors, pp.group)

    @abstractmethod
    def _export_pp_local_lora(self, adapter) -> list[tuple[str, torch.Tensor]]:
        """The adapter's HF-named tensors, TP/EP gathered, PP-local."""


def get_hf_weight_iterator(
    args: Namespace,
    model: Sequence[torch.nn.Module],
    *,
    required_placement: WeightUpdatePlacement,
    model_name: str,
    quantization_config: dict | None,
) -> HfWeightIteratorBase:
    # Local: the implementations subclass MegatronHfWeightIteratorBase from
    # this module, so importing them at the top would be a cycle.
    from miles.backends.megatron_utils.update_weight.hf_weight_iterator_bridge import HfWeightIteratorBridge
    from miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct import HfWeightIteratorDirect

    cls = {
        "raw": HfWeightIteratorDirect,
        "bridge": HfWeightIteratorBridge,
    }[args.megatron_to_hf_mode]

    return cls(
        args,
        model,
        placement=resolve_placement(required_placement, cls.forced_placement),
        model_name=model_name,
        quantization_config=quantization_config,
    )
