"""Unit tests for the backend-neutral HF weight iterator base.

Covers WeightUpdatePlacement / resolve_placement and the get_hf_lora_weights
template method (validation + PP assembly gating), which every backend shares.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])


from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from miles.backends.training_utils.hf_weight_iterator import (
    HfWeightIteratorBase,
    WeightUpdatePlacement,
    resolve_placement,
)

_MODULE = "miles.backends.training_utils.hf_weight_iterator"

SAMPLE_LORA_WEIGHTS = [
    ("model.layers.0.self_attn.q_proj.lora_A.weight", torch.randn(4, 2)),
    ("model.layers.0.self_attn.q_proj.lora_B.weight", torch.randn(2, 4)),
]

SAMPLE_BASE_ONLY_WEIGHTS = [
    ("model.layers.0.self_attn.q_proj.weight", torch.randn(4, 4)),
]


class TestWeightUpdatePlacement:
    def test_constants(self):
        assert WeightUpdatePlacement.FULL == WeightUpdatePlacement(gather_pp=True, gather_tp=True, gather_ep=True)
        assert WeightUpdatePlacement.KEEP_PP == WeightUpdatePlacement(gather_pp=False, gather_tp=True, gather_ep=True)

    def test_resolve_without_forced_returns_required(self):
        assert resolve_placement(WeightUpdatePlacement.KEEP_PP, None) == WeightUpdatePlacement.KEEP_PP

    def test_resolve_joins_gathered_dims(self):
        assert (
            resolve_placement(WeightUpdatePlacement.KEEP_PP, WeightUpdatePlacement.FULL) == WeightUpdatePlacement.FULL
        )
        assert (
            resolve_placement(WeightUpdatePlacement.FULL, WeightUpdatePlacement.KEEP_PP) == WeightUpdatePlacement.FULL
        )


class _StubIterator(HfWeightIteratorBase):
    """Concrete subclass with a canned adapter export."""

    def __init__(self, exported):
        super().__init__(
            Namespace(),
            [],
            placement=WeightUpdatePlacement.FULL,
            model_name="stub",
            quantization_config=None,
        )
        self._exported = exported
        self.export_calls = []

    def iter_hf_base_weights(self, weights, *, materialize=True):
        yield from []

    def _export_lora_named_tensors(self, adapter):
        self.export_calls.append(adapter)
        return self._exported


def _patch_pp_size_one():
    return patch(f"{_MODULE}.get_parallel_state", return_value=SimpleNamespace(pp=SimpleNamespace(size=1)))


class TestGetHfLoraWeightsTemplate:
    def test_returns_exported_adapter(self):
        iterator = _StubIterator(SAMPLE_LORA_WEIGHTS)
        with _patch_pp_size_one():
            assert iterator.get_hf_lora_weights() == SAMPLE_LORA_WEIGHTS
        assert iterator.export_calls == [None]

    def test_adapter_argument_reaches_the_hook(self):
        adapter = SimpleNamespace(name="slot0")
        iterator = _StubIterator(SAMPLE_LORA_WEIGHTS)
        with _patch_pp_size_one():
            iterator.get_hf_lora_weights(adapter)
        assert iterator.export_calls == [adapter]

    def test_raises_on_empty_export(self):
        iterator = _StubIterator([])
        with pytest.raises(RuntimeError, match="zero chunks"):
            iterator.get_hf_lora_weights()

    def test_empty_export_error_names_the_adapter(self):
        iterator = _StubIterator([])
        with pytest.raises(RuntimeError, match="slot0"):
            iterator.get_hf_lora_weights(SimpleNamespace(name="slot0"))

    def test_raises_when_export_has_no_lora_names(self):
        iterator = _StubIterator(SAMPLE_BASE_ONLY_WEIGHTS)
        with _patch_pp_size_one():
            with pytest.raises(RuntimeError, match="no LoRA weights"):
                iterator.get_hf_lora_weights()
