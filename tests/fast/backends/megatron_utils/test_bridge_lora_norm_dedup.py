"""Grad-norm dedup for TP-duplicated grouped-expert adapter weights
(``_dedup_expert_adapter_norm_attrs``).

The bridge marks grouped-expert adapter weights tensor_model_parallel=True
unconditionally; at the only supported multi-LoRA MoE config (ETP=1, TP>1)
they are fully TP-duplicated, so the flag makes every TP rank contribute the
same gradient to the world-reduced per-slot norm — sqrt(TP)-inflated
grad_norm, under-scaled grad_clip_norm (measured exactly sqrt(2) at TP=2 on
the fixed bridge). The pre-wrap hook clears the flag on exactly those
weights so Megatron's stock filter counts each logical parameter once.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import pytest
import torch
import torch.nn as nn

pytest.importorskip("megatron.bridge.peft.multi_lora_layers")

import megatron.bridge.peft.multi_lora_layers as mll

from miles.backends.megatron_utils.bridge_lora_helpers import _dedup_expert_adapter_norm_attrs


class _StubGroupedExpert(nn.Module):
    """Stands in for MultiLoRAGroupedExpertLinear via monkeypatched isinstance
    target: carries the same .adapters[*].linear_in/out.weight surface."""

    def __init__(self, n_adapters: int = 2):
        super().__init__()
        self.adapters = nn.ModuleList()
        for _ in range(n_adapters):
            adapter = nn.Module()
            adapter.linear_in = nn.Module()
            adapter.linear_in.weight = nn.Parameter(torch.zeros(2, 2))
            adapter.linear_out = nn.Module()
            adapter.linear_out.weight = nn.Parameter(torch.zeros(2, 2))
            for weight in (adapter.linear_in.weight, adapter.linear_out.weight):
                weight.tensor_model_parallel = True  # the bridge's unconditional stamp
                weight.allreduce = False  # expert-bucket routing, must stay put
            self.adapters.append(adapter)


def _chunk_with_expert_and_attention():
    chunk = nn.Module()
    chunk.experts = _StubGroupedExpert()
    # A genuinely TP-sharded (attention) adapter param: must never be touched.
    chunk.attn = nn.Module()
    chunk.attn.weight = nn.Parameter(torch.zeros(2, 2))
    chunk.attn.weight.tensor_model_parallel = True
    return chunk


@pytest.fixture
def stub_grouped_class(monkeypatch):
    monkeypatch.setattr(mll, "MultiLoRAGroupedExpertLinear", _StubGroupedExpert)


def _expert_weights(chunk):
    for adapter in chunk.experts.adapters:
        yield adapter.linear_in.weight
        yield adapter.linear_out.weight


class TestDedupExpertAdapterNormAttrs:
    def test_tp_duplicated_expert_weights_are_cleared(self, stub_grouped_class):
        chunk = _chunk_with_expert_and_attention()
        out = _dedup_expert_adapter_norm_attrs([chunk], tensor_parallel_size=2, expert_tensor_parallel_size=1)
        assert out == [chunk]
        for weight in _expert_weights(chunk):
            assert weight.tensor_model_parallel is False, "TP-duplicated expert adapter must count once"
            assert weight.allreduce is False, "DDP expert-bucket routing must stay untouched"
        assert chunk.attn.weight.tensor_model_parallel is True, "sharded attention adapters keep the flag"

    def test_noop_when_tp_equals_expert_tp(self, stub_grouped_class):
        # TP=1 (or ETP==TP): the weights are NOT duplicated; the bridge's
        # attribute is consistent and must stay.
        chunk = _chunk_with_expert_and_attention()
        _dedup_expert_adapter_norm_attrs([chunk], tensor_parallel_size=1, expert_tensor_parallel_size=1)
        for weight in _expert_weights(chunk):
            assert weight.tensor_model_parallel is True

    def test_single_chunk_object_is_accepted(self, stub_grouped_class):
        chunk = _chunk_with_expert_and_attention()
        out = _dedup_expert_adapter_norm_attrs(chunk, tensor_parallel_size=2, expert_tensor_parallel_size=1)
        assert out is chunk
        assert all(w.tensor_model_parallel is False for w in _expert_weights(chunk))

    def test_none_expert_tp_defaults_to_one(self, stub_grouped_class):
        chunk = _chunk_with_expert_and_attention()
        _dedup_expert_adapter_norm_attrs([chunk], tensor_parallel_size=2, expert_tensor_parallel_size=None)
        assert all(w.tensor_model_parallel is False for w in _expert_weights(chunk))
