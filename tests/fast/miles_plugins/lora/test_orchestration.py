"""CPU tests for apply_native_lora's orchestration layer and its run guards.

These drive the real seams the layered split introduced — registry resolution,
config normalization, spec validation, and per-layer attach — on fake decoder
modules, with only megatron's parallel_state patched to a single rank.
"""

import json
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from miles_plugins.lora.config import LoRAConfig
from miles_plugins.lora.lora import _assert_supported_run, apply_native_lora
from miles_plugins.lora.modules.linear import iter_adapters
from miles_plugins.lora.spec.attention import MLA_ATTENTION_SPEC, MLA_TARGETS
from miles_plugins.lora.spec.base import AttachContext

ALL_GQA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class _FakeTELinear(nn.Module):
    """Mimics an MCore/TE linear: weight parameter, forward returning (out, bias)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        return F.linear(x, self.weight), None


class _FakeAttention(nn.Module):
    def __init__(self, hidden: int, num_q: int, num_kv: int, head_dim: int):
        super().__init__()
        self.linear_qkv = _FakeTELinear(hidden, (num_q + 2 * num_kv) * head_dim)
        self.linear_proj = _FakeTELinear(num_q * head_dim, hidden)
        self.num_attention_heads_per_partition = num_q
        self.num_query_groups_per_partition = num_kv
        self.hidden_size_per_attention_head = head_dim


class _FakeMLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.linear_fc1 = _FakeTELinear(hidden, 2 * intermediate)
        self.linear_fc2 = _FakeTELinear(intermediate, hidden)
        self.config = SimpleNamespace(gated_linear_unit=True)


class _RoutedOnlyMoEMLP(nn.Module):
    """A qwen3_moe-style MoE MLP: routed experts, no shared expert, no dense fc1."""

    def __init__(self):
        super().__init__()
        self.experts = nn.Module()


class _MixerOnlyAttention(nn.Module):
    """A GDN/linear-attention mixer: no fused linear_qkv to adapt."""


class _FakeLayer(nn.Module):
    def __init__(self, layer_number: int, attention: nn.Module, mlp: nn.Module):
        super().__init__()
        self.layer_number = layer_number
        self.self_attention = attention
        self.mlp = mlp


class _FakeModel(nn.Module):
    def __init__(self, layers, hidden: int, num_query_groups: int):
        super().__init__()
        self.embedding = nn.Embedding(8, hidden)
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList(layers)
        self.config = SimpleNamespace(
            hidden_size=hidden,
            layernorm_epsilon=1e-6,
            sequence_parallel=False,
            num_query_groups=num_query_groups,
        )


def _gqa_model(num_layers: int = 2, hidden: int = 8, num_q: int = 4, num_kv: int = 2, head_dim: int = 2):
    layers = [
        _FakeLayer(i + 1, _FakeAttention(hidden, num_q, num_kv, head_dim), _FakeMLP(hidden, 16))
        for i in range(num_layers)
    ]
    return _FakeModel(layers, hidden=hidden, num_query_groups=num_kv)


def _args(targets, **overrides) -> Namespace:
    values = dict(
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        target_modules=list(targets),
        hf_checkpoint=None,
    )
    values.update(overrides)
    return Namespace(**values)


@pytest.fixture
def single_rank(monkeypatch):
    from megatron.core import parallel_state

    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda: 0)


class TestApplyNativeLora:
    def test_full_target_set_attaches_every_seam(self, single_rank):
        model = apply_native_lora(_gqa_model(num_layers=2), _args(ALL_GQA_TARGETS))

        # qkv + o + fc1 + fc2 adapters per layer
        assert len(list(iter_adapters([model]))) == 2 * 4
        for name, parameter in model.named_parameters():
            assert parameter.requires_grad == ("lora_" in name), name

    def test_wrapped_forward_is_identity_at_init_and_trains_adapters(self, single_rank):
        model = apply_native_lora(_gqa_model(num_layers=1), _args(ALL_GQA_TARGETS))
        attention = model.decoder.layers[0].self_attention
        x = torch.randn(3, 8)

        out, bias = attention.linear_qkv(x)
        assert bias is None
        assert torch.allclose(out, F.linear(x, attention.linear_qkv.weight))  # B starts at zero

        out.sum().backward()
        adapter = attention.lora_qkv_adapter
        assert adapter.q_A.grad is not None
        assert adapter.q_B.grad is not None
        assert adapter.q_B.grad.abs().sum() > 0  # the delta path is live, not detached

    def test_no_matching_module_fails_with_the_supported_set(self, single_rank):
        layers = [_FakeLayer(1, _MixerOnlyAttention(), nn.Module())]
        model = _FakeModel(layers, hidden=8, num_query_groups=2)
        with pytest.raises(AssertionError, match="matched no modules"):
            apply_native_lora(model, _args(["q_proj"]))

    def test_routed_only_moe_with_expanded_targets_trains_attention_only(self, single_rank):
        """qwen3_moe + all-linear: parser-added MLP targets skip instead of crashing the run."""
        layers = [_FakeLayer(1, _FakeAttention(8, 4, 2, 2), _RoutedOnlyMoEMLP())]
        model = _FakeModel(layers, hidden=8, num_query_groups=2)
        args = _args(ALL_GQA_TARGETS, _target_modules_expanded_from_all_linear=True)

        model = apply_native_lora(model, args)

        adapters = list(iter_adapters([model]))
        assert len(adapters) == 2  # qkv + o only; no MLP adapters exist to attach

    def test_routed_only_moe_with_explicit_mlp_targets_fails_closed(self, single_rank):
        layers = [_FakeLayer(1, _FakeAttention(8, 4, 2, 2), _RoutedOnlyMoEMLP())]
        model = _FakeModel(layers, hidden=8, num_query_groups=2)
        with pytest.raises(AssertionError, match="routed/grouped expert"):
            apply_native_lora(model, _args(["q_proj", "gate_proj"]))

    def test_registry_resolves_model_type_from_config_json(self, single_rank, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
        model = apply_native_lora(_gqa_model(num_layers=1), _args(ALL_GQA_TARGETS, hf_checkpoint=str(tmp_path)))
        assert len(list(iter_adapters([model]))) == 4

    def test_unregistered_model_type_fails_before_any_attach(self, single_rank, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "gpt_oss"}))
        with pytest.raises(AssertionError, match="no spec registered"):
            apply_native_lora(_gqa_model(num_layers=1), _args(ALL_GQA_TARGETS, hf_checkpoint=str(tmp_path)))


def _context(*targets: str) -> AttachContext:
    return AttachContext(
        lora=LoRAConfig(rank=2, alpha=4, dropout=0.0, target_modules=frozenset(targets)),
        transformer_config=SimpleNamespace(
            hidden_size=8,
            layernorm_epsilon=1e-6,
            sequence_parallel=False,
        ),
        tp_size=1,
        tp_rank=0,
        layer_prefix="model.layers.",
        shared_expert="mlp.shared_expert.",
    )


class TestAssertSupportedRun:
    def test_plain_run_passes(self):
        _assert_supported_run(Namespace(), _context("q_proj"))

    def test_overlap_param_gather_is_rejected(self):
        with pytest.raises(AssertionError, match="overlap-param-gather"):
            _assert_supported_run(Namespace(overlap_param_gather=True), _context("q_proj"))

    def test_moe_shared_expert_overlap_is_rejected(self):
        with pytest.raises(AssertionError, match="moe-shared-expert-overlap"):
            _assert_supported_run(Namespace(moe_shared_expert_overlap=True), _context("q_proj"))

    def test_colocate_without_weights_backuper_is_rejected(self):
        args = Namespace(colocate=True, enable_weights_backuper=False)
        with pytest.raises(AssertionError, match="backuper"):
            _assert_supported_run(args, _context("q_proj"))

    def test_colocate_with_weights_backuper_passes(self):
        _assert_supported_run(Namespace(colocate=True, enable_weights_backuper=True), _context("q_proj"))


class _FakeMLAAttention(nn.Module):
    def __init__(self, hidden=8, heads=2, q_head_dim=3, q_lora_rank=4, kv_lora_rank=4, qk_pos=2, with_q_path=True):
        super().__init__()
        if with_q_path:
            self.linear_q_down_proj = _FakeTELinear(hidden, q_lora_rank)
            self.linear_q_up_proj = _FakeTELinear(q_lora_rank, heads * q_head_dim)
        self.linear_kv_down_proj = _FakeTELinear(hidden, kv_lora_rank + qk_pos)
        self.linear_kv_up_proj = _FakeTELinear(kv_lora_rank, heads * (2 + 2))
        self.linear_proj = _FakeTELinear(heads * 2, hidden)
        self.num_attention_heads_per_partition = heads
        self.q_head_dim = q_head_dim


def _mla_context(*targets: str) -> AttachContext:
    return AttachContext(
        lora=LoRAConfig(rank=2, alpha=4, dropout=0.0, target_modules=frozenset(targets)),
        transformer_config=SimpleNamespace(
            hidden_size=8,
            layernorm_epsilon=1e-6,
            sequence_parallel=False,
            q_lora_rank=4,
            kv_lora_rank=4,
            qk_pos_emb_head_dim=2,
            qk_head_dim=2,
            v_head_dim=2,
        ),
        tp_size=1,
        tp_rank=0,
        layer_prefix="model.layers.",
        shared_expert="mlp.shared_expert.",
    )


class TestMLAAttach:
    def test_compressed_query_path_attaches_all_five_projections(self):
        attention = _FakeMLAAttention()
        count = MLA_ATTENTION_SPEC.attach(attention, "model.layers.0.self_attn.", _mla_context(*MLA_TARGETS))
        assert count == 5
        assert attention.lora_mla_q_a_adapter.a_B.shape == (4, 2)  # q_lora_rank x rank
        assert attention.lora_mla_q_b_adapter.b_B.shape == (6, 2)  # heads*q_head_dim x rank
        assert attention.lora_mla_kv_a_adapter.a_B.shape == (6, 2)  # kv_lora_rank+qk_pos x rank
        assert attention.lora_mla_kv_b_adapter.b_B.shape == (8, 2)  # heads*(qk+v head dims) x rank
        assert attention.lora_o_adapter.o_B.shape == (8, 2)  # hidden x rank

    def test_uncompressed_query_path_attaches_only_kv_and_o(self):
        attention = _FakeMLAAttention(with_q_path=False)
        count = MLA_ATTENTION_SPEC.attach(attention, "model.layers.0.self_attn.", _mla_context(*MLA_TARGETS))
        assert count == 3
        assert not hasattr(attention, "lora_mla_q_a_adapter")
        assert hasattr(attention, "lora_mla_kv_a_adapter")
        assert hasattr(attention, "lora_o_adapter")

    def test_sharded_down_projection_is_rejected(self):
        attention = _FakeMLAAttention()
        attention.linear_kv_down_proj = _FakeTELinear(8, 3)  # narrower than kv_lora_rank + qk_pos
        with pytest.raises(AssertionError, match="replicated"):
            MLA_ATTENTION_SPEC.attach(attention, "model.layers.0.self_attn.", _mla_context("kv_a_proj_with_mqa"))
