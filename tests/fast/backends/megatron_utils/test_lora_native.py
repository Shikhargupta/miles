"""Unit tests for native (raw-mode) LoRA helpers — no GPU, no distributed init.

Covers the qkv output permutation, the provider-protocol resolver, architecture
guards, the per-rank adapter shard naming, and the rollout gate.
"""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.lora_native import (
    _assert_supported_architecture,
    _build_qkv_perm,
    export_lora_hf_named,
    load_lora_adapter_hf,
    resolve_lora_provider,
    wrap_model_provider_with_lora,
)
from miles.backends.megatron_utils.lora_utils import _native_adapter_shard_name, reduce_marked_lora_grads
from miles.utils.lora import lora_rollout_enabled


def _fake_model(num_layers=2, *, output_gate=False, mla=False, with_qkv=True, num_query_groups=8):
    layers = []
    for i in range(num_layers):
        attn = SimpleNamespace(layer_number=i + 1)
        if with_qkv:
            attn.linear_qkv = object()
        layers.append(SimpleNamespace(layer_number=i + 1, self_attention=attn))
    cfg = SimpleNamespace(
        attention_output_gate=output_gate,
        multi_latent_attention=mla,
        num_query_groups=num_query_groups,
    )
    return SimpleNamespace(config=cfg, decoder=SimpleNamespace(layers=layers))


# ---------------------------------------------------------------------------
# _build_qkv_perm
# ---------------------------------------------------------------------------


class TestBuildQkvPerm:
    def test_mha_single_group(self):
        # 1 query head, 1 group, head_dim 2 -> plain [q; k; v] is already interleaved
        perm = _build_qkv_perm(num_q_heads=1, num_groups=1, head_dim=2, device="cpu")
        assert perm.tolist() == [0, 1, 2, 3, 4, 5]

    def test_gqa_two_groups_matches_mcore_layout(self):
        # 4 query heads / 2 groups / head_dim 1: mcore emits [q0 q1 k0 v0 q2 q3 k1 v1]
        perm = _build_qkv_perm(num_q_heads=4, num_groups=2, head_dim=1, device="cpu")
        assert perm.tolist() == [0, 1, 4, 6, 2, 3, 5, 7]

    def test_permutation_is_a_bijection(self):
        nq, ng, hd = 8, 4, 3
        perm = _build_qkv_perm(num_q_heads=nq, num_groups=ng, head_dim=hd, device="cpu")
        total = (nq + 2 * ng) * hd
        assert perm.numel() == total
        assert sorted(perm.tolist()) == list(range(total))

    def test_applied_to_delta_places_projections_per_group(self):
        nq, ng, hd = 4, 2, 1
        perm = _build_qkv_perm(num_q_heads=nq, num_groups=ng, head_dim=hd, device="cpu")
        # plain layout: q rows 0..3, k rows 4..5, v rows 6..7
        plain = torch.tensor([[10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 30.0, 31.0]])
        out = plain.index_select(-1, perm)
        # group 0 -> q0 q1 k0 v0, group 1 -> q2 q3 k1 v1
        assert out.tolist() == [[10.0, 11.0, 20.0, 30.0, 12.0, 13.0, 21.0, 31.0]]


# ---------------------------------------------------------------------------
# Architecture guards
# ---------------------------------------------------------------------------


class TestArchitectureGuards:
    def test_plain_gqa_model_passes(self):
        model = _fake_model()
        _assert_supported_architecture(model.config, model)  # does not raise

    def test_output_gate_rejected(self):
        model = _fake_model(output_gate=True)
        with pytest.raises(AssertionError, match="attention_output_gate"):
            _assert_supported_architecture(model.config, model)

    def test_mla_rejected(self):
        model = _fake_model(mla=True)
        with pytest.raises(AssertionError, match="multi_latent_attention"):
            _assert_supported_architecture(model.config, model)

    def test_missing_linear_qkv_rejected(self):
        model = _fake_model(with_qkv=False)
        with pytest.raises(AssertionError, match="no linear_qkv"):
            _assert_supported_architecture(model.config, model)

    def test_query_groups_below_tp_size_rejected(self):
        model = _fake_model(num_query_groups=2)
        with pytest.raises(AssertionError, match="num_query_groups"):
            _assert_supported_architecture(model.config, model, tp_size=4)

    def test_query_groups_equal_to_tp_size_passes(self):
        model = _fake_model(num_query_groups=4)
        _assert_supported_architecture(model.config, model, tp_size=4)  # does not raise

    def test_error_names_the_escape_hatch(self):
        model = _fake_model(output_gate=True)
        with pytest.raises(AssertionError, match="--lora-provider-path"):
            _assert_supported_architecture(model.config, model)


# ---------------------------------------------------------------------------
# resolve_lora_provider
# ---------------------------------------------------------------------------


class TestResolveLoraProvider:
    def test_default_is_this_module(self):
        mod = resolve_lora_provider(Namespace())
        assert mod.wrap_model_provider_with_lora is wrap_model_provider_with_lora
        assert mod.export_lora_hf_named is export_lora_hf_named
        assert mod.load_lora_adapter_hf is load_lora_adapter_hf

    def test_explicit_path_is_imported(self):
        args = Namespace(lora_provider_path="miles.backends.megatron_utils.lora_native")
        assert resolve_lora_provider(args).export_lora_hf_named is export_lora_hf_named

    def test_module_without_protocol_is_rejected(self):
        args = Namespace(lora_provider_path="json")
        with pytest.raises(AssertionError, match="wrap_model_provider_with_lora"):
            resolve_lora_provider(args)


# ---------------------------------------------------------------------------
# wrap_model_provider_with_lora
# ---------------------------------------------------------------------------


class TestWrapModelProvider:
    def test_provider_args_are_forwarded_and_result_wrapped(self):
        seen = {}

        def provider(pre_process, post_process, vp_stage=None):
            seen.update(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
            return _fake_model()

        calls = []
        wrapped = wrap_model_provider_with_lora(provider, Namespace(lora_rank=8))
        # patch the applier so the test stays free of megatron parallel state
        import miles.backends.megatron_utils.lora_native as ln

        orig = ln.apply_native_lora
        ln.apply_native_lora = lambda m, a: calls.append((m, a)) or m
        try:
            out = wrapped(True, False, vp_stage=1)
        finally:
            ln.apply_native_lora = orig

        assert seen == {"pre_process": True, "post_process": False, "vp_stage": 1}
        assert out is calls[0][0]


# ---------------------------------------------------------------------------
# Per-rank adapter shard naming
# ---------------------------------------------------------------------------


class TestNativeAdapterShardName:
    def test_no_ep_keeps_legacy_name(self):
        assert _native_adapter_shard_name(1, 2, 0) == "adapter_megatron_tp1_pp2.pt"

    def test_ep_rank_is_included(self):
        assert _native_adapter_shard_name(1, 2, 3) == "adapter_megatron_tp1_pp2_ep3.pt"

    def test_ranks_sharing_tp_pp_get_distinct_names(self):
        names = {_native_adapter_shard_name(0, 0, ep) for ep in range(4)}
        assert len(names) == 4


# ---------------------------------------------------------------------------
# reduce_marked_lora_grads
# ---------------------------------------------------------------------------


class TestReduceMarkedLoraGrads:
    def test_no_marked_params_is_a_noop_without_distributed(self):
        chunk = torch.nn.Linear(2, 2)
        # no _lora_grad_sum_group tags -> returns before touching parallel state
        reduce_marked_lora_grads([chunk])

    def test_empty_model_list_is_a_noop(self):
        reduce_marked_lora_grads([])


# ---------------------------------------------------------------------------
# lora_rollout_enabled
# ---------------------------------------------------------------------------


class TestLoraRolloutEnabled:
    def test_enabled_when_lora_on_and_not_train_only(self):
        assert lora_rollout_enabled(Namespace(lora_rank=16, lora_train_only=False))

    def test_disabled_under_train_only(self):
        assert not lora_rollout_enabled(Namespace(lora_rank=16, lora_train_only=True))

    def test_disabled_without_lora(self):
        assert not lora_rollout_enabled(Namespace(lora_rank=0, lora_train_only=False))

    def test_missing_train_only_attr_defaults_to_enabled(self):
        assert lora_rollout_enabled(Namespace(lora_rank=8))
