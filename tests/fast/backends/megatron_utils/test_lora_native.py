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
    _hf_naming,
    _require_grad_on_first_activation,
    _rmsnorm,
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

    def test_output_gate_deinterleaves_the_query_slices(self):
        # 2 query heads, 1 group, head_dim 1, gated: HF q_proj is [q0 g0 q1 g1],
        # mcore's group is [q0 q1 g0 g1] followed by k and v.
        perm = _build_qkv_perm(num_q_heads=2, num_groups=1, head_dim=1, device="cpu", output_gate=True)
        assert perm.tolist() == [0, 2, 1, 3, 4, 5]

    def test_output_gate_permutation_is_a_bijection(self):
        nq, ng, hd = 8, 2, 3
        perm = _build_qkv_perm(num_q_heads=nq, num_groups=ng, head_dim=hd, device="cpu", output_gate=True)
        total = (2 * nq + 2 * ng) * hd
        assert perm.numel() == total
        assert sorted(perm.tolist()) == list(range(total))

    def test_output_gate_applied_to_delta(self):
        perm = _build_qkv_perm(num_q_heads=4, num_groups=2, head_dim=1, device="cpu", output_gate=True)
        # HF order: [q0 g0 q1 g1 | q2 g2 q3 g3 | k0 k1 | v0 v1]
        plain = torch.tensor([[10.0, 40.0, 11.0, 41.0, 12.0, 42.0, 13.0, 43.0, 20.0, 21.0, 30.0, 31.0]])
        out = plain.index_select(-1, perm)
        # group 0 -> q0 q1 g0 g1 k0 v0, group 1 -> q2 q3 g2 g3 k1 v1
        assert out.tolist() == [[10.0, 11.0, 40.0, 41.0, 20.0, 30.0, 12.0, 13.0, 42.0, 43.0, 21.0, 31.0]]


# ---------------------------------------------------------------------------
# Recomputing the fused layernorm the branch input goes through
# ---------------------------------------------------------------------------


class TestRmsNorm:
    def test_plain_gamma_scales_by_the_stored_weight(self):
        x = torch.tensor([[3.0, 4.0]])
        gamma = torch.tensor([2.0, 2.0])
        got = _rmsnorm(x, gamma, eps=0.0)
        assert torch.allclose(got, torch.tensor([[3.0, 4.0]]) / 3.5355339 * 2.0, atol=1e-5)

    def test_zero_centered_gamma_adds_the_one_back(self):
        """--apply-layernorm-1p stores gamma - 1; the branch must see the same input
        the base GEMM does, or the adapter is fed a differently scaled activation."""
        x = torch.tensor([[3.0, 4.0]])
        stored = torch.tensor([1.0, 1.0])
        assert torch.allclose(
            _rmsnorm(x, stored, eps=0.0, zero_centered_gamma=True),
            _rmsnorm(x, stored + 1.0, eps=0.0),
        )


# ---------------------------------------------------------------------------
# Keeping the graph alive through activation recomputation
# ---------------------------------------------------------------------------


class TestFirstActivationGrad:
    """A frozen base plus recomputation is the case that silently trains nothing.

    Every adapter param sits inside a checkpointed block, so unless the block's
    input requires grad, autograd never enters the region and every adapter
    gradient comes back zero while all the sync checks still pass.
    """

    def test_a_frozen_embedding_output_has_no_graph_on_its_own(self):
        embedding = torch.nn.Embedding(4, 3)
        embedding.weight.requires_grad_(False)
        assert not embedding(torch.tensor([0, 1])).requires_grad

    def test_hook_makes_the_first_activation_require_grad(self):
        embedding = torch.nn.Embedding(4, 3)
        embedding.weight.requires_grad_(False)
        model = SimpleNamespace(embedding=embedding)
        assert _require_grad_on_first_activation(model) is embedding
        assert embedding(torch.tensor([0, 1])).requires_grad

    def test_stage_without_an_embedding_is_a_noop(self):
        assert _require_grad_on_first_activation(SimpleNamespace()) is None


# ---------------------------------------------------------------------------
# HF naming read off the checkpoint
# ---------------------------------------------------------------------------


def _write_index(tmp_path, keys):
    import json

    weight_map = {key: "a.safetensors" for key in keys}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return str(tmp_path)


class TestHfNaming:
    def test_deepseek_style_plural_shared_expert(self, tmp_path):
        path = _write_index(
            tmp_path,
            [
                "model.layers.0.self_attn.o_proj.weight",
                "model.layers.1.mlp.shared_experts.gate_proj.weight",
            ],
        )
        assert _hf_naming(path) == ("model.layers.", "mlp.shared_experts.")

    def test_qwen3_5_nests_the_decoder_and_uses_singular(self, tmp_path):
        """The mtp block also has `layers.N.`; it must not win the prefix vote."""
        path = _write_index(
            tmp_path,
            [
                "model.language_model.layers.0.self_attn.q_proj.weight",
                "model.language_model.layers.1.mlp.shared_expert.up_proj.weight",
                "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
                "vision_tower.encoder.blocks.0.wo.weight",
            ],
        )
        assert _hf_naming(path) == ("model.language_model.layers.", "mlp.shared_expert.")

    def test_missing_index_falls_back_to_the_plain_layout(self, tmp_path):
        assert _hf_naming(str(tmp_path)) == ("model.layers.", "mlp.shared_expert.")
        assert _hf_naming(None) == ("model.layers.", "mlp.shared_expert.")


# ---------------------------------------------------------------------------
# Architecture guards
# ---------------------------------------------------------------------------


class TestArchitectureGuards:
    def test_plain_gqa_model_passes(self):
        model = _fake_model()
        _assert_supported_architecture(model.config)  # does not raise

    def test_output_gate_is_supported(self):
        """The gated query slice is handled by the permutation, not rejected."""
        model = _fake_model(output_gate=True)
        _assert_supported_architecture(model.config)  # does not raise

    def test_mla_is_supported(self):
        """MLA has its own projection set; the fused-qkv guards must not fire on it."""
        model = _fake_model(mla=True, with_qkv=False)
        _assert_supported_architecture(model.config, tp_size=2)  # does not raise

    def test_missing_linear_qkv_is_not_an_error(self):
        """Mixer-only layers carry no attention adapter; apply_native_lora reports them."""
        model = _fake_model(with_qkv=False)
        _assert_supported_architecture(model.config)  # does not raise

    def test_query_groups_below_tp_size_rejected(self):
        model = _fake_model(num_query_groups=2)
        with pytest.raises(AssertionError, match="num_query_groups"):
            _assert_supported_architecture(model.config, tp_size=4)

    def test_query_groups_equal_to_tp_size_passes(self):
        model = _fake_model(num_query_groups=4)
        _assert_supported_architecture(model.config, tp_size=4)  # does not raise

    def test_error_names_the_escape_hatch(self):
        model = _fake_model(num_query_groups=2)
        with pytest.raises(AssertionError, match="--lora-provider-path"):
            _assert_supported_architecture(model.config, tp_size=4)


# ---------------------------------------------------------------------------
# Real checkpoints that need their own provider
# ---------------------------------------------------------------------------


class TestShippedRegistries:
    """Lock in which shipped model registries the generic provider serves.

    Values mirror scripts/models/*.sh. These are not hypothetical: each is a
    checkpoint someone will eventually point at --megatron-to-hf-mode raw, so a
    layout the generic path cannot slice has to fail with a startup assert naming
    --lora-provider-path rather than produce silently wrong gradients.
    """

    @pytest.mark.parametrize(
        "registry,kwargs",
        [
            # glm4.7-flash.sh: --multi-latent-attention --q-lora-rank 768 --kv-lora-rank 512
            ("glm4.7-flash", dict(mla=True)),
            # kimi-k25_2layer.sh -> kimi-k2-thinking.sh: --multi-latent-attention
            ("kimi-k25_2layer", dict(mla=True)),
            # glm5-744B-A40B_4layer.sh -> glm5-744B-A40B.sh: --multi-latent-attention
            ("glm5-744B-A40B_4layer", dict(mla=True)),
            # deepseek-v4-flash-4layer.sh: --multi-latent-attention
            ("deepseek-v4-flash-4layer", dict(mla=True)),
        ],
    )
    def test_mla_registries_are_accepted(self, registry, kwargs):
        """MLA is covered by _attach_mla_attention, including when TP exceeds the
        (meaningless for MLA) query-group count."""
        model = _fake_model(num_query_groups=2, **kwargs)
        _assert_supported_architecture(model.config, tp_size=4)  # does not raise

    def test_qwen3_5_gated_hybrid_is_accepted(self):
        """qwen3.5-35B-A3B.sh: --attention-output-gate plus GDN mixer layers.

        The gated query slice is permuted like any other, and a mixer layer simply
        carries no attention adapter, so TP <= num_query_groups is the only bar left.
        """
        model = _fake_model(output_gate=True, num_query_groups=2)
        _assert_supported_architecture(model.config, tp_size=2)  # does not raise

    def test_qwen3_5_above_query_group_count_still_rejected(self):
        model = _fake_model(output_gate=True, num_query_groups=2)
        with pytest.raises(AssertionError) as excinfo:
            _assert_supported_architecture(model.config, tp_size=4)
        message = str(excinfo.value)
        assert "num_query_groups" in message
        assert "--lora-provider-path" in message


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
