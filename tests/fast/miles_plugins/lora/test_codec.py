"""HF codec tests for exact native-LoRA projection selection."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from miles.backends.megatron_utils.lora_utils import (
    _adapter_shards_are_ep_sharded,
    _all_ranks_agree_on_training_state,
    load_lora_adapter,
)
from miles_plugins.lora.codec.checkpoint import (
    load_native_adapter_state_dict,
    model_chunk_state_key,
    native_adapter_state_dict,
)
from miles_plugins.lora.codec.hf import export_lora_hf_named, load_lora_adapter_hf, target_modules_from_hf_names
from miles_plugins.lora.config import LoRAConfig
from miles_plugins.lora.modules.linear import LoRASplitFC1, LoRASplitQKV
from miles_plugins.lora.spec.attention import QKV_PROJECTIONS
from miles_plugins.lora.spec.base import AttachContext
from miles_plugins.lora.spec.mlp import FC1_PROJECTIONS


def _context(*targets: str, tp_size: int = 1, tp_rank: int = 0) -> AttachContext:
    return AttachContext(
        lora=LoRAConfig(
            rank=2,
            alpha=4,
            dropout=0.0,
            target_modules=frozenset(targets),
        ),
        transformer_config=SimpleNamespace(
            hidden_size=8,
            layernorm_epsilon=1e-6,
            sequence_parallel=False,
            layernorm_zero_centered_gamma=False,
            attention_output_gate=False,
        ),
        tp_size=tp_size,
        tp_rank=tp_rank,
        layer_prefix="model.layers.",
        shared_expert="mlp.shared_expert.",
    )


def _partial_model() -> nn.Module:
    model = nn.Module()
    model._miles_native_lora_provider = True
    model.q_adapter = LoRASplitQKV(
        hf_prefix="model.layers.0.self_attn.",
        reference=torch.empty(4, 8),
        context=_context("q_proj"),
        projections=QKV_PROJECTIONS[:1],
        num_q=2,
        num_kv=1,
        head_dim=1,
    )
    model.gate_adapter = LoRASplitFC1(
        hf_prefix="model.layers.0.mlp.",
        reference=torch.empty(16, 8),
        context=_context("gate_proj"),
        projections=FC1_PROJECTIONS[:1],
        inter_local=8,
    )
    return model


def _tp_gate_model(*, tp_rank: int = 0) -> nn.Module:
    model = nn.Module()
    model.gate_adapter = LoRASplitFC1(
        hf_prefix="model.layers.0.mlp.",
        reference=torch.empty(16, 8),
        context=_context("gate_proj", tp_size=2, tp_rank=tp_rank),
        projections=FC1_PROJECTIONS[:1],
        inter_local=8,
    )
    return model


def test_partial_targets_export_and_load_only_requested_projections(tmp_path):
    source = _partial_model()
    with torch.no_grad():
        for index, parameter in enumerate(source.parameters(), start=1):
            parameter.fill_(index)

    exported = dict(export_lora_hf_named([source]))
    assert set(exported) == {
        "model.layers.0.self_attn.q_proj.lora_A.weight",
        "model.layers.0.self_attn.q_proj.lora_B.weight",
        "model.layers.0.mlp.gate_proj.lora_A.weight",
        "model.layers.0.mlp.gate_proj.lora_B.weight",
    }
    assert target_modules_from_hf_names(exported) == ["gate_proj", "q_proj"]
    save_file(exported, tmp_path / "adapter_model.safetensors")

    target = _partial_model()
    assert load_lora_adapter_hf([target], str(tmp_path)) == 4
    for source_parameter, target_parameter in zip(source.parameters(), target.parameters(), strict=True):
        assert torch.equal(source_parameter, target_parameter)


def test_native_checkpoint_codec_reports_legacy_extra_projection_tensors():
    source = _partial_model()
    state = native_adapter_state_dict([source])
    assert set(state) == {
        model_chunk_state_key(0, "q_adapter.q_A"),
        model_chunk_state_key(0, "q_adapter.q_B"),
        model_chunk_state_key(0, "gate_adapter.gate_A"),
        model_chunk_state_key(0, "gate_adapter.gate_B"),
    }
    state[model_chunk_state_key(0, "q_adapter.k_A")] = torch.zeros(2, 8)
    state[model_chunk_state_key(0, "q_adapter.k_B")] = torch.zeros(1, 2)

    target = _partial_model()
    loaded, unexpected, missing = load_native_adapter_state_dict([target], state)
    assert loaded == 4
    assert unexpected == [
        model_chunk_state_key(0, "q_adapter.k_A"),
        model_chunk_state_key(0, "q_adapter.k_B"),
    ]
    assert missing == []


def test_native_checkpoint_namespaces_virtual_pipeline_chunks_and_round_trips():
    source = [_partial_model(), _partial_model()]
    with torch.no_grad():
        for chunk_index, chunk in enumerate(source, start=1):
            for parameter_index, parameter in enumerate(chunk.parameters(), start=1):
                parameter.fill_(chunk_index * 10 + parameter_index)

    state = native_adapter_state_dict(source)
    assert len(state) == 8
    assert model_chunk_state_key(0, "q_adapter.q_A") in state
    assert model_chunk_state_key(1, "q_adapter.q_A") in state

    target = [_partial_model(), _partial_model()]
    loaded, unexpected, missing = load_native_adapter_state_dict(target, state)

    assert loaded == 8
    assert unexpected == []
    assert missing == []
    for source_chunk, target_chunk in zip(source, target, strict=True):
        for source_parameter, target_parameter in zip(
            source_chunk.parameters(), target_chunk.parameters(), strict=True
        ):
            assert torch.equal(source_parameter, target_parameter)


def test_legacy_single_chunk_native_checkpoint_still_loads():
    source = _partial_model()
    namespaced = native_adapter_state_dict([source])
    prefix = model_chunk_state_key(0, "")
    legacy = {name.removeprefix(prefix): tensor for name, tensor in namespaced.items()}
    target = _partial_model()

    loaded, unexpected, missing = load_native_adapter_state_dict([target], legacy)

    assert loaded == 4
    assert unexpected == []
    assert missing == []
    for source_parameter, target_parameter in zip(source.parameters(), target.parameters(), strict=True):
        assert torch.equal(source_parameter, target_parameter)


def test_legacy_multi_chunk_checkpoint_loads_when_names_are_unambiguous():
    empty_source = nn.Module()
    empty_source._miles_native_lora_provider = True
    source = [empty_source, _partial_model()]
    namespaced = native_adapter_state_dict(source)
    prefix = model_chunk_state_key(1, "")
    legacy = {name.removeprefix(prefix): tensor for name, tensor in namespaced.items()}

    empty_target = nn.Module()
    empty_target._miles_native_lora_provider = True
    target = [empty_target, _partial_model()]
    loaded, unexpected, missing = load_native_adapter_state_dict(target, legacy)

    assert loaded == 4
    assert unexpected == []
    assert missing == []
    for source_parameter, target_parameter in zip(source[1].parameters(), target[1].parameters(), strict=True):
        assert torch.equal(source_parameter, target_parameter)


def test_legacy_multi_chunk_checkpoint_rejects_ambiguous_names():
    chunks = [_partial_model(), _partial_model()]
    namespaced = native_adapter_state_dict(chunks)
    legacy = {
        name.split(".", 2)[-1]: tensor
        for name, tensor in namespaced.items()
        if name.startswith("_miles_model_chunks.1.")
    }

    with pytest.raises(ValueError, match="ambiguous across multiple model chunks"):
        load_native_adapter_state_dict(chunks, legacy)


def test_native_shape_mismatch_is_rejected_before_any_copy():
    source = _partial_model()
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(7)
    state = native_adapter_state_dict([source])
    state[model_chunk_state_key(0, "q_adapter.q_A")] = torch.full((1, 8), 7.0)
    target = _partial_model()
    before = [parameter.detach().clone() for parameter in target.parameters()]

    with pytest.raises(ValueError, match="shape mismatch"):
        load_native_adapter_state_dict([target], state)

    for expected, parameter in zip(before, target.parameters(), strict=True):
        assert torch.equal(expected, parameter)


def test_hf_loader_rejects_oversized_global_tp_tensor_before_slice(tmp_path):
    target = _tp_gate_model(tp_rank=0)
    before = [parameter.detach().clone() for parameter in target.parameters()]
    save_file(
        {
            "model.layers.0.mlp.gate_proj.lora_A.weight": torch.full((2, 8), 3.0),
            # Expected global B is (16, 2); the old loader sliced the first
            # eight rows and silently ignored this oversized tail.
            "model.layers.0.mlp.gate_proj.lora_B.weight": torch.full((17, 2), 4.0),
        },
        tmp_path / "adapter_model.safetensors",
    )

    with pytest.raises(ValueError, match=r"checkpoint \(17, 2\) != expected \(16, 2\)"):
        load_lora_adapter_hf([target], str(tmp_path))

    for expected, parameter in zip(before, target.parameters(), strict=True):
        assert torch.equal(expected, parameter)


def test_explicit_provider_marker_is_the_native_shard_policy_source():
    assert not _adapter_shards_are_ep_sharded([_partial_model()])
    assert _adapter_shards_are_ep_sharded([nn.Module()])

    unmarked_custom_model = _partial_model()
    del unmarked_custom_model._miles_native_lora_provider
    assert _adapter_shards_are_ep_sharded([unmarked_custom_model])
    assert _adapter_shards_are_ep_sharded([_partial_model(), unmarked_custom_model])

    empty_native_chunk = nn.Module()
    empty_native_chunk._miles_native_lora_provider = True
    assert not _adapter_shards_are_ep_sharded([empty_native_chunk])


def test_native_shard_loads_on_a_nonzero_ep_rank(tmp_path, monkeypatch):
    """Native state is EP-invariant: an EP rank > 0 must find the shard written for its (tp, pp).

    Regression guard — the shard name used to carry ep_rank while only dense-DP-rank-0 ranks wrote,
    so under TP2/EP4 the EP 2/3 ranks found no file and silently kept fresh adapters.
    """
    torch.save(native_adapter_state_dict([_partial_model()]), tmp_path / "adapter_megatron_tp0_pp0.pt")
    parallel_state = SimpleNamespace(
        tp=SimpleNamespace(rank=0),
        pp=SimpleNamespace(rank=0),
        ep=SimpleNamespace(rank=3),
    )
    monkeypatch.setattr(
        "miles.backends.megatron_utils.lora_utils.get_parallel_state",
        lambda: parallel_state,
    )

    loaded, iteration = load_lora_adapter([_partial_model()], str(tmp_path))

    assert loaded
    assert iteration is None


def test_native_checkpoint_codec_reports_missing_projection_tensors():
    """A shard saved with fewer targets leaves the extra adapters fresh — that must be visible."""
    source = _partial_model()
    state = native_adapter_state_dict([source])
    del state[model_chunk_state_key(0, "gate_adapter.gate_A")]
    del state[model_chunk_state_key(0, "gate_adapter.gate_B")]

    target = _partial_model()
    loaded, unexpected, missing = load_native_adapter_state_dict([target], state)
    assert loaded == 2
    assert unexpected == []
    assert missing == [
        model_chunk_state_key(0, "gate_adapter.gate_A"),
        model_chunk_state_key(0, "gate_adapter.gate_B"),
    ]


def test_legacy_partial_shard_skips_incompatible_optimizer_restore(tmp_path, monkeypatch):
    source = _partial_model()
    state = native_adapter_state_dict([source])
    state[model_chunk_state_key(0, "q_adapter.k_A")] = torch.zeros(2, 8)
    state[model_chunk_state_key(0, "q_adapter.k_B")] = torch.zeros(1, 2)
    torch.save(state, tmp_path / "adapter_megatron_tp0_pp0.pt")
    torch.save(
        {"iteration": 7, "optimizer": {"legacy": True}, "opt_param_scheduler": None},
        tmp_path / "training_state_rank0.pt",
    )
    parallel_state = SimpleNamespace(
        tp=SimpleNamespace(rank=0),
        pp=SimpleNamespace(rank=0),
        ep=SimpleNamespace(rank=0),
    )
    monkeypatch.setattr(
        "miles.backends.megatron_utils.lora_utils.get_parallel_state",
        lambda: parallel_state,
    )
    optimizer = MagicMock()

    loaded, iteration = load_lora_adapter([_partial_model()], str(tmp_path), optimizer=optimizer)

    assert not loaded
    assert iteration is None
    optimizer.load_state_dict.assert_not_called()


def test_narrower_shard_skips_incompatible_optimizer_restore(tmp_path, monkeypatch):
    """Resuming with more targets than the shard was saved with must not restore optimizer state."""
    source = _partial_model()
    state = native_adapter_state_dict([source])
    del state[model_chunk_state_key(0, "gate_adapter.gate_A")]
    del state[model_chunk_state_key(0, "gate_adapter.gate_B")]
    torch.save(state, tmp_path / "adapter_megatron_tp0_pp0.pt")
    torch.save(
        {"iteration": 7, "optimizer": {"legacy": True}, "opt_param_scheduler": None},
        tmp_path / "training_state_rank0.pt",
    )
    parallel_state = SimpleNamespace(
        tp=SimpleNamespace(rank=0),
        pp=SimpleNamespace(rank=0),
        ep=SimpleNamespace(rank=0),
    )
    monkeypatch.setattr(
        "miles.backends.megatron_utils.lora_utils.get_parallel_state",
        lambda: parallel_state,
    )
    optimizer = MagicMock()

    loaded, iteration = load_lora_adapter([_partial_model()], str(tmp_path), optimizer=optimizer)

    assert not loaded
    assert iteration is None
    optimizer.load_state_dict.assert_not_called()


def test_native_resume_shape_mismatch_keeps_all_parameters_unmodified(tmp_path, monkeypatch):
    source = _partial_model()
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(8)
    state = native_adapter_state_dict([source])
    state[model_chunk_state_key(0, "q_adapter.q_A")] = torch.full((1, 8), 8.0)
    torch.save(state, tmp_path / "adapter_megatron_tp0_pp0.pt")
    parallel_state = SimpleNamespace(
        tp=SimpleNamespace(rank=0),
        pp=SimpleNamespace(rank=0),
        ep=SimpleNamespace(rank=0),
    )
    monkeypatch.setattr(
        "miles.backends.megatron_utils.lora_utils.get_parallel_state",
        lambda: parallel_state,
    )
    target = _partial_model()
    before = [parameter.detach().clone() for parameter in target.parameters()]

    loaded, iteration = load_lora_adapter([target], str(tmp_path))

    assert not loaded
    assert iteration is None
    for expected, parameter in zip(before, target.parameters(), strict=True):
        assert torch.equal(expected, parameter)


def test_remote_rank_preflight_failure_keeps_local_adapter_unmodified(tmp_path, monkeypatch):
    source = _partial_model()
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(9)
    torch.save(native_adapter_state_dict([source]), tmp_path / "adapter_megatron_tp0_pp0.pt")
    parallel_state = SimpleNamespace(
        tp=SimpleNamespace(rank=0),
        pp=SimpleNamespace(rank=0),
        ep=SimpleNamespace(rank=0),
    )
    monkeypatch.setattr(
        "miles.backends.megatron_utils.lora_utils.get_parallel_state",
        lambda: parallel_state,
    )
    monkeypatch.setattr(
        "miles.backends.megatron_utils.lora_utils._all_ranks_can_restore_training_state",
        lambda _local_ok: False,
    )
    target = _partial_model()
    before = [parameter.detach().clone() for parameter in target.parameters()]

    loaded, iteration = load_lora_adapter([target], str(tmp_path))

    assert not loaded
    assert iteration is None
    for expected, parameter in zip(before, target.parameters(), strict=True):
        assert torch.equal(expected, parameter)


def test_matching_shard_restores_optimizer_state(tmp_path, monkeypatch):
    source = _partial_model()
    torch.save(native_adapter_state_dict([source]), tmp_path / "adapter_megatron_tp0_pp0.pt")
    torch.save(
        {"iteration": 7, "optimizer": {"step": 7}, "opt_param_scheduler": None},
        tmp_path / "training_state_rank0.pt",
    )
    parallel_state = SimpleNamespace(
        tp=SimpleNamespace(rank=0),
        pp=SimpleNamespace(rank=0),
        ep=SimpleNamespace(rank=0),
    )
    monkeypatch.setattr(
        "miles.backends.megatron_utils.lora_utils.get_parallel_state",
        lambda: parallel_state,
    )
    optimizer = MagicMock()

    loaded, iteration = load_lora_adapter([_partial_model()], str(tmp_path), optimizer=optimizer)

    assert loaded
    assert iteration == 7
    optimizer.load_state_dict.assert_called_once_with({"step": 7})
    optimizer.reload_model_params.assert_not_called()


def test_corrupt_training_state_falls_back_collectively_and_refreshes_master_params(tmp_path, monkeypatch):
    source = _partial_model()
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(11)
    torch.save(native_adapter_state_dict([source]), tmp_path / "adapter_megatron_tp0_pp0.pt")
    (tmp_path / "training_state_rank0.pt").write_bytes(b"not a torch checkpoint")
    parallel_state = SimpleNamespace(
        tp=SimpleNamespace(rank=0),
        pp=SimpleNamespace(rank=0),
        ep=SimpleNamespace(rank=0),
    )
    monkeypatch.setattr(
        "miles.backends.megatron_utils.lora_utils.get_parallel_state",
        lambda: parallel_state,
    )
    target = _partial_model()
    optimizer = MagicMock()

    loaded, iteration = load_lora_adapter([target], str(tmp_path), optimizer=optimizer)

    assert loaded
    assert iteration is None
    assert target._miles_lora_native_checkpoint_loaded is True
    optimizer.load_state_dict.assert_not_called()
    optimizer.reload_model_params.assert_called_once_with()
    for parameter in target.parameters():
        assert torch.all(parameter == 11)


def test_training_state_consensus_rejects_different_rank_iterations(monkeypatch):
    monkeypatch.setattr(
        "miles.backends.megatron_utils.lora_utils.dist.is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        "miles.backends.megatron_utils.lora_utils.dist.get_world_size",
        lambda: 2,
    )

    def gather(states, _local):
        states[:] = [(True, 7), (True, 6)]

    monkeypatch.setattr(
        "miles.backends.megatron_utils.lora_utils.dist.all_gather_object",
        gather,
    )

    assert not _all_ranks_agree_on_training_state(True, 7)
