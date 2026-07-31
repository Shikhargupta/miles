"""HF codec tests for exact native-LoRA projection selection."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from safetensors.torch import save_file

from miles.backends.megatron_utils.lora_utils import load_lora_adapter
from miles_plugins.lora.codec.checkpoint import load_native_adapter_state_dict, native_adapter_state_dict
from miles_plugins.lora.codec.hf import export_lora_hf_named, load_lora_adapter_hf, target_modules_from_hf_names
from miles_plugins.lora.config import LoRAConfig
from miles_plugins.lora.modules.linear import SplitFC1, SplitQKV
from miles_plugins.lora.spec.attention import QKV_PROJECTIONS
from miles_plugins.lora.spec.base import AttachContext
from miles_plugins.lora.spec.mlp import FC1_PROJECTIONS


def _context(*targets: str) -> AttachContext:
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
        tp_size=1,
        tp_rank=0,
        layer_prefix="model.layers.",
        shared_expert="mlp.shared_expert.",
    )


def _partial_model() -> nn.Module:
    model = nn.Module()
    model.q_adapter = SplitQKV(
        hf_prefix="model.layers.0.self_attn.",
        reference=torch.empty(4, 8),
        context=_context("q_proj"),
        projections=QKV_PROJECTIONS[:1],
        num_q=2,
        num_kv=1,
        head_dim=1,
    )
    model.gate_adapter = SplitFC1(
        hf_prefix="model.layers.0.mlp.",
        reference=torch.empty(16, 8),
        context=_context("gate_proj"),
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
        "q_adapter.q_A",
        "q_adapter.q_B",
        "gate_adapter.gate_A",
        "gate_adapter.gate_B",
    }
    state["q_adapter.k_A"] = torch.zeros(2, 8)
    state["q_adapter.k_B"] = torch.zeros(1, 2)

    target = _partial_model()
    loaded, unexpected = load_native_adapter_state_dict([target], state)
    assert loaded == 4
    assert unexpected == ["q_adapter.k_A", "q_adapter.k_B"]


def test_legacy_partial_shard_skips_incompatible_optimizer_restore(tmp_path, monkeypatch):
    source = _partial_model()
    state = native_adapter_state_dict([source])
    state["q_adapter.k_A"] = torch.zeros(2, 8)
    state["q_adapter.k_B"] = torch.zeros(1, 2)
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

    assert loaded
    assert iteration is None
    optimizer.load_state_dict.assert_not_called()
