"""Serving-only expansion for SGLang fused LoRA buffers."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from miles_plugins.lora.codec.hf import export_lora_hf_named
from miles_plugins.lora.codec.sglang import expand_sglang_target_modules, export_lora_sglang_named
from miles_plugins.lora.config import LoRAConfig
from miles_plugins.lora.modules.linear import LoRALinear, LoRASplitFC1, LoRASplitQKV
from miles_plugins.lora.spec.attention import MLA_KV_A_PROJECTION, MLA_Q_A_PROJECTION, QKV_PROJECTIONS
from miles_plugins.lora.spec.base import AttachContext
from miles_plugins.lora.spec.mlp import FC1_PROJECTIONS


def _context(*targets: str) -> AttachContext:
    return AttachContext(
        lora=LoRAConfig(rank=2, alpha=4, dropout=0.0, target_modules=frozenset(targets)),
        transformer_config=SimpleNamespace(
            hidden_size=8,
            layernorm_epsilon=1e-6,
            sequence_parallel=False,
            layernorm_zero_centered_gamma=False,
            attention_output_gate=False,
            q_lora_rank=5,
            kv_lora_rank=3,
            qk_pos_emb_head_dim=1,
        ),
        tp_size=1,
        tp_rank=0,
        layer_prefix="model.layers.",
        shared_expert="mlp.shared_expert.",
    )


def _qkv_model(*attrs: str) -> nn.Module:
    by_attr = {projection.attr: projection for projection in QKV_PROJECTIONS}
    model = nn.Module()
    model.adapter = LoRASplitQKV(
        hf_prefix="model.layers.0.self_attn.",
        reference=torch.empty(4, 8),
        context=_context(*(by_attr[attr].hf for attr in attrs)),
        projections=tuple(by_attr[attr] for attr in attrs),
        num_q=2,
        num_kv=1,
        head_dim=1,
    )
    return model


def _fc1_model(*attrs: str) -> nn.Module:
    by_attr = {projection.attr: projection for projection in FC1_PROJECTIONS}
    model = nn.Module()
    model.adapter = LoRASplitFC1(
        hf_prefix="model.layers.0.mlp.",
        reference=torch.empty(16, 8),
        context=_context(*(by_attr[attr].hf for attr in attrs)),
        projections=tuple(by_attr[attr] for attr in attrs),
        inter_local=8,
    )
    return model


def _mla_a_model(*hf_names: str) -> nn.Module:
    by_hf = {
        MLA_Q_A_PROJECTION.hf: (MLA_Q_A_PROJECTION, 5),
        MLA_KV_A_PROJECTION.hf: (MLA_KV_A_PROJECTION, 4),
    }
    model = nn.Module()
    context = _context(*hf_names)
    for hf_name in hf_names:
        projection, rows = by_hf[hf_name]
        model.add_module(
            projection.attr + "_" + hf_name,
            LoRALinear(
                hf_prefix="model.layers.0.self_attn.",
                projection=projection,
                reference=torch.empty(rows, 8),
                context=context,
                in_features=8,
                out_features=rows,
            ),
        )
    return model


@pytest.mark.parametrize(
    "active",
    [("q",), ("k",), ("v",), ("q", "k"), ("q", "v"), ("k", "v")],
)
def test_sglang_export_zero_pads_every_partial_qkv_subset(active):
    model = _qkv_model(*active)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(3)

    exact = dict(export_lora_hf_named([model]))
    serving = dict(export_lora_sglang_named([model]))

    assert {name.split(".")[-3] for name in serving} == {"q_proj", "k_proj", "v_proj"}
    assert set(exact) < set(serving)
    for attr, hf_name, rows in (("q", "q_proj", 2), ("k", "k_proj", 1), ("v", "v_proj", 1)):
        a = serving[f"model.layers.0.self_attn.{hf_name}.lora_A.weight"]
        b = serving[f"model.layers.0.self_attn.{hf_name}.lora_B.weight"]
        assert a.shape == (2, 8)
        assert b.shape == (rows, 2)
        expected = 3 if attr in active else 0
        assert torch.all(a == expected)
        assert torch.all(b == expected)


@pytest.mark.parametrize("active", [("gate",), ("up",)])
def test_sglang_export_zero_pads_partial_gate_up(active):
    model = _fc1_model(*active)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(5)

    serving = dict(export_lora_sglang_named([model]))

    assert {name.split(".")[-3] for name in serving} == {"gate_proj", "up_proj"}
    for attr, hf_name in (("gate", "gate_proj"), ("up", "up_proj")):
        expected = 5 if attr in active else 0
        assert torch.all(serving[f"model.layers.0.mlp.{hf_name}.lora_A.weight"] == expected)
        assert torch.all(serving[f"model.layers.0.mlp.{hf_name}.lora_B.weight"] == expected)


@pytest.mark.parametrize("active", [("q_a_proj",), ("kv_a_proj_with_mqa",)])
def test_sglang_export_zero_pads_partial_mla_fused_down_projection(active):
    model = _mla_a_model(*active)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(7)

    serving = dict(export_lora_sglang_named([model]))

    for hf_name, rows in (("q_a_proj", 5), ("kv_a_proj_with_mqa", 4)):
        expected = 7 if hf_name in active else 0
        assert serving[f"model.layers.0.self_attn.{hf_name}.lora_A.weight"].shape == (2, 8)
        assert serving[f"model.layers.0.self_attn.{hf_name}.lora_B.weight"].shape == (rows, 2)
        assert torch.all(serving[f"model.layers.0.self_attn.{hf_name}.lora_A.weight"] == expected)
        assert torch.all(serving[f"model.layers.0.self_attn.{hf_name}.lora_B.weight"] == expected)


def test_sglang_target_expansion_preserves_order_and_adds_fused_siblings():
    assert expand_sglang_target_modules(["o_proj", "k_proj", "up_proj", "kv_a_proj_with_mqa"]) == [
        "o_proj",
        "k_proj",
        "up_proj",
        "kv_a_proj_with_mqa",
        "q_proj",
        "v_proj",
        "gate_proj",
        "q_a_proj",
    ]
