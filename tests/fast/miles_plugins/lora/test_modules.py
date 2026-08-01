"""CPU tests for concrete native-LoRA linear adapter modules."""

from types import SimpleNamespace

import pytest
import torch

from miles_plugins.lora.config import LoRAConfig
from miles_plugins.lora.modules.linear import LoRALinear, LoRASplitFC1, LoRASplitQKV
from miles_plugins.lora.spec.attention import QKV_PROJECTIONS
from miles_plugins.lora.spec.base import COLUMN, REPLICATED, ROW, AttachContext, ProjectionSpec
from miles_plugins.lora.spec.mlp import FC1_PROJECTIONS


def _context(*targets: str, sequence_parallel: bool = False) -> AttachContext:
    transformer_config = SimpleNamespace(
        hidden_size=8,
        layernorm_epsilon=1e-6,
        sequence_parallel=sequence_parallel,
        layernorm_zero_centered_gamma=False,
        attention_output_gate=False,
    )
    return AttachContext(
        lora=LoRAConfig(
            rank=2,
            alpha=4,
            dropout=0.0,
            target_modules=frozenset(targets),
        ),
        transformer_config=transformer_config,
        tp_size=1,
        tp_rank=0,
        layer_prefix="model.layers.",
        shared_expert="mlp.shared_expert.",
    )


def test_split_qkv_registers_only_requested_logical_projection():
    adapter = LoRASplitQKV(
        hf_prefix="model.layers.0.self_attn.",
        reference=torch.empty(8, 8),
        context=_context("q_proj"),
        projections=QKV_PROJECTIONS[:1],
        num_q=4,
        num_kv=2,
        head_dim=1,
    )
    assert {projection.hf for projection in adapter.projection_specs} == {"q_proj"}
    assert set(dict(adapter.named_parameters())) == {"q_A", "q_B"}


def test_split_qkv_leaves_unrequested_kv_delta_zero():
    adapter = LoRASplitQKV(
        hf_prefix="model.layers.0.self_attn.",
        reference=torch.empty(4, 8),
        context=_context("q_proj"),
        projections=QKV_PROJECTIONS[:1],
        num_q=2,
        num_kv=1,
        head_dim=1,
    )
    with torch.no_grad():
        adapter.q_A.fill_(1.0)
        adapter.q_B.fill_(1.0)
    base_module = torch.nn.Linear(8, 4, bias=False)
    delta = adapter(torch.ones(1, 8), base_module)
    assert torch.count_nonzero(delta[..., :2]) == 2
    assert torch.count_nonzero(delta[..., 2:]) == 0


def test_split_fc1_registers_only_requested_logical_projection():
    adapter = LoRASplitFC1(
        hf_prefix="model.layers.0.mlp.",
        reference=torch.empty(16, 8),
        context=_context("gate_proj"),
        projections=FC1_PROJECTIONS[:1],
        inter_local=8,
    )
    assert {projection.hf for projection in adapter.projection_specs} == {"gate_proj"}
    assert set(dict(adapter.named_parameters())) == {"gate_A", "gate_B"}


def test_split_fc1_leaves_unrequested_up_delta_zero():
    adapter = LoRASplitFC1(
        hf_prefix="model.layers.0.mlp.",
        reference=torch.empty(16, 8),
        context=_context("gate_proj"),
        projections=FC1_PROJECTIONS[:1],
        inter_local=8,
    )
    with torch.no_grad():
        adapter.gate_A.fill_(1.0)
        adapter.gate_B.fill_(1.0)
    base_module = torch.nn.Linear(8, 16, bias=False)
    delta = adapter(torch.ones(1, 8), base_module)
    assert torch.count_nonzero(delta[..., :8]) == 8
    assert torch.count_nonzero(delta[..., 8:]) == 0


def test_split_fc1_is_a_callable_module_and_preserves_fused_output_shape():
    adapter = LoRASplitFC1(
        hf_prefix="model.layers.0.mlp.",
        reference=torch.empty(16, 8),
        context=_context("gate_proj", "up_proj"),
        projections=FC1_PROJECTIONS,
        inter_local=8,
    )
    base_module = torch.nn.Linear(8, 16, bias=False)
    delta = adapter(torch.randn(3, 8), base_module)
    assert delta.shape == (3, 16)


def test_split_modules_canonicalize_projection_execution_order():
    qkv = LoRASplitQKV(
        hf_prefix="model.layers.0.self_attn.",
        reference=torch.empty(4, 8),
        context=_context("q_proj", "v_proj"),
        projections=(QKV_PROJECTIONS[2], QKV_PROJECTIONS[0]),
        num_q=2,
        num_kv=1,
        head_dim=1,
    )
    fc1 = LoRASplitFC1(
        hf_prefix="model.layers.0.mlp.",
        reference=torch.empty(16, 8),
        context=_context("gate_proj", "up_proj"),
        projections=tuple(reversed(FC1_PROJECTIONS)),
        inter_local=8,
    )
    assert [projection.attr for projection in qkv.projection_specs] == ["q", "v"]
    assert [projection.attr for projection in fc1.projection_specs] == ["gate", "up"]


@pytest.mark.parametrize(
    ("layout", "a_metadata", "b_metadata"),
    [
        (COLUMN, (False, -1), (True, 0)),
        (ROW, (True, 1), (False, -1)),
        (REPLICATED, (False, -1), (False, -1)),
    ],
)
def test_lora_linear_marks_only_tp_sharded_parameter(layout, a_metadata, b_metadata):
    adapter = LoRALinear(
        hf_prefix="model.layers.0.projection.",
        projection=ProjectionSpec("proj", "proj", layout),
        reference=torch.empty(8, 8),
        context=_context("proj"),
        in_features=8,
        out_features=8,
    )

    assert (adapter.proj_A.tensor_model_parallel, adapter.proj_A.partition_dim) == a_metadata
    assert (adapter.proj_B.tensor_model_parallel, adapter.proj_B.partition_dim) == b_metadata
    assert adapter.proj_A.partition_stride == 1
    assert adapter.proj_B.partition_stride == 1


@pytest.mark.parametrize(
    ("layout", "a_grad_group", "b_grad_group"),
    [
        (COLUMN, "tp", None),
        (ROW, None, "tp"),
        (REPLICATED, "tp", "tp"),
    ],
)
def test_lora_linear_preserves_sequence_parallel_grad_sum_groups(layout, a_grad_group, b_grad_group):
    adapter = LoRALinear(
        hf_prefix="model.layers.0.projection.",
        projection=ProjectionSpec("proj", "proj", layout),
        reference=torch.empty(8, 8),
        context=_context("proj", sequence_parallel=True),
        in_features=8,
        out_features=8,
    )

    assert getattr(adapter.proj_A, "_lora_grad_sum_group", None) == a_grad_group
    assert getattr(adapter.proj_B, "_lora_grad_sum_group", None) == b_grad_group


def test_split_qkv_marks_only_output_factors_tp_sharded():
    adapter = LoRASplitQKV(
        hf_prefix="model.layers.0.self_attn.",
        reference=torch.empty(4, 8),
        context=_context("q_proj", "v_proj"),
        projections=(QKV_PROJECTIONS[0], QKV_PROJECTIONS[2]),
        num_q=2,
        num_kv=1,
        head_dim=1,
    )

    for name in ("q", "v"):
        a = getattr(adapter, f"{name}_A")
        b = getattr(adapter, f"{name}_B")
        assert (a.tensor_model_parallel, a.partition_dim) == (False, -1)
        assert (b.tensor_model_parallel, b.partition_dim) == (True, 0)


def test_split_fc1_marks_only_output_factors_tp_sharded():
    adapter = LoRASplitFC1(
        hf_prefix="model.layers.0.mlp.",
        reference=torch.empty(16, 8),
        context=_context("gate_proj", "up_proj"),
        projections=FC1_PROJECTIONS,
        inter_local=8,
    )

    for name in ("gate", "up"):
        a = getattr(adapter, f"{name}_A")
        b = getattr(adapter, f"{name}_B")
        assert (a.tensor_model_parallel, a.partition_dim) == (False, -1)
        assert (b.tensor_model_parallel, b.partition_dim) == (True, 0)
