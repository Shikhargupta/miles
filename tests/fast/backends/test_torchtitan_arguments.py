"""The torchtitan backend's argument gate.

These rejections are load-bearing rather than cosmetic: with the causal-only
sdpa attention backend, a microbatch holding more than one document would let
tokens attend across the document boundary and train on silently wrong
attention. Failing at startup is the difference between a clear error and a run
whose numbers are quietly meaningless.
"""

from argparse import Namespace

import pytest

from miles.backends.fsdp_utils.arguments import FSDPArgs
from miles.backends.torchtitan_utils.arguments import TorchtitanArgs, validate_torchtitan_args


def _args(**overrides) -> Namespace:
    base = dict(
        titan_attn_backend="sdpa",
        titan_pp_size=1,
        use_dynamic_batch_size=False,
        micro_batch_size=1,
        kl_coef=0.0,
        use_kl_loss=False,
        save_debug_train_data=None,
    )
    return Namespace(**{**base, **overrides})


def test_the_supported_configuration_passes():
    validate_torchtitan_args(_args())


@pytest.mark.parametrize("backend", ["flex", "flex_flash", "varlen"])
def test_attention_backends_needing_a_newer_torch_are_rejected(backend):
    with pytest.raises(ValueError, match="torch>=2.12"):
        validate_torchtitan_args(_args(titan_attn_backend=backend))


def test_dynamic_batching_is_rejected():
    with pytest.raises(ValueError, match="causal-only"):
        validate_torchtitan_args(_args(use_dynamic_batch_size=True))


def test_multi_document_microbatches_are_rejected():
    with pytest.raises(ValueError, match="causal-only"):
        validate_torchtitan_args(_args(micro_batch_size=4))


def test_pipeline_parallelism_is_rejected():
    with pytest.raises(ValueError, match="titan-pp-size"):
        validate_torchtitan_args(_args(titan_pp_size=2))


def test_kl_options_are_rejected_while_there_is_no_reference_model():
    """placement_group.py builds a ref model iff kl_coef != 0 or use_kl_loss, and the
    torchtitan actor has none, so accepting either would train a different objective."""
    with pytest.raises(ValueError, match="reference model"):
        validate_torchtitan_args(_args(use_kl_loss=True))
    with pytest.raises(ValueError, match="reference model"):
        validate_torchtitan_args(_args(kl_coef=0.01))


def test_debug_train_dump_is_rejected_rather_than_ignored():
    with pytest.raises(ValueError, match="save-debug-train-data"):
        validate_torchtitan_args(_args(save_debug_train_data="/tmp/dump"))


def test_args_extend_rather_than_restate_the_common_training_options():
    """FSDPArgs holds the non-Megatron training options (optimizer, LR schedule,
    precision); torchtitan inherits them so the two cannot drift apart."""
    assert issubclass(TorchtitanArgs, FSDPArgs)
    common = {"lr", "adam_beta1", "adam_beta2", "adam_eps", "weight_decay", "fp16"}
    assert common <= set(TorchtitanArgs.__dataclass_fields__)


def test_titan_specific_fields_are_namespaced():
    titan_only = set(TorchtitanArgs.__dataclass_fields__) - set(FSDPArgs.__dataclass_fields__)
    assert titan_only and all(name.startswith("titan_") for name in titan_only)
