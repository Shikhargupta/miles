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
        ref_update_interval=None,
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


def test_periodic_reference_refresh_is_rejected_rather_than_ignored():
    """The reference model is built once from --ref-load; the actor-to-ref copy FSDP
    does on an interval is not wired up, and ignoring the flag would train against a
    reference the user believes is being refreshed."""
    with pytest.raises(ValueError, match="ref-update-interval"):
        validate_torchtitan_args(_args(ref_update_interval=4))


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
