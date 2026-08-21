"""Truth tables for Tinker protocol mode and the Multi-LoRA executor,
plus launch rejection of Tinker mode without adapter slots."""

from types import SimpleNamespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import pytest

from miles.utils.multi_lora import uses_multi_lora_operation_executor, validate_multi_lora_args
from miles.utils.tinker import is_tinker_enabled, uses_explicit_training_operations, validate_tinker_args


def _args(tinker_backend: bool, n_adapters: int) -> SimpleNamespace:
    return SimpleNamespace(
        tinker_backend=tinker_backend,
        multi_lora_n_adapters=n_adapters,
        multi_lora=n_adapters > 0,
    )


class TestPredicateRoles:
    def test_operation_semantics_is_the_protocol_flag_alone(self):
        assert uses_explicit_training_operations(_args(True, 0))
        assert uses_explicit_training_operations(_args(True, 4))
        assert not uses_explicit_training_operations(_args(False, 4))
        assert not uses_explicit_training_operations(_args(False, 0))

    def test_executor_requires_protocol_and_slots(self):
        assert uses_multi_lora_operation_executor(_args(True, 4))
        assert not uses_multi_lora_operation_executor(_args(True, 0))
        assert not uses_multi_lora_operation_executor(_args(False, 4))

    def test_is_tinker_enabled_is_unchanged(self):
        """Characterization: the legacy predicate keeps its exact truth table."""
        for tinker, n in [(True, 4), (True, 0), (False, 4), (False, 0)]:
            assert is_tinker_enabled(_args(tinker, n)) == (tinker and n > 0)


class TestValidationClosesTheGap:
    """Every flag combination either fails validation or makes the protocol
    predicate equal to the multi-LoRA one — so swapping the train_one_step
    policy gate cannot change any launched run."""

    def _validate(self, args) -> None:
        validate_multi_lora_args(args)
        validate_tinker_args(args)

    def test_tinker_without_slots_is_rejected(self):
        with pytest.raises(AssertionError, match="--multi-lora-n-adapters"):
            self._validate(_args(True, 0))
