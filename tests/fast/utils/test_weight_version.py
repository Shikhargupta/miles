from types import SimpleNamespace

import pytest

from miles.utils.run_uuid import generate_run_uuid
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall
from miles.utils.weight_version import (
    WeightVersion,
    assert_samples_weight_version_sane,
    try_parse_num_trained_rollouts,
)

RUN_UUID = "ab12cd34ef5678ab"
OTHER_RUN_UUID = "ffee0011223344ff"


class TestWeightVersion:
    def test_serialize_deserialize_roundtrip(self):
        """A serialized weight version decodes back to the same run uuid and rollout id."""
        version = WeightVersion(run_uuid=RUN_UUID, num_trained_rollouts=42)
        assert WeightVersion.deserialize(version.serialize()) == version

    def test_serializes_to_the_exact_wire_string(self):
        """Pinning the wire form catches a serializer and parser that drift together."""
        assert WeightVersion(run_uuid=RUN_UUID, num_trained_rollouts=7).serialize() == "ab12cd34ef5678ab-00000007"

    def test_the_launch_run_uuid_survives_a_roundtrip(self):
        """The weight version embeds whatever the run identity produced, so the two formats must agree."""
        run_uuid = generate_run_uuid()
        assert (
            WeightVersion.deserialize(WeightVersion(run_uuid=run_uuid, num_trained_rollouts=0).serialize()).run_uuid
            == run_uuid
        )

    @pytest.mark.parametrize(
        "run_uuid, rollout_id",
        [
            ("AB12CD34EF5678AB", 7),
            ("ab12cd34", 7),
            ("ab12cd34ef5678abcd", 7),
            ("ab12cd34ef5678ab", 10**8),
            ("ab12cd34ef5678ab", -1),
        ],
    )
    def test_serializing_an_unrepresentable_version_fails_loudly(self, run_uuid, rollout_id):
        """A version that cannot be read back would silently stamp samples with an unattributable string."""
        with pytest.raises((AssertionError, ValueError)):
            WeightVersion(run_uuid=run_uuid, num_trained_rollouts=rollout_id).serialize()

    @pytest.mark.parametrize("bad", ["default", "0", "1", "", "ab12cd34ef5678ab-1", "ab12cd34ef5678ab:00000001", None])
    def test_deserialize_rejects_legacy_and_malformed_versions(self, bad):
        """Bare counters, sglang defaults, and malformed strings are not valid weight versions."""
        with pytest.raises(ValueError, match="invalid weight version"):
            WeightVersion.deserialize(bad)

    def test_parse_rollout_id_rejects_anything_but_the_run_scoped_format(self):
        """A bare counter from an older run is not comparable to a run-scoped id, so it is not one."""
        assert try_parse_num_trained_rollouts(WeightVersion(run_uuid=RUN_UUID, num_trained_rollouts=7).serialize()) == 7
        assert try_parse_num_trained_rollouts("12") is None
        assert try_parse_num_trained_rollouts("default") is None


def _make_args(max_weight_staleness: int | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        run_uuid=RUN_UUID,
        max_weight_staleness=max_weight_staleness,
        debug_rollout_only=False,
        debug_skip_weight_update=False,
        update_weights_interval=1,
        lora_rank=0,
        lora_adapter_path=None,
    )


def _unstamped_sample(index: int = 0, response_length: int = 4) -> Sample:
    return Sample(index=index, tokens=list(range(response_length)), response_length=response_length)


def _make_sample(versions: list[tuple[str, int]], response_length: int = 4, index: int = 0) -> Sample:
    calls = [
        WeightVersionsPerCall(spans=[WeightVersionSpan(WeightVersion(run_uuid, rollout_id).serialize(), i, i + 1)])
        for i, (run_uuid, rollout_id) in enumerate(versions)
    ]
    return Sample(
        index=index,
        tokens=list(range(response_length)),
        response_length=response_length,
        weight_versions=calls,
    )


class TestAssertSamplesWeightVersionSane:
    def test_weights_served_for_this_rollout_pass(self):
        """A sample generated under the weights published for this rollout is sane."""
        assert_samples_weight_version_sane(_make_args(), [_make_sample([(RUN_UUID, 5)])], rollout_id=5)

    def test_one_rollout_of_lag_is_allowed(self):
        """Async drivers launch the next generation before the update, so it runs on the previous weights."""
        assert_samples_weight_version_sane(_make_args(), [_make_sample([(RUN_UUID, 4)])], rollout_id=5)

    def test_weights_stuck_far_behind_fail(self):
        """Weights that stopped reaching the engines fall further behind than any driver explains."""
        with pytest.raises(AssertionError, match="not reaching"):
            assert_samples_weight_version_sane(_make_args(), [_make_sample([(RUN_UUID, 1)])], rollout_id=5)

    def test_eval_may_run_on_newer_weights(self):
        """The sync driver publishes the next rollout's weights before evaluating this one."""
        assert_samples_weight_version_sane(_make_args(), [_make_sample([(RUN_UUID, 6)])], rollout_id=5, is_eval=True)

    def test_training_batch_from_a_future_rollout_fails(self):
        """Outside eval, weights published for a later rollout cannot have produced this batch."""
        with pytest.raises(AssertionError, match="newer than the rollout"):
            assert_samples_weight_version_sane(_make_args(), [_make_sample([(RUN_UUID, 6)])], rollout_id=5)

    def test_rollout_functions_that_never_query_an_engine_are_left_alone(self):
        """SFT and hand-assembled rollouts carry no versions at all and must not be rejected."""
        assert_samples_weight_version_sane(_make_args(), [_unstamped_sample(), _unstamped_sample(1)], rollout_id=5)

    def test_losing_the_version_on_only_some_samples_fails(self):
        """A batch where attribution went missing part-way is the case worth catching."""
        samples = [_make_sample([(RUN_UUID, 5)]), _unstamped_sample(1)]
        with pytest.raises(AssertionError, match="while other samples"):
            assert_samples_weight_version_sane(_make_args(), samples, rollout_id=5)

    def test_empty_response_sample_needs_no_version(self):
        """A sample that generated nothing has nothing to attribute."""
        samples = [_make_sample([(RUN_UUID, 5)]), _unstamped_sample(1, response_length=0)]
        assert_samples_weight_version_sane(_make_args(), samples, rollout_id=5)

    def test_staleness_is_measured_as_a_distance_not_a_version_count(self):
        """Two versions three rollouts apart are staler than two adjacent ones."""
        with pytest.raises(AssertionError, match="past"):
            assert_samples_weight_version_sane(
                _make_args(max_weight_staleness=1), [_make_sample([(RUN_UUID, 3), (RUN_UUID, 5)])], rollout_id=5
            )
        assert_samples_weight_version_sane(
            _make_args(max_weight_staleness=2), [_make_sample([(RUN_UUID, 3), (RUN_UUID, 5)])], rollout_id=5
        )

    def test_foreign_run_sample_fails_even_beside_a_current_run_sample(self):
        """A sample served entirely by another run's engine is rejected per sample."""
        samples = [_make_sample([(RUN_UUID, 5)]), _make_sample([(OTHER_RUN_UUID, 5)], index=1)]
        with pytest.raises(AssertionError, match="another run"):
            assert_samples_weight_version_sane(_make_args(), samples, rollout_id=5)

    def test_foreign_run_versions_may_precede_current_run_versions(self):
        """A trajectory carried across a restart starts on the previous run's weights."""
        samples = [_make_sample([(OTHER_RUN_UUID, 2), (RUN_UUID, 5)])]
        assert_samples_weight_version_sane(_make_args(), samples, rollout_id=5)

    def test_unparseable_version_fails(self):
        """Default-like engine versions such as 'default' are rejected."""
        sample = _make_sample([(RUN_UUID, 5)])
        sample.weight_versions[0].spans[0] = WeightVersionSpan("default", 0, 4)
        with pytest.raises(ValueError, match="invalid weight version"):
            assert_samples_weight_version_sane(_make_args(), [sample], rollout_id=5)

    def test_modes_without_a_trustworthy_version_are_exempt(self):
        """Rollout-only, skipped updates and LoRA all leave the engine version meaningless."""
        for field in ("debug_rollout_only", "debug_skip_weight_update"):
            args = _make_args()
            setattr(args, field, True)
            assert_samples_weight_version_sane(args, [_unstamped_sample()], rollout_id=5)
        lora = _make_args()
        lora.lora_rank = 32
        assert_samples_weight_version_sane(lora, [_unstamped_sample()], rollout_id=5)

    def test_update_weights_interval_relaxes_the_freshness_bound(self):
        """Batched weight updates leave rollouts on the previous version by design."""
        args = _make_args()
        args.update_weights_interval = 2
        assert_samples_weight_version_sane(args, [_make_sample([(RUN_UUID, 1)])], rollout_id=5)

    def test_empty_batch_passes(self):
        """An empty sample list is trivially sane."""
        assert_samples_weight_version_sane(_make_args(), [], rollout_id=5)
