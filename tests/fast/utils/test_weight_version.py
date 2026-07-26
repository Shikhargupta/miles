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
        assert (
            try_parse_num_trained_rollouts(WeightVersion(run_uuid=RUN_UUID, num_trained_rollouts=7).serialize()) == 7
        )
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
        assert_samples_weight_version_sane(_make_args(), [_make_sample([(RUN_UUID, 5)])], rollout_id=5, is_eval=False)

    def test_one_rollout_of_lag_is_allowed(self):
        """Async drivers launch the next generation before the update, so it runs on the previous weights."""
        assert_samples_weight_version_sane(_make_args(), [_make_sample([(RUN_UUID, 4)])], rollout_id=5, is_eval=False)

    def test_an_unbounded_backlog_is_allowed_when_no_bound_was_declared(self):
        """Fully async runs legitimately train on a deep backlog; only an explicit bound may reject it."""
        assert_samples_weight_version_sane(_make_args(), [_make_sample([(RUN_UUID, 1)])], rollout_id=5, is_eval=False)

    def test_training_batch_from_a_future_rollout_fails(self):
        """Outside eval, weights published for a later rollout cannot have produced this batch."""
        with pytest.raises(AssertionError, match="newer than the rollout"):
            assert_samples_weight_version_sane(
                _make_args(), [_make_sample([(RUN_UUID, 6)])], rollout_id=5, is_eval=False
            )

    def test_rollout_functions_that_never_query_an_engine_are_left_alone(self):
        """SFT and hand-assembled rollouts carry no versions at all and must not be rejected."""
        assert_samples_weight_version_sane(
            _make_args(), [_unstamped_sample(), _unstamped_sample(1)], rollout_id=5, is_eval=False
        )

    def test_losing_the_version_on_only_some_samples_fails(self):
        """A batch where attribution went missing part-way is the case worth catching."""
        samples = [_make_sample([(RUN_UUID, 5)]), _unstamped_sample(1)]
        with pytest.raises(AssertionError, match="while other samples"):
            assert_samples_weight_version_sane(_make_args(), samples, rollout_id=5, is_eval=False)

    def test_empty_response_sample_needs_no_version(self):
        """A sample that generated nothing has nothing to attribute."""
        samples = [_make_sample([(RUN_UUID, 5)]), _unstamped_sample(1, response_length=0)]
        assert_samples_weight_version_sane(_make_args(), samples, rollout_id=5, is_eval=False)

    def test_staleness_is_measured_as_a_distance_not_a_version_count(self):
        """Two versions three rollouts apart are staler than two adjacent ones."""
        with pytest.raises(AssertionError, match="past"):
            assert_samples_weight_version_sane(
                _make_args(max_weight_staleness=1),
                [_make_sample([(RUN_UUID, 3), (RUN_UUID, 5)])],
                rollout_id=5,
                is_eval=False,
            )
        assert_samples_weight_version_sane(
            _make_args(max_weight_staleness=2),
            [_make_sample([(RUN_UUID, 3), (RUN_UUID, 5)])],
            rollout_id=5,
            is_eval=False,
        )

    def test_foreign_run_sample_fails_even_beside_a_current_run_sample(self):
        """A sample served entirely by another run's engine is rejected per sample."""
        samples = [_make_sample([(RUN_UUID, 5)]), _make_sample([(OTHER_RUN_UUID, 5)], index=1)]
        with pytest.raises(AssertionError, match="another run"):
            assert_samples_weight_version_sane(_make_args(), samples, rollout_id=5, is_eval=False)

    def test_foreign_run_versions_may_precede_current_run_versions(self):
        """A trajectory carried across a restart starts on the previous run's weights."""
        samples = [_make_sample([(OTHER_RUN_UUID, 2), (RUN_UUID, 5)])]
        assert_samples_weight_version_sane(_make_args(), samples, rollout_id=5, is_eval=False)

    def test_unparseable_version_fails(self):
        """Default-like engine versions such as 'default' are rejected."""
        sample = _make_sample([(RUN_UUID, 5)])
        sample.weight_versions[0].spans[0] = WeightVersionSpan("default", 0, 4)
        with pytest.raises(ValueError, match="invalid weight version"):
            assert_samples_weight_version_sane(_make_args(), [sample], rollout_id=5, is_eval=False)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("debug_rollout_only", True),
            ("debug_skip_weight_update", True),
            ("lora_rank", 32),
            ("lora_adapter_path", "/adapter"),
        ],
    )
    def test_modes_without_a_trustworthy_version_are_exempt(self, field, value):
        """Rollout-only, skipped updates and either way of enabling LoRA leave the version meaningless."""
        args = _make_args(max_weight_staleness=0)
        setattr(args, field, value)
        samples = [_make_sample([(RUN_UUID, 9)]), _make_sample([(OTHER_RUN_UUID, 1)], index=1)]

        assert_samples_weight_version_sane(args, samples, rollout_id=5, is_eval=False)

    def test_batched_weight_updates_do_not_count_as_staleness(self):
        """With interval > 1 the engines lag rollout_id by design; a uniform batch has zero spread."""
        args = _make_args(max_weight_staleness=0)
        args.update_weights_interval = 2
        assert_samples_weight_version_sane(args, [_make_sample([(RUN_UUID, 1)])], rollout_id=5, is_eval=False)

    def test_samples_without_an_index_are_each_still_checked(self):
        """Hand-built rollout functions leave every index None; they must not collapse onto one entry."""
        fresh, stale = _make_sample([(RUN_UUID, 5)]), _make_sample([(RUN_UUID, 2)])
        fresh.index = stale.index = None

        with pytest.raises(AssertionError, match="past max_weight_staleness"):
            assert_samples_weight_version_sane(_make_args(max_weight_staleness=1), [stale, fresh], rollout_id=9)

    def test_staleness_is_measured_against_the_batch_not_the_rollout_id(self):
        """A deep but uniform backlog is not stale; what matters is how far apart the samples are."""
        samples = [_make_sample([(RUN_UUID, 5)]), _make_sample([(RUN_UUID, 4)], index=1)]
        assert_samples_weight_version_sane(_make_args(max_weight_staleness=1), samples, rollout_id=9, is_eval=False)

        stale = [_make_sample([(RUN_UUID, 5)]), _make_sample([(RUN_UUID, 2)], index=1)]
        with pytest.raises(AssertionError, match="past max_weight_staleness"):
            assert_samples_weight_version_sane(_make_args(max_weight_staleness=1), stale, rollout_id=9, is_eval=False)

    def test_empty_batch_passes(self):
        """An empty sample list is trivially sane."""
        assert_samples_weight_version_sane(_make_args(), [], rollout_id=5, is_eval=False)


class TestEvalRunsOnExactlyOneWeightVersion:
    def test_two_samples_on_different_versions_fail(self):
        """Scores from different weights are not comparable, so they cannot be averaged into one number."""
        samples = [_make_sample([(RUN_UUID, 6)]), _make_sample([(RUN_UUID, 5)], index=1)]

        with pytest.raises(AssertionError, match="eval batch spans 2 weight versions"):
            assert_samples_weight_version_sane(_make_args(), samples, rollout_id=5, is_eval=True)

    def test_one_sample_straddling_a_weight_swap_fails(self):
        """A multi-turn eval sample that crossed an update measured neither checkpoint."""
        samples = [_make_sample([(RUN_UUID, 5), (RUN_UUID, 6)])]

        with pytest.raises(AssertionError, match="eval batch spans 2 weight versions"):
            assert_samples_weight_version_sane(_make_args(), samples, rollout_id=5, is_eval=True)

    def test_the_same_version_repeated_across_turns_and_samples_passes(self):
        """Multi-turn eval is fine as long as every turn saw the same weights."""
        samples = [_make_sample([(RUN_UUID, 6), (RUN_UUID, 6)]), _make_sample([(RUN_UUID, 6)], index=1)]

        assert_samples_weight_version_sane(_make_args(), samples, rollout_id=5, is_eval=True)

    def test_eval_may_run_on_weights_newer_than_the_rollout_being_evaluated(self):
        """Both drivers publish the next rollout's weights before evaluating this one."""
        assert_samples_weight_version_sane(_make_args(), [_make_sample([(RUN_UUID, 6)])], rollout_id=5, is_eval=True)

    def test_a_single_version_eval_batch_passes_whether_ahead_or_behind(self):
        """The rule is one version, not a particular one; the driver decides which weights eval sees."""
        for version_rollout_id in (4, 5, 6):
            assert_samples_weight_version_sane(
                _make_args(), [_make_sample([(RUN_UUID, version_rollout_id)])], rollout_id=5, is_eval=True
            )

    def test_unstamped_eval_samples_are_still_left_alone(self):
        """An eval rollout function that never queries an engine reports no versions to disagree about."""
        assert_samples_weight_version_sane(
            _make_args(), [_unstamped_sample(), _unstamped_sample(1)], rollout_id=5, is_eval=True
        )

    def test_training_batches_may_still_span_versions(self):
        """The one-version rule is an eval rule; training on a mixed batch is normal."""
        samples = [_make_sample([(RUN_UUID, 4)]), _make_sample([(RUN_UUID, 5)], index=1)]

        assert_samples_weight_version_sane(_make_args(), samples, rollout_id=5, is_eval=False)
