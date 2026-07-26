from __future__ import annotations

from unittest.mock import patch

import pytest
from tests.fast.ray.rollout.conftest import make_args, make_sample, make_samples_grouped, make_test_weight_version

from miles.ray.rollout.metrics import (
    _compute_metrics_from_samples,
    _compute_passrate_from_samples,
    _compute_zero_std_metrics,
    log_eval_rollout_data,
    log_rollout_data,
)
from miles.utils.tracking_utils import tracking
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall


class TestComputeZeroStdMetrics:
    def test_returns_empty_for_ppo_regardless_of_reward_distribution(self):
        args = make_args(advantage_estimator="ppo")
        out = _compute_zero_std_metrics(args, make_samples_grouped(2, 4, rewards=[1.0] * 8))
        assert out == {}

    def test_grpo_mixed_rewards_yield_zero_percentages_and_no_buckets(self):
        """Happy path: every group has reward variation → no group is zero-std →
        no bucket counts; the all_zero/all_one percentages are 0."""
        args = make_args(advantage_estimator="grpo", reward_key=None)
        samples = make_samples_grouped(2, 4, rewards=[0.0, 0.5, 1.0, 0.7, 0.2, 0.8, 0.3, 0.6])
        out = _compute_zero_std_metrics(args, samples)
        assert out == {"zero_std/all_zero_percentage": 0.0, "zero_std/all_one_percentage": 0.0}

    def test_grpo_zero_std_groups_produce_bucket_counts_and_percentages(self):
        """1 group all-1, 1 group all-0, 1 group mixed → bucket counts plus the
        all_zero/all_one percentages over total groups."""
        args = make_args(advantage_estimator="grpo", reward_key=None)
        samples = make_samples_grouped(3, 4, rewards=[1.0] * 4 + [0.0] * 4 + [0.0, 1.0, 0.0, 1.0])
        out = _compute_zero_std_metrics(args, samples)
        assert out["zero_std/count_1.0"] == 1
        assert out["zero_std/count_0.0"] == 1
        assert out["zero_std/all_zero_percentage"] == pytest.approx(1 / 3)
        assert out["zero_std/all_one_percentage"] == pytest.approx(1 / 3)

    def test_grpo_uniform_non_binary_reward_gets_its_own_bucket(self):
        """Every group zero-std at reward=0.5 → bucket count_0.5=2, but
        all_zero/all_one percentages stay 0 because they only count 0.0 and 1.0."""
        args = make_args(advantage_estimator="grpo", reward_key=None)
        samples = make_samples_grouped(2, 4, rewards=[0.5] * 8)
        out = _compute_zero_std_metrics(args, samples)
        assert out["zero_std/count_0.5"] == 2
        assert out["zero_std/all_zero_percentage"] == 0.0
        assert out["zero_std/all_one_percentage"] == 0.0

    def test_empty_samples_does_not_crash(self):
        args = make_args(advantage_estimator="grpo", reward_key=None)
        out = _compute_zero_std_metrics(args, [])
        # No groups → no all_zero/all_one keys (the function guards on total_groups>0).
        assert "zero_std/all_zero_percentage" not in out
        assert "zero_std/all_one_percentage" not in out


class TestTitoMismatchMetrics:
    def test_no_tito_metadata_emits_no_tito_keys(self):
        args = make_args(advantage_estimator="ppo", ci_test=False, log_passrate=False)
        samples = make_samples_grouped(1, 4)
        out = _compute_metrics_from_samples(args, samples, rollout_id=0)
        assert "tito_session_mismatch_rate" not in out

    def test_clean_tito_metadata_yields_zero_rates_per_mismatch_type(self):
        args = make_args(advantage_estimator="ppo", ci_test=False, log_passrate=False)
        samples = make_samples_grouped(1, 4)
        for s in samples:
            s.metadata = {"tito_session_mismatch": []}
        out = _compute_metrics_from_samples(args, samples, rollout_id=0)
        assert out["tito_session_mismatch_rate"] == 0.0
        for mtype in ("special_token_count", "special_token_type", "non_assistant_text", "assistant_text"):
            assert out[f"tito_session_mismatch_rate/{mtype}"] == 0.0

    def test_strict_mismatch_raises_under_ci_test(self):
        """Under ci_test=True, a non-zero rate on the strict mismatch types
        (special_token_count / special_token_type / non_assistant_text) must
        hard-fail — these signal a TITO algorithm or chat-template bug."""
        args = make_args(advantage_estimator="ppo", ci_test=True, log_passrate=False)
        samples = make_samples_grouped(1, 4)
        samples[0].metadata = {"tito_session_mismatch": [{"type": "special_token_count"}]}
        for s in samples[1:]:
            s.metadata = {"tito_session_mismatch": []}
        with pytest.raises(AssertionError, match="special_token_count"):
            _compute_metrics_from_samples(args, samples, rollout_id=0)

    def test_assistant_text_mismatch_does_not_raise_under_ci_test(self):
        """assistant_text mismatch is non-critical (tokens inherited from the
        pretokenized prefix) — even under ci_test, must not raise."""
        args = make_args(advantage_estimator="ppo", ci_test=True, log_passrate=False)
        samples = make_samples_grouped(1, 4)
        samples[0].metadata = {"tito_session_mismatch": [{"type": "assistant_text"}]}
        for s in samples[1:]:
            s.metadata = {"tito_session_mismatch": []}
        out = _compute_metrics_from_samples(args, samples, rollout_id=0)
        assert out["tito_session_mismatch_rate/assistant_text"] > 0


class TestComputePassrateFromSamples:
    def test_returns_empty_when_group_size_is_one(self):
        args = make_args(n_samples_per_prompt=1)
        samples = make_samples_grouped(4, 1, rewards=[1.0, 0.0, 1.0, 0.0])

        assert _compute_passrate_from_samples(args, samples) == {}

    @pytest.mark.parametrize("reward, expected", [(1.0, 1.0), (0.0, 0.0)])
    def test_uniform_rewards(self, reward, expected):
        args = make_args(n_samples_per_prompt=4, reward_key=None)
        samples = make_samples_grouped(2, 4, rewards=[reward] * 8)

        out = _compute_passrate_from_samples(args, samples)

        assert out == {
            "pass@1": pytest.approx(expected),
            "pass@2": pytest.approx(expected),
            "pass@4": pytest.approx(expected),
        }

    def test_mixed_rewards_pass_at_k_increases_with_k(self):
        args = make_args(n_samples_per_prompt=4, reward_key=None)
        rewards = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        samples = make_samples_grouped(2, 4, rewards=rewards)

        out = _compute_passrate_from_samples(args, samples)

        assert out["pass@1"] < out["pass@2"] < out["pass@4"]

    def test_excludes_incomplete_groups(self):
        args = make_args(n_samples_per_prompt=4, reward_key=None)
        samples = make_samples_grouped(2, 4, rewards=[1.0] * 4 + [0.0] * 4)
        samples.pop()

        out = _compute_passrate_from_samples(args, samples)

        assert out == {
            "pass@1": pytest.approx(1.0),
            "pass@2": pytest.approx(1.0),
            "pass@4": pytest.approx(1.0),
        }


class TestWeightVersionMetrics:
    def test_reports_oldest_version_statistics_and_mixed_ratio(self):
        """weight_version/* summarises each sample's oldest version; mixed counts samples spanning an update."""
        samples = [
            _make_versioned_sample([4], index=0),
            _make_versioned_sample([5, 6], index=1),
        ]

        out = _compute_metrics_from_samples(make_args(), samples, rollout_id=9)

        assert out["weight_version/min"] == 4
        assert out["weight_version/max"] == 5
        assert out["weight_version/mixed_version_ratio"] == 0.5

    def test_a_call_spanning_no_update_is_not_mixed(self):
        """Two calls that both saw the same version must not count as mixed."""
        samples = [_make_versioned_sample([7, 7], index=0)]

        out = _compute_metrics_from_samples(make_args(), samples, rollout_id=9)

        assert out["weight_version/mixed_version_ratio"] == 0.0

    def test_no_version_metrics_when_nothing_was_stamped(self):
        """SFT-style batches carry no versions and must not synthesise the series."""
        out = _compute_metrics_from_samples(make_args(), [make_sample(index=0, group_index=0)], rollout_id=9)

        assert not any(key.startswith("weight_version/") for key in out)


def _make_versioned_sample(rollout_ids: list[int], *, index: int) -> Sample:
    sample = make_sample(index=index, group_index=0)
    sample.weight_versions = [
        WeightVersionsPerCall(spans=[WeightVersionSpan(make_test_weight_version(rollout_id), i, i + 1)])
        for i, rollout_id in enumerate(rollout_ids)
    ]
    return sample


class TestWeightStalenessMetrics:
    def test_staleness_is_measured_from_each_sample_oldest_weights(self):
        """Staleness spreads over the batch from the oldest weights each sample was generated with."""
        samples = [
            _make_staleness_sample([5], index=0),
            _make_staleness_sample([3, 5], index=1),
            _make_staleness_sample([4], index=2),
        ]

        log_dict = _compute_metrics_from_samples(make_args(), samples, rollout_id=5)

        assert log_dict["weight_staleness/max"] == 2
        assert log_dict["weight_staleness/min"] == 0
        assert log_dict["weight_staleness/mean"] == 1.0

    def test_fully_on_policy_batch_reports_zero_staleness(self):
        """Every sample generated with this rollout's own weights is not stale at all."""
        samples = [_make_staleness_sample([7], index=0), _make_staleness_sample([7], index=1)]

        log_dict = _compute_metrics_from_samples(make_args(), samples, rollout_id=7)

        assert log_dict["weight_staleness/max"] == 0
        assert log_dict["weight_staleness/mean"] == 0.0

    def test_staleness_uses_the_oldest_weights_whatever_order_they_appear_in(self):
        """Taking the first span instead of the oldest would pass on ascending data alone."""
        ascending = _compute_metrics_from_samples(make_args(), [_make_staleness_sample([3, 5], index=0)], rollout_id=5)
        descending = _compute_metrics_from_samples(
            make_args(), [_make_staleness_sample([5, 3], index=0)], rollout_id=5
        )

        assert ascending["weight_staleness/max"] == 2
        assert descending["weight_staleness/max"] == 2

    def test_train_entry_passes_the_rollout_being_generated(self):
        """The staleness series is meaningless if the entry point hands over the wrong rollout."""
        samples = [_make_staleness_sample([4], index=0)]
        logged = {}

        with (
            patch.object(tracking, "log", lambda args, log_dict, step_key: logged.update(log_dict)),
            patch("miles.ray.rollout.metrics._compute_perf_metrics_from_samples", return_value={}),
        ):
            log_rollout_data(6, make_args(), samples, None, 1.0)

        assert logged["rollout/weight_staleness/max"] == 2

    def test_eval_reports_versions_but_no_staleness(self):
        """Eval runs on weights published for the next rollout, so the distance would go negative."""
        samples = [_make_staleness_sample([7], index=0)]
        logged = {}

        with patch.object(tracking, "log", lambda args, log_dict, step_key: logged.update(log_dict)):
            log_eval_rollout_data(6, make_args(), {"ds": {"rewards": [1.0], "samples": samples}})

        assert logged["eval/ds/weight_version/max"] == 7
        assert not any("weight_staleness/" in key for key in logged)

    def test_eval_on_newer_weights_never_reports_a_negative_distance(self):
        """The sync driver publishes rollout N+1's weights before evaluating rollout N."""
        samples = [_make_staleness_sample([7], index=0)]

        log_dict = _compute_metrics_from_samples(make_args(), samples, rollout_id=6, is_eval=True)

        assert not any(key.startswith("weight_staleness/") for key in log_dict)

    @pytest.mark.parametrize("field,value", [("lora_rank", 32), ("lora_adapter_path", "/adapter")])
    def test_no_staleness_metrics_under_lora(self, field, value):
        """A LoRA engine's version describes the frozen base, so the distance from it means nothing."""
        args = make_args(**{field: value})
        log_dict = _compute_metrics_from_samples(args, [_make_staleness_sample([5], index=0)], rollout_id=7)

        assert not any(key.startswith("weight_staleness/") for key in log_dict)
        assert log_dict["weight_version/min"] == 5

    def test_no_staleness_metrics_without_weight_versions(self):
        """Samples the engine never stamped contribute no staleness series."""
        log_dict = _compute_metrics_from_samples(make_args(), [_make_staleness_sample([], index=0)], rollout_id=5)

        assert not any(key.startswith("weight_staleness/") for key in log_dict)

    def test_a_call_that_produced_no_spans_is_not_a_version(self):
        """Real generation leaves an empty call behind rather than an empty version list."""
        sample = make_sample(index=0, group_index=0)
        sample.weight_versions = [WeightVersionsPerCall(spans=[])]

        log_dict = _compute_metrics_from_samples(make_args(), [sample], rollout_id=5)

        assert not any(key.startswith("weight_staleness/") for key in log_dict)
        assert not any(key.startswith("weight_version/") for key in log_dict)


def _make_staleness_sample(version_rollout_ids: list[int], *, index: int) -> Sample:
    calls = [
        WeightVersionsPerCall(spans=[WeightVersionSpan(make_test_weight_version(rollout_id), i, i + 1)])
        for i, rollout_id in enumerate(version_rollout_ids)
    ]
    sample = make_sample(index=index, group_index=0)
    sample.weight_versions = calls
    return sample
