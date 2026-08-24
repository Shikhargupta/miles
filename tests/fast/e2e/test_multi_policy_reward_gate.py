from pathlib import Path

import pytest
from tests.e2e import conftest_multi_policy
from tests.e2e.conftest_multi_policy import TrainRewardBounds, _compute_reward_window_means


class TestComputeRewardWindowMeans:
    def test_the_early_value_averages_the_first_three_steps(self):
        """One noisy first rollout cannot decide whether training began unsolved."""
        windows = _compute_reward_window_means([0.9, 0.0, 0.3, 0.6, 0.9, 0.9])

        assert windows.initial == pytest.approx(0.4)

    def test_the_final_value_averages_the_last_third(self):
        """The learning gate measures a sustained tail rather than one lucky final rollout."""
        windows = _compute_reward_window_means([0.0, 0.1, 0.2, 0.3, 0.8, 1.0])

        assert windows.final == pytest.approx(0.9)

    def test_fewer_than_three_points_cannot_define_the_early_window(self):
        """A shortened run must not silently weaken the three-step early baseline."""
        with pytest.raises(AssertionError, match="at least three raw reward points"):
            _compute_reward_window_means([0.1, 0.2])


class TestAssertEveryPolicyReportedRewardInBounds:
    def test_default_growth_bound_allows_a_short_noisy_run_to_decline(self, monkeypatch):
        """A reward gate without a growth requirement accepts a noisy decline."""
        monkeypatch.setattr(
            conftest_multi_policy, "_read_train_reward_series", lambda *_args, **_kwargs: [0.5, 0.4, 0.3]
        )

        conftest_multi_policy.assert_every_policy_reported_reward_in_bounds(
            Path("unused"),
            bounds={"solver": TrainRewardBounds(initial_max=0.9, final_min=0.01)},
        )

    def test_explicit_growth_bound_rejects_insufficient_improvement(self, monkeypatch):
        """An explicit growth requirement still rejects insufficient improvement."""
        monkeypatch.setattr(
            conftest_multi_policy, "_read_train_reward_series", lambda *_args, **_kwargs: [0.2, 0.2, 0.2]
        )

        with pytest.raises(AssertionError, match="raw reward grew by 0.0, below 0.2"):
            conftest_multi_policy.assert_every_policy_reported_reward_in_bounds(
                Path("unused"),
                bounds={"solver": TrainRewardBounds(initial_max=0.9, final_min=0.01, min_growth=0.2)},
            )
