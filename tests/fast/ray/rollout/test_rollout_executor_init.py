from unittest.mock import MagicMock

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout.rollout_executor import RolloutExecutor


def _executor(*, init_called: bool) -> RolloutExecutor:
    executor = RolloutExecutor.__new__(RolloutExecutor)
    executor._init_called = init_called
    return executor


class TestInitRunsExactlyOnce:
    def test_a_constructed_executor_reports_itself_uninitialized(self):
        """The constructor the run really uses is what has to leave the flag clear."""
        executor = RolloutExecutor(
            args=make_args(debug_train_only=True),
            router_providers=[],
            session_server_provider=None,
            inference_controller_provider=MagicMock(),
        )

        assert executor.is_initialized() is False

    def test_an_executor_that_never_ran_init_reports_itself_uninitialized(self):
        """A restarted script waits out an executor that still answers as the previous script's."""
        assert _executor(init_called=False).is_initialized() is False

    def test_an_executor_that_ran_init_reports_itself_initialized(self):
        """The wait at the start of the rollout components only ends once this answer flips back."""
        assert _executor(init_called=True).is_initialized() is True

    async def test_a_second_init_is_refused(self):
        """An executor process the previous script initialized is about to be replaced, not re-initialized."""
        with pytest.raises(AssertionError):
            await _executor(init_called=True).init()
