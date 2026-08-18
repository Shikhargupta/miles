from argparse import Namespace

import pytest

from miles.ray.rollout.rollout_executor import RolloutExecutor


class _RecordingDataSource:
    def __init__(self, *, loaded: bool) -> None:
        self._loaded = loaded
        self.rollout_ids: list[int | None] = []

    def load(self, rollout_id: int | None = None) -> bool:
        self.rollout_ids.append(rollout_id)
        return self._loaded


def _executor(*, loaded: bool) -> RolloutExecutor:
    executor = RolloutExecutor.__new__(RolloutExecutor)
    executor.args = Namespace()
    executor.data_source = _RecordingDataSource(loaded=loaded)
    executor.use_experimental_refactor = False
    return executor


class TestLoadingTheRolloutState:
    def test_a_take_over_refuses_a_load_that_found_no_state(self):
        """Loading nothing here restarts the dataset at a position the trainers already passed."""
        executor = _executor(loaded=False)

        with pytest.raises(AssertionError):
            executor.load(50, require_state=True)

    def test_a_take_over_accepts_a_load_that_found_its_state(self):
        """This is the ordinary take-over, where the previous script saved the rollout state it was asked to."""
        executor = _executor(loaded=True)

        executor.load(50, require_state=True)

        assert executor.data_source.rollout_ids == [50]

    def test_a_cold_start_loads_whatever_is_there(self):
        """A run being built has no position to insist on, and a fresh --save legitimately holds nothing."""
        executor = _executor(loaded=False)

        executor.load(50)

        assert executor.data_source.rollout_ids == [50]
