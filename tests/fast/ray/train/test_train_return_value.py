from types import SimpleNamespace

import pytest
import ray
from tests.fast.ray.train.conftest import make_alive_cell

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.ray.train.group import RayTrainGroup

pytestmark = pytest.mark.asyncio

_DUMMY_DATA_PACK = {"data_ref": "data", "sample_indices": [0]}


def _make_group(cells: list) -> RayTrainGroup:
    group = object.__new__(RayTrainGroup)
    group._cells = cells
    group.args = SimpleNamespace(enable_event_analyzer=False, save_debug_event_data=None)
    group._witness_allocator = None
    group._indep_dp_quorum_id = 0
    group._health_checker_activeness = True
    group._test_action_executor = SimpleNamespace(run_after_step=lambda **kwargs: None)
    return group


class TestTrainReturnValue:
    async def test_one_result_per_worker_reaches_the_caller(self):
        """The critic values leave the group per worker so the driver can feed them to the actor."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        for handle in cell._get_actor_handles():
            ray.get(handle.set_train_return_value.remote({"train_step_outcome": TrainStepOutcome.NORMAL}))
        group = _make_group([cell])

        results = await group.train(3, _DUMMY_DATA_PACK)

        assert results == [{"train_step_outcome": TrainStepOutcome.NORMAL}] * 2

    async def test_results_of_several_cells_are_concatenated_in_cell_order(self):
        """Independent DP ranks are positional, so a reordered result list misroutes values."""
        cells = [make_alive_cell(index, alive_cell_indices=[0, 1]) for index in range(2)]
        for index, cell in enumerate(cells):
            for handle in cell._get_actor_handles():
                ray.get(handle.set_train_return_value.remote(index))
        group = _make_group(cells)

        results = await group.train(3, _DUMMY_DATA_PACK)

        assert results == [0, 0, 1, 1]

    async def test_a_failed_cell_contributes_no_result(self):
        """A raw exception object in the returned list would be fed straight into the next train call."""
        cells = [make_alive_cell(index, alive_cell_indices=[0, 1]) for index in range(2)]
        ray.get(cells[0]._get_actor_handles()[0].set_fail_methods.remote(["train"]))
        for handle in cells[1]._get_actor_handles():
            ray.get(handle.set_train_return_value.remote("ok"))
        group = _make_group(cells)

        results = await group.train(3, _DUMMY_DATA_PACK)

        assert results == ["ok", "ok"]


class TestRetryReturnsTheValue:
    async def test_the_value_of_the_successful_attempt_is_returned(self):
        """train() reads its result through retry, so retry must stop swallowing it."""
        from miles.utils.retry_utils import retry

        attempts = []

        async def _fn(attempt: int) -> str:
            attempts.append(attempt)
            if attempt == 0:
                raise RuntimeError("boom")
            return "second"

        async def _no_sleep(_seconds: float) -> None:
            return None

        assert await retry(_fn, sleep_fn=_no_sleep) == "second"
        assert attempts == [0, 1]


class TestWorkerResultShape:
    async def test_a_critic_dict_does_not_break_the_discarded_check(self):
        """The critic returns a dict per worker, which is not comparable to a TrainStepOutcome."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        for handle in cell._get_actor_handles():
            ray.get(handle.set_train_return_value.remote({"train_step_outcome": TrainStepOutcome.NORMAL}))
        group = _make_group([cell])

        await group.train(3, _DUMMY_DATA_PACK)

    async def test_a_discarded_outcome_inside_a_dict_is_still_seen(self):
        """Reading the outcome only from bare enums would silently skip the critic's retry request."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        results = [{"train_step_outcome": TrainStepOutcome.DISCARDED_SHOULD_RETRY}]

        outcomes = RayTrainGroup._compute_attempt_outcomes([cell], [results])

        assert outcomes["discarded"] == [0]
