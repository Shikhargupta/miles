from types import SimpleNamespace

import pytest
import ray
from tests.fast.ray.train.conftest import make_alive_cell

from miles.ray.train.group import RayTrainGroup


async def _noop_run_after_step(**kwargs) -> None:
    return None


pytestmark = pytest.mark.asyncio

_DUMMY_DATA_PACK = {"data_ref": "data", "sample_indices": [0]}


def _make_group(cells: list) -> RayTrainGroup:
    group = object.__new__(RayTrainGroup)
    group._cells = cells
    group.args = SimpleNamespace(enable_event_analyzer=False, save_debug_event_data=None)
    group._witness_allocator = None
    group._indep_dp_quorum_id = 0
    group._test_action_executor = SimpleNamespace(run_after_step=_noop_run_after_step)
    return group


def _train_calls_of(cell) -> list[tuple]:
    return [
        [call for call in ray.get(handle.get_calls.remote()) if call[0] == "train"]
        for handle in cell._get_actor_handles()
    ]


class TestCellDistributesExternalData:
    async def test_each_worker_receives_its_own_payload(self):
        """The critic payload of worker i must reach exactly worker i, like the v1 group did."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        payloads = [{"values": ["v0"]}, {"values": ["v1"]}]

        await cell.train(
            rollout_id=5,
            rollout_data_ref="rollout",
            witness_info=None,
            attempt=0,
            external_data=payloads,
        )

        calls_per_worker = _train_calls_of(cell)
        assert [call[2]["external_data"] for [call] in calls_per_worker] == payloads
        assert all(call[1] == () for [call] in calls_per_worker)
        assert all(
            (call[2]["rollout_id"], call[2]["rollout_data_ref"]) == (5, "rollout") for [call] in calls_per_worker
        )

    async def test_no_payload_omits_the_keyword_entirely(self):
        """Backends without an external_data parameter (fsdp) must keep working."""
        cell = make_alive_cell(0, alive_cell_indices=[0])

        await cell.train(rollout_id=7, rollout_data_ref="rollout", witness_info=None, attempt=0)

        for [call] in _train_calls_of(cell):
            assert call[2] == {"rollout_id": 7, "rollout_data_ref": "rollout", "witness_info": None, "attempt": 0}

    async def test_a_payload_count_mismatch_is_rejected(self):
        """A wrong payload count would silently misroute values, so it must raise."""
        cell = make_alive_cell(0, alive_cell_indices=[0])

        with pytest.raises(ValueError, match="one payload per train worker"):
            await cell.train(
                rollout_id=5,
                rollout_data_ref="rollout",
                witness_info=None,
                attempt=0,
                external_data=[{"values": []}],
            )


class TestGroupExternalData:
    async def test_train_forwards_external_data_to_the_cell(self):
        """The group hands the driver-provided payloads down to its only cell."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        group = _make_group([cell])
        payloads = [{"values": ["v0"]}, {"values": ["v1"]}]

        await group.train(3, _DUMMY_DATA_PACK, external_data=payloads)

        calls_per_worker = _train_calls_of(cell)
        assert [call[2]["external_data"] for [call] in calls_per_worker] == payloads

    async def test_external_data_is_rejected_with_multiple_cells(self):
        """Independent DP has no defined mapping from payloads to cells yet."""
        cells = [make_alive_cell(index, alive_cell_indices=[0, 1]) for index in range(2)]
        group = _make_group(cells)

        with pytest.raises(AssertionError, match="single cell"):
            await group.train(3, _DUMMY_DATA_PACK, external_data=[{"values": []}])
