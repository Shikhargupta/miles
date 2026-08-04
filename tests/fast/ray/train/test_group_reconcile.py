from types import SimpleNamespace

import pytest

from miles.ray.specs.train import compute_trainer_spec_name
from miles.ray.train.group import TrainerController
from miles.utils.workers.worker_provider.base import CellInfo

pytestmark = pytest.mark.asyncio

_SPEC_NAME = compute_trainer_spec_name("actor")


def _make_group(*, num_cells: int = 2) -> TrainerController:
    group = object.__new__(TrainerController)
    group.args = SimpleNamespace(
        indep_dp=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=num_cells,
        train_backend="megatron",
    )
    group._role = "actor"
    group._with_ref = False
    group._with_opd_teacher = False
    group._spec_name = _SPEC_NAME
    group._rollout_executor = None
    group._health_checker_config = None
    group._health_checker_activeness = True
    group._cells_by_id = {}
    return group


def _make_cell_info(cell_index: int, *, workers_hash: str = "pseudo-hash-1") -> CellInfo:
    return CellInfo(
        cell_id=f"{_SPEC_NAME}-{cell_index}",
        spec_name=_SPEC_NAME,
        alive=True,
        worker_names=[f"{_SPEC_NAME}-{cell_index}-0"],
        workers_hash=workers_hash,
        meta={"role": "actor", "cell_index": cell_index},
    )


class TestReconcile:
    async def test_an_observed_cell_is_added(self):
        """The group learns about its cells from the manager instead of creating them."""
        group = _make_group()

        await group._reconcile(f"{_SPEC_NAME}-0", _make_cell_info(0))

        assert [cell.cell_index for cell in group._cells] == [0]

    async def test_a_disappeared_cell_is_dropped(self):
        """A cell the manager no longer reports must stop being trained."""
        group = _make_group()
        await group._reconcile(f"{_SPEC_NAME}-1", _make_cell_info(1))

        await group._reconcile(f"{_SPEC_NAME}-1", None)

        assert group._cells == []

    async def test_reobserving_a_known_cell_keeps_the_same_object(self):
        """Recreating the cell would throw away its state machine and health checker."""
        group = _make_group()
        await group._reconcile(f"{_SPEC_NAME}-0", _make_cell_info(0))
        first = group._cells[0]

        await group._reconcile(f"{_SPEC_NAME}-0", _make_cell_info(0))

        assert group._cells[0] is first

    async def test_a_relaunched_cell_is_replaced(self):
        """A new generation hands out new actor handles, so keeping the old object would use dead ones."""
        group = _make_group()
        await group._reconcile(f"{_SPEC_NAME}-0", _make_cell_info(0))
        first = group._cells[0]

        await group._reconcile(f"{_SPEC_NAME}-0", _make_cell_info(0, workers_hash="pseudo-hash-2"))

        assert group._cells[0] is not first
        assert group._cells[0].workers_hash == "pseudo-hash-2"

    async def test_cells_are_ordered_by_index_whatever_the_arrival_order(self):
        """Independent DP ranks are derived from position, so order must be stable."""
        group = _make_group(num_cells=3)

        for cell_index in [2, 0, 1]:
            await group._reconcile(f"{_SPEC_NAME}-{cell_index}", _make_cell_info(cell_index))

        assert [cell.cell_index for cell in group._cells] == [0, 1, 2]


class TestWaitExpectedNumCells:
    async def test_waiting_returns_once_every_cell_is_observed(self):
        """Training must not start against half a fleet."""
        group = _make_group(num_cells=2)
        for cell_index in range(2):
            await group._reconcile(f"{_SPEC_NAME}-{cell_index}", _make_cell_info(cell_index))

        await group._wait_expected_num_cells(timeout=1.0)

    async def test_waiting_gives_up_when_cells_never_appear(self):
        """A silent hang here would look like a stuck first step."""
        group = _make_group(num_cells=2)

        with pytest.raises(TimeoutError):
            await group._wait_expected_num_cells(timeout=1.0)
