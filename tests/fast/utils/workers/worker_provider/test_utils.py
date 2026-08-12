import pytest

from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_provider.utils import apply_cell_observation

pytestmark = pytest.mark.asyncio

_CELL_ID = "spec-0"


def _make_cell_info(workers_hash: str = "hash-1", *, worker_names: list[str] | None = None, **meta) -> CellInfo:
    return CellInfo(
        cell_id=_CELL_ID,
        pool_id="spec",
        alive=True,
        worker_names=worker_names if worker_names is not None else ["spec-0-0"],
        workers_hash=workers_hash,
        meta=meta,
    )


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def add(self, cell_id: str, observed: CellInfo) -> None:
        self.calls.append(("add", cell_id, observed.workers_hash))

    async def remove(self, cell_id: str) -> None:
        self.calls.append(("remove", cell_id))


class TestApplyCellObservation:
    async def test_an_unknown_observed_cell_is_added(self):
        """A newly reported cell must enter the bookkeeping."""
        recorder = _Recorder()

        await apply_cell_observation(
            cell_id=_CELL_ID,
            observed=_make_cell_info(),
            actual=None,
            add=recorder.add,
            remove=recorder.remove,
        )

        assert recorder.calls == [("add", _CELL_ID, "hash-1")]

    async def test_a_disappeared_known_cell_is_removed(self):
        """A cell the provider stops reporting must leave the bookkeeping."""
        recorder = _Recorder()

        await apply_cell_observation(
            cell_id=_CELL_ID,
            observed=None,
            actual=_make_cell_info(),
            add=recorder.add,
            remove=recorder.remove,
        )

        assert recorder.calls == [("remove", _CELL_ID)]

    async def test_a_disappeared_unknown_cell_is_ignored(self):
        """Removing what was never added would raise in the callbacks."""
        recorder = _Recorder()

        await apply_cell_observation(
            cell_id=_CELL_ID, observed=None, actual=None, add=recorder.add, remove=recorder.remove
        )

        assert recorder.calls == []

    async def test_a_changed_workers_hash_replaces_the_cell(self):
        """A new worker generation must not be served through the old cell object."""
        recorder = _Recorder()

        await apply_cell_observation(
            cell_id=_CELL_ID,
            observed=_make_cell_info("hash-2"),
            actual=_make_cell_info("hash-1"),
            add=recorder.add,
            remove=recorder.remove,
        )

        assert recorder.calls == [("remove", _CELL_ID), ("add", _CELL_ID, "hash-2")]

    async def test_an_unchanged_observation_keeps_the_cell(self):
        """Recreating the cell would throw away its accumulated state."""
        recorder = _Recorder()

        await apply_cell_observation(
            cell_id=_CELL_ID,
            observed=_make_cell_info(),
            actual=_make_cell_info(),
            add=recorder.add,
            remove=recorder.remove,
        )

        assert recorder.calls == []

    async def test_a_changed_worker_name_replaces_the_cell_even_at_the_same_hash(self):
        """The addresses a cell is driven through live outside its hash, so a hash-blind compare would strand them."""
        recorder = _Recorder()

        await apply_cell_observation(
            cell_id=_CELL_ID,
            observed=_make_cell_info(worker_names=["spec-0-0", "spec-0-1"]),
            actual=_make_cell_info(),
            add=recorder.add,
            remove=recorder.remove,
        )

        assert recorder.calls == [("remove", _CELL_ID), ("add", _CELL_ID, "hash-1")]

    async def test_changed_metadata_replaces_the_cell_even_at_the_same_hash(self):
        """A cell builds its whole identity from meta once, so a changed field has to rebuild it."""
        recorder = _Recorder()

        await apply_cell_observation(
            cell_id=_CELL_ID,
            observed=_make_cell_info(model_id="actor"),
            actual=_make_cell_info(model_id="critic"),
            add=recorder.add,
            remove=recorder.remove,
        )

        assert recorder.calls == [("remove", _CELL_ID), ("add", _CELL_ID, "hash-1")]
