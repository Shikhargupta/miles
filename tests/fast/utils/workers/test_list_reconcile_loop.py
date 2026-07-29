import pytest

from miles.utils.test_utils.clock import FakeClock
from miles.utils.workers.list_reconcile_loop import ListBasedReconcileLoop
from miles.utils.workers.worker_provider.base import CellInfo

_POLL_INTERVAL_SECONDS = 10.0


def _make_cell(index: int, *, members_hash: str | None = None) -> CellInfo:
    return CellInfo(
        cell_id=f"cell-{index}",
        spec_name="engine",
        members_hash=members_hash or f"hash-{index}",
        member_urls=[f"http://host-{index}:8000"],
    )


class _Harness:
    def __init__(self, initial_cells: list[CellInfo]) -> None:
        self.cells = list(initial_cells)
        self.list_error: Exception | None = None
        self.reconcile_calls: list[tuple[str, CellInfo | None]] = []
        self.clock = FakeClock()
        self.loop = ListBasedReconcileLoop(
            list_cells_fn=self._list_cells,
            reconcile_fn=self._reconcile,
            poll_interval_seconds=_POLL_INTERVAL_SECONDS,
            clock=self.clock,
        )

    async def _list_cells(self) -> list[CellInfo]:
        if self.list_error is not None:
            raise self.list_error
        return list(self.cells)

    async def _reconcile(self, cell_id: str, observed: CellInfo | None) -> None:
        self.reconcile_calls.append((cell_id, observed))


class TestStart:
    async def test_announces_initial_list_before_returning(self):
        """All initially listed cells are announced before start returns."""
        harness = _Harness([_make_cell(0), _make_cell(1)])
        await harness.loop.start()
        assert harness.reconcile_calls == [("cell-0", _make_cell(0)), ("cell-1", _make_cell(1))]
        await harness.loop.stop()

    async def test_initial_list_error_propagates(self):
        """A failing initial list fails start instead of being swallowed."""
        harness = _Harness([])
        harness.list_error = RuntimeError("list failed")
        with pytest.raises(RuntimeError, match="list failed"):
            await harness.loop.start()

    async def test_rejects_second_start(self):
        """Starting the loop twice is rejected."""
        harness = _Harness([])
        await harness.loop.start()
        with pytest.raises(AssertionError):
            await harness.loop.start()
        await harness.loop.stop()

    async def test_rejects_duplicate_cell_ids(self):
        """A listing with duplicate cell ids is rejected."""
        harness = _Harness([_make_cell(0), _make_cell(0)])
        with pytest.raises(AssertionError):
            await harness.loop.start()


class TestPolling:
    async def test_unchanged_list_is_not_reannounced(self):
        """Polls that observe no change do not call the reconcile fn again."""
        harness = _Harness([_make_cell(0)])
        await harness.loop.start()
        await harness.clock.elapse(_POLL_INTERVAL_SECONDS * 3)
        assert harness.reconcile_calls == [("cell-0", _make_cell(0))]
        await harness.loop.stop()

    async def test_new_cell_is_announced(self):
        """A cell appearing in a later poll is announced once."""
        harness = _Harness([_make_cell(0)])
        await harness.loop.start()

        harness.cells.append(_make_cell(1))
        await harness.clock.elapse(_POLL_INTERVAL_SECONDS)

        assert harness.reconcile_calls[-1] == ("cell-1", _make_cell(1))
        await harness.loop.stop()

    async def test_removed_cell_is_announced_as_none(self):
        """A cell disappearing from a later poll is announced with observed=None."""
        harness = _Harness([_make_cell(0), _make_cell(1)])
        await harness.loop.start()

        harness.cells = [_make_cell(0)]
        await harness.clock.elapse(_POLL_INTERVAL_SECONDS)

        assert harness.reconcile_calls[-1] == ("cell-1", None)
        await harness.loop.stop()

    async def test_changed_members_hash_is_reannounced(self):
        """A cell whose members hash changes is announced with the new info."""
        harness = _Harness([_make_cell(0)])
        await harness.loop.start()

        harness.cells = [_make_cell(0, members_hash="hash-changed")]
        await harness.clock.elapse(_POLL_INTERVAL_SECONDS)

        assert harness.reconcile_calls[-1] == ("cell-0", _make_cell(0, members_hash="hash-changed"))
        await harness.loop.stop()

    async def test_poll_error_is_survived_and_retried(self):
        """A failing poll keeps the loop alive and later polls observe changes."""
        harness = _Harness([_make_cell(0)])
        await harness.loop.start()

        harness.list_error = RuntimeError("poll failed")
        await harness.clock.elapse(_POLL_INTERVAL_SECONDS)
        harness.list_error = None
        harness.cells.append(_make_cell(1))
        await harness.clock.elapse(_POLL_INTERVAL_SECONDS)

        assert harness.reconcile_calls[-1] == ("cell-1", _make_cell(1))
        await harness.loop.stop()


class TestStop:
    async def test_stop_halts_polling(self):
        """After stop, further time does not trigger reconcile calls."""
        harness = _Harness([_make_cell(0)])
        await harness.loop.start()
        await harness.loop.stop()

        harness.cells.append(_make_cell(1))
        await harness.clock.elapse(_POLL_INTERVAL_SECONDS * 3)

        assert harness.reconcile_calls == [("cell-0", _make_cell(0))]

    async def test_stop_before_start_is_a_noop(self):
        """Stopping a never-started loop does nothing."""
        harness = _Harness([])
        await harness.loop.stop()

    async def test_stop_is_idempotent(self):
        """Stopping twice is safe."""
        harness = _Harness([])
        await harness.loop.start()
        await harness.loop.stop()
        await harness.loop.stop()
