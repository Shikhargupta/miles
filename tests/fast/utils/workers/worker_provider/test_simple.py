import inspect

import pytest

from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo
from miles.utils.workers.worker_provider.simple import SimpleWorkerProvider


def _make_cell(index: int) -> CellInfo:
    return CellInfo(
        cell_id=f"cell-{index}",
        spec_name="engine",
        members_hash=f"hash-{index}",
        member_urls=[f"http://host-{index}:8000"],
    )


def _make_provider(**overrides) -> SimpleWorkerProvider:
    kwargs = dict(
        worker_urls={"router": "http://router:30000", "session-server": "http://session:20000"},
        cells=[_make_cell(0), _make_cell(1)],
    )
    kwargs.update(overrides)
    return SimpleWorkerProvider(**kwargs)


class _RecordingReconcileFn:
    def __init__(self) -> None:
        self.calls: list[tuple[str, CellInfo | None]] = []

    async def __call__(self, cell_id: str, observed: CellInfo | None) -> None:
        self.calls.append((cell_id, observed))


class TestInit:
    def test_is_a_worker_provider(self):
        """The simple provider satisfies the provider contract."""
        assert isinstance(_make_provider(), BaseWorkerProvider)

    def test_rejects_duplicate_cell_ids(self):
        """Two cells sharing an id are rejected at construction."""
        with pytest.raises(AssertionError):
            _make_provider(cells=[_make_cell(0), _make_cell(0)])

    async def test_is_isolated_from_later_input_mutation(self):
        """Mutating the input containers after construction does not change the provider."""
        worker_urls = {"router": "http://router:30000"}
        cells = [_make_cell(0)]
        provider = _make_provider(worker_urls=worker_urls, cells=cells)

        worker_urls["router"] = "http://other:1"
        cells.clear()

        assert await provider.get_url("router") == "http://router:30000"


class TestGetUrl:
    async def test_returns_url_from_address_book(self):
        """A configured worker name resolves to its url."""
        provider = _make_provider()
        assert await provider.get_url("router") == "http://router:30000"
        assert await provider.get_url("session-server") == "http://session:20000"

    async def test_rejects_unknown_worker_name(self):
        """An unknown worker name is rejected instead of returning a guess."""
        with pytest.raises(AssertionError):
            await _make_provider().get_url("nonexistent")


class TestGetHandle:
    async def test_raises_not_implemented(self):
        """Handles are not supported until the rpc worker handle layer lands."""
        with pytest.raises(NotImplementedError):
            await _make_provider().get_handle("router")


class TestWatchCells:
    async def test_announces_every_cell_before_returning(self):
        """All configured cells are announced as new exactly once, in order, before watch_cells returns."""
        reconcile_fn = _RecordingReconcileFn()
        await _make_provider().watch_cells(reconcile_fn)
        assert reconcile_fn.calls == [("cell-0", _make_cell(0)), ("cell-1", _make_cell(1))]

    async def test_announces_nothing_without_cells(self):
        """A provider with an empty cell list never calls the reconcile fn."""
        reconcile_fn = _RecordingReconcileFn()
        await _make_provider(cells=[]).watch_cells(reconcile_fn)
        assert reconcile_fn.calls == []

    async def test_stop_fn_is_an_awaitable_noop(self):
        """The returned stop fn can be awaited and does nothing."""
        stop_watch_fn = await _make_provider().watch_cells(_RecordingReconcileFn())
        assert inspect.iscoroutinefunction(stop_watch_fn)
        assert await stop_watch_fn() is None

    async def test_each_watch_replays_the_full_list(self):
        """A second watch_cells call announces the same static cells again."""
        provider = _make_provider()
        first_fn = _RecordingReconcileFn()
        second_fn = _RecordingReconcileFn()

        await provider.watch_cells(first_fn)
        await provider.watch_cells(second_fn)

        assert first_fn.calls == second_fn.calls
        assert len(second_fn.calls) == 2

    async def test_propagates_reconcile_fn_errors(self):
        """An error raised by the reconcile fn during the initial announcement propagates to the caller."""

        async def failing_fn(cell_id: str, observed: CellInfo | None) -> None:
            raise RuntimeError("reconcile failed")

        with pytest.raises(RuntimeError, match="reconcile failed"):
            await _make_provider().watch_cells(failing_fn)
