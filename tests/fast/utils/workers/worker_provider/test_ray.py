import pytest

from miles.utils.test_utils.clock import FakeClock
from miles.utils.workers.ray_worker_handle import RayWorkerHandle
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo
from miles.utils.workers.worker_provider.ray import RayWorkerInfo, RayWorkerProvider

_POLL_INTERVAL_SECONDS = 10.0


def _make_worker_info(**overrides) -> RayWorkerInfo:
    kwargs = dict(name="engine-0-0", cell_id="cell-0", generation=0, url="http://host-0:8000")
    kwargs.update(overrides)
    return RayWorkerInfo(**kwargs)


class _FakeManagerMethod:
    def __init__(self, fn) -> None:
        self._fn = fn

    def remote(self, **kwargs):
        return self._fn(**kwargs)


class _FakeManager:
    def __init__(self, worker_infos: list[RayWorkerInfo]) -> None:
        self.worker_infos = list(worker_infos)
        self.requested_spec_names: list[str] = []
        self.get_worker_infos = _FakeManagerMethod(self._get_worker_infos)

    async def _get_worker_infos(self, *, spec_name: str) -> list[RayWorkerInfo]:
        self.requested_spec_names.append(spec_name)
        return list(self.worker_infos)


class _Harness:
    def __init__(self, worker_infos: list[RayWorkerInfo]) -> None:
        self.manager = _FakeManager(worker_infos)
        self.clock = FakeClock()
        self.reconcile_calls: list[tuple[str, CellInfo | None]] = []
        self.provider = RayWorkerProvider(
            manager=self.manager,
            spec_name="engine",
            poll_interval_seconds=_POLL_INTERVAL_SECONDS,
            clock=self.clock,
        )

    async def reconcile_fn(self, cell_id: str, observed: CellInfo | None) -> None:
        self.reconcile_calls.append((cell_id, observed))


class TestInit:
    def test_is_a_worker_provider(self):
        """The ray provider satisfies the provider contract."""
        assert isinstance(_Harness([]).provider, BaseWorkerProvider)


class TestGetHandle:
    async def test_wraps_the_named_actor(self, monkeypatch):
        """The handle wraps the ray named actor resolved by worker name."""
        import ray

        requested_names = []
        monkeypatch.setattr(ray, "get_actor", lambda name: requested_names.append(name) or object())

        handle = await _Harness([]).provider.get_handle("engine-0-0")

        assert isinstance(handle, RayWorkerHandle)
        assert requested_names == ["engine-0-0"]


class TestGetUrl:
    async def test_returns_url_from_manager_metadata(self):
        """The url comes from the manager's worker info for that name."""
        harness = _Harness([_make_worker_info()])
        assert await harness.provider.get_url("engine-0-0") == "http://host-0:8000"
        assert harness.manager.requested_spec_names == ["engine"]

    async def test_rejects_unknown_worker_name(self):
        """A name the manager does not report is rejected."""
        harness = _Harness([_make_worker_info()])
        with pytest.raises(AssertionError):
            await harness.provider.get_url("nonexistent")

    async def test_rejects_worker_without_url(self):
        """A worker the manager reports without a url is rejected."""
        harness = _Harness([_make_worker_info(url=None)])
        with pytest.raises(AssertionError):
            await harness.provider.get_url("engine-0-0")


class TestWatchCells:
    async def test_announces_initial_cells_grouped_by_cell_id(self):
        """Workers sharing a cell id are announced as one cell with member urls in listing order."""
        harness = _Harness(
            [
                _make_worker_info(name="engine-0-0", cell_id="cell-0", url="http://host-0:8000"),
                _make_worker_info(name="engine-0-1", cell_id="cell-0", url="http://host-1:8000"),
                _make_worker_info(name="engine-1-0", cell_id="cell-1", url="http://host-2:8000"),
            ]
        )

        stop_watch_fn = await harness.provider.watch_cells(harness.reconcile_fn)

        assert [cell_id for cell_id, _ in harness.reconcile_calls] == ["cell-0", "cell-1"]
        cell_0 = harness.reconcile_calls[0][1]
        assert cell_0.member_urls == ["http://host-0:8000", "http://host-1:8000"]
        await stop_watch_fn()

    async def test_excludes_urlless_members_from_member_urls(self):
        """Members without a url still shape the cell but contribute no url."""
        harness = _Harness([_make_worker_info(url=None)])

        stop_watch_fn = await harness.provider.watch_cells(harness.reconcile_fn)

        assert harness.reconcile_calls[0][1].member_urls == []
        await stop_watch_fn()

    async def test_generation_bump_changes_members_hash(self):
        """A restarted worker (generation +1) is observed as a changed cell."""
        harness = _Harness([_make_worker_info(generation=0)])
        stop_watch_fn = await harness.provider.watch_cells(harness.reconcile_fn)
        initial_hash = harness.reconcile_calls[0][1].members_hash

        harness.manager.worker_infos = [_make_worker_info(generation=1)]
        await harness.clock.elapse(_POLL_INTERVAL_SECONDS)

        assert harness.reconcile_calls[-1][0] == "cell-0"
        assert harness.reconcile_calls[-1][1].members_hash != initial_hash
        await stop_watch_fn()

    async def test_removed_worker_removes_its_cell(self):
        """A cell whose workers disappear is announced with observed=None."""
        harness = _Harness([_make_worker_info()])
        stop_watch_fn = await harness.provider.watch_cells(harness.reconcile_fn)

        harness.manager.worker_infos = []
        await harness.clock.elapse(_POLL_INTERVAL_SECONDS)

        assert harness.reconcile_calls[-1] == ("cell-0", None)
        await stop_watch_fn()

    async def test_stop_halts_polling(self):
        """After stopping the watch, membership changes are no longer announced."""
        harness = _Harness([_make_worker_info()])
        stop_watch_fn = await harness.provider.watch_cells(harness.reconcile_fn)
        await stop_watch_fn()

        harness.manager.worker_infos = []
        await harness.clock.elapse(_POLL_INTERVAL_SECONDS * 3)

        assert harness.reconcile_calls == [harness.reconcile_calls[0]]
