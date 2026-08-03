from types import SimpleNamespace

from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata
from miles.utils.context_lock import ContextLock
from miles.utils.ft_utils.health_checker import NoopHealthChecker, SimpleHealthChecker, SimpleHealthCheckerConfig


class _RecordingHealthChecker(NoopHealthChecker):
    def __init__(self) -> None:
        self.events: list[str] = []

    async def start(self) -> None:
        self.events.append("start")

    def stop(self) -> None:
        self.events.append("stop")

    def pause(self) -> None:
        self.events.append("pause")

    def resume(self) -> None:
        self.events.append("resume")


def _make_config(**overrides) -> SimpleHealthCheckerConfig:
    defaults = dict(interval=10.0, timeout=10.0, first_wait=300.0, failure_threshold=3)
    return SimpleHealthCheckerConfig(**{**defaults, **overrides})


def _make_meta(cell_id: str = "cell-0") -> ServerCellMetadata:
    return ServerCellMetadata(
        model_id="default",
        worker_type="regular",
        cell_id=cell_id,
        num_gpus_per_engine=1,
        gpu_offset=0,
        worker_name=f"{cell_id}-0",
        needs_offload=False,
        update_weights=True,
        workers_hash="pseudo-hash-0",
    )


def _make_server(**overrides) -> RolloutServer:
    return RolloutServer(
        server_cells={},
        args=SimpleNamespace(rollout_external=False, debug_rollout_only=False),
        context_lock=ContextLock("InferenceController"),
        **overrides,
    )


def _attach_cell(srv: RolloutServer, cell_id: str = "cell-0") -> _RecordingHealthChecker:
    checker = _RecordingHealthChecker()
    srv.server_cells[cell_id] = ServerCell(
        args=srv.args,
        meta=_make_meta(cell_id),
        router_api_client=SimpleNamespace(),
        health_checker=checker,
    )
    return checker


async def _stub_add(self: ServerCell) -> None:
    self._mark_pending_weights(server_url="http://10.0.0.1:30000", bootstrap_port=None)
    await self.health_checker.start()


class TestHealthCheckingPauseAndResume:
    async def test_pausing_pauses_every_cell(self):
        """Offload puts engines to sleep, so probing them would report a false failure."""
        srv = _make_server()
        checkers = [_attach_cell(srv, f"cell-{i}") for i in range(2)]

        async with srv.context_lock:
            srv.health_checking_pause()

        assert [c.events for c in checkers] == [["pause"], ["pause"]]

    async def test_resuming_resumes_every_cell(self):
        """Probing must come back once the engines are usable again."""
        srv = _make_server()
        checkers = [_attach_cell(srv, f"cell-{i}") for i in range(2)]

        async with srv.context_lock:
            srv.health_checking_pause()
            srv.health_checking_resume()

        assert [c.events for c in checkers] == [["pause", "resume"], ["pause", "resume"]]

    async def test_pausing_is_recorded_for_cells_added_later(self):
        """Reconcile can add a cell mid-offload; it must not become the one live prober."""
        srv = _make_server()

        async with srv.context_lock:
            srv.health_checking_pause()
        assert srv.health_checking_paused

        async with srv.context_lock:
            srv.health_checking_resume()
        assert not srv.health_checking_paused


class TestAddCellHealthChecker:
    async def test_a_new_cell_inherits_the_paused_state(self, monkeypatch):
        """A cell added while probing is paused must start out paused too."""
        monkeypatch.setattr(ServerCell, "add", _stub_add)
        srv = _make_server(health_checker_config=_make_config())

        async with srv.context_lock:
            srv.health_checking_pause()
            await srv.add_cell(_make_meta())

        checker = srv.server_cells["cell-0"].health_checker
        assert checker._paused
        checker.stop()

    async def test_a_new_cell_probes_when_not_paused(self, monkeypatch):
        """The steady state is an actively probing checker per cell."""
        monkeypatch.setattr(ServerCell, "add", _stub_add)
        srv = _make_server(health_checker_config=_make_config())

        async with srv.context_lock:
            await srv.add_cell(_make_meta())

        checker = srv.server_cells["cell-0"].health_checker
        assert isinstance(checker, SimpleHealthChecker)
        assert not checker._paused
        checker.stop()

    async def test_no_checker_is_created_without_rollout_fault_tolerance(self, monkeypatch):
        """Without rollout FT nothing consumes the health status, so nothing probes."""
        monkeypatch.setattr(ServerCell, "add", _stub_add)
        srv = _make_server(health_checker_config=None)

        async with srv.context_lock:
            await srv.add_cell(_make_meta())

        assert isinstance(srv.server_cells["cell-0"].health_checker, NoopHealthChecker)


class TestServerCellHealthCheckerLifetime:
    async def test_add_starts_the_checker(self, monkeypatch):
        """Probing may only begin once the engine url is known."""
        checker = _RecordingHealthChecker()
        cell = ServerCell(
            args=SimpleNamespace(rollout_external=False, debug_rollout_only=False),
            meta=_make_meta(),
            router_api_client=SimpleNamespace(),
            health_checker=checker,
        )
        monkeypatch.setattr(
            "miles.ray.rollout.server_cell.RayWorkerProvider",
            SimpleNamespace(create=lambda: _StubProvider()),
        )

        await cell.add()

        assert checker.events == ["start"]

    async def test_dispose_stops_the_checker(self):
        """A stopped cell must not keep probing an engine that is gone."""
        checker = _RecordingHealthChecker()
        cell = ServerCell(
            args=SimpleNamespace(rollout_external=False, debug_rollout_only=False),
            meta=_make_meta(),
            router_api_client=SimpleNamespace(),
            health_checker=checker,
        )

        await cell.dispose()

        assert checker.events == ["stop"]


class _StubProvider:
    async def get_addrs(self, worker_name: str) -> dict:
        return {"primary": SimpleNamespace(host="10.0.0.1", port=30000)}
