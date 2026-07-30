from __future__ import annotations

import asyncio
import textwrap
import time

import pytest
import ray
from tests.fast.ray.rollout.conftest import make_args

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.ray.rollout.inference_controller import InferenceController


class _NoopRouterApiClient:
    """The rollout process registers its engines for real; ``sglang_router_ip``
    here is a placeholder that keeps ``start_router`` short-circuited, and no
    router listens on it."""

    def __init__(self, router_url: str):
        self.router_url = router_url

    async def add_worker(self, **kwargs):
        return None

    async def remove_worker(self, **kwargs):
        return None


@pytest.fixture
def patch_low_level(monkeypatch):
    """Replace, in the test process:
    - ``CommandActor`` → ``MockSGLangEngine`` so created actors are mocks
      (the real addr allocator runs; each mock serves HTTP on its port).
    - ``SGLangRouterApiClient`` → no-op (no router runs at the placeholder address).
    - ``start_session_server`` → no-op (the production default touches network)."""
    import miles.ray.rollout.inference_controller as ictl
    import miles.ray.rollout.rollout_server as rsrv
    import miles.ray.rollout.server_cell as scell
    from miles.utils.test_utils.mock_sglang_engine import MockSGLangEngine

    monkeypatch.setattr(scell, "CommandActor", MockSGLangEngine.__ray_actor_class__)
    # multi-model tests would otherwise spawn a real router subprocess for
    # ``model_idx > 0`` (force_new=True bypasses the args.sglang_router_ip cache).
    monkeypatch.setattr(
        rsrv,
        "start_router",
        lambda args, **kw: (args.sglang_router_ip, args.sglang_router_port),
    )

    monkeypatch.setattr(rsrv, "SGLangRouterApiClient", _NoopRouterApiClient)
    monkeypatch.setattr(ictl, "start_session_server", lambda args: None)


def _write_sglang_config(tmp_path, *, models: list[tuple[str, bool]]) -> str:
    """Write a multi-model sglang yaml — each entry ``(name, update_weights)``.
    Each model gets one regular group with 2 engines × 1 GPU = 2 GPUs. With N
    models, total GPUs = 2N; ``args.rollout_num_gpus`` must match."""
    lines = ["sglang:"]
    for name, update_weights in models:
        lines.extend(
            [
                f"  - name: {name}",
                f"    update_weights: {str(update_weights).lower()}",
                "    server_groups:",
                "      - worker_type: regular",
                "        num_gpus: 2",
                "        num_gpus_per_engine: 1",
            ]
        )
    cfg_path = tmp_path / "sglang.yaml"
    cfg_path.write_text(textwrap.dedent("\n".join(lines)) + "\n")
    return str(cfg_path)


def _make_test_args(tmp_path, *, models: list[tuple[str, bool]]):
    """Build args that drive ``InferenceController.create`` →
    ``start_rollout_servers`` → N model servers each with 1 group of 2 mock
    engines."""
    cfg = _write_sglang_config(tmp_path, models=models)
    rollout_num_gpus = 2 * len(models)
    return make_args(
        sglang_config=cfg,
        rollout_num_gpus=rollout_num_gpus,
        # short-circuit start_router (returns early when ip+port already set)
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        # disable everything else that would spawn subprocesses or hit network
        use_session_server=False,
        use_fault_tolerance=False,
        use_wandb=False,
        use_tensorboard=False,
        use_mlflow=False,
        use_distributed_post=False,
        sglang_server_concurrency=1,
    )


async def _sync_cells(controller):
    """Reconcile against the provider right now instead of waiting for its next poll."""
    observed = await controller.provider.list_cells()
    for cell_id in controller.list_cell_ids():
        await controller._reconcile(cell_id, observed.get(cell_id))


def _cells(controller, model: str = "actor"):
    return list(controller.servers[model].server_cells.values())


async def _assert_engine_dies(actor_handle, *, deadline_s: float = 15.0, poll_interval_s: float = 0.2) -> None:
    deadline = time.monotonic() + deadline_s
    while True:
        try:
            ray.get(actor_handle.get_calls.remote(), timeout=5.0)
        except (ray.exceptions.RayActorError, ray.exceptions.RayTaskError):
            return
        except ray.exceptions.GetTimeoutError:
            pass
        if time.monotonic() >= deadline:
            pytest.fail(f"engine actor still alive {deadline_s}s after stop_cell")
        await asyncio.sleep(poll_interval_s)


@pytest.mark.asyncio
class TestInferenceControllerInit:
    async def test_init_creates_live_mock_engines_via_real_start_rollout_servers(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """End-to-end smoke: production ``create`` + ``start_rollout_servers``
        runs against MockSGLangEngine; the resulting engines are addressable over
        http via the public ``get_updatable_engines_and_lock``, and their launcher
        actors are reachable through the engine slots."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        controller = await InferenceController.create(args, pg)
        eal = await controller.get_updatable_engines_and_lock()
        assert len(eal.rollout_engines) == 2
        for api_client in eal.rollout_engines:
            assert isinstance(api_client, SGLangApiClient)
            assert await api_client.health_generate(timeout=5.0) is True
        for cell in _cells(controller):
            assert isinstance(ray.get(cell.primary_actor_handle.get_calls.remote()), list)


@pytest.mark.asyncio
class TestStartStopCell:
    async def test_stop_cell_kills_target_engine_only(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """``stop_cell`` kills cell 0's actor; cell 1 untouched."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        controller = await InferenceController.create(args, pg)
        await controller.get_updatable_engines_and_lock()
        actor0, actor1 = [cell.primary_actor_handle for cell in _cells(controller)]

        await controller.worker_manager.stop_cell("actor-0")
        await _sync_cells(controller)

        await _assert_engine_dies(actor0)
        assert isinstance(ray.get(actor1.get_calls.remote()), list)

    async def test_start_cell_recovers_after_stop_cell(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """stop_cell → start_cell brings up a fresh mock actor, and the reconcile
        pass re-attaches the cell to it."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        controller = await InferenceController.create(args, pg)
        eal_before = await controller.get_updatable_engines_and_lock()
        actor0_before = _cells(controller)[0].primary_actor_handle
        url_before = eal_before.rollout_engines[0].server_url

        await controller.worker_manager.stop_cell("actor-0")
        await _sync_cells(controller)
        await controller.worker_manager.start_cell("actor-0")
        await _sync_cells(controller)

        eal_after = await controller.get_updatable_engines_and_lock()
        actor0_after = _cells(controller)[0].primary_actor_handle

        assert actor0_after is not actor0_before, "start_cell must produce a fresh actor"
        assert eal_after.rollout_engines[0].server_url != url_before, "the recovered engine serves on a new port"
        assert await eal_after.rollout_engines[0].health_generate(timeout=5.0) is True
        assert isinstance(ray.get(actor0_after.get_calls.remote()), list)

    async def test_stop_cell_targets_high_id_correctly(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """``stop_cell("actor-1")`` (not 0) must kill engine 1, leaving engine 0
        alive — guards against addressing the wrong cell."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        controller = await InferenceController.create(args, pg)
        await controller.get_updatable_engines_and_lock()
        actor0, actor1 = [cell.primary_actor_handle for cell in _cells(controller)]

        await controller.worker_manager.stop_cell("actor-1")

        assert isinstance(ray.get(actor0.get_calls.remote()), list)
        await _assert_engine_dies(actor1)

    async def test_stop_cell_is_idempotent_on_already_stopped(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """Stopping an already stopped cell does not raise — production code logs
        and proceeds when the engine is already de-allocated."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        controller = await InferenceController.create(args, pg)
        await controller.get_updatable_engines_and_lock()  # ensure engines are alive

        await controller.worker_manager.stop_cell("actor-0")
        await controller.worker_manager.stop_cell("actor-0")  # must not raise


@pytest.mark.asyncio
class TestCellDispatchAcrossModels:
    async def test_cells_route_to_correct_model_by_sorted_srv_key(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """Cells are flattened in sorted-srv-key order: with models ("actor",
        "ref") the cells map (0,1)→actor, (2,3)→ref. Stopping cell 2 must hit
        ref's first engine and leave actor's engines untouched."""
        args = _make_test_args(tmp_path, models=[("actor", True), ("ref", False)])
        pg = placement_group_factory(4)

        controller = await InferenceController.create(args, pg)
        actor_handles = [cell.primary_actor_handle for cell in _cells(controller, "actor")]
        ref_handles = [cell.primary_actor_handle for cell in _cells(controller, "ref")]

        await controller.worker_manager.stop_cell("ref-0")

        # actor untouched
        for h in actor_handles:
            assert isinstance(ray.get(h.get_calls.remote()), list)
        # ref engine 0 dead, ref engine 1 alive
        await _assert_engine_dies(ref_handles[0])
        assert isinstance(ray.get(ref_handles[1].get_calls.remote()), list)


@pytest.mark.asyncio
class TestGetUpdatableEnginesAndLock:
    async def test_returns_only_updatable_servers_engines_in_multi_model_setup(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """With actor (update_weights=True) + ref (update_weights=False), the
        returned EnginesAndLock contains the actor's engines only."""
        args = _make_test_args(tmp_path, models=[("actor", True), ("ref", False)])
        pg = placement_group_factory(4)

        controller = await InferenceController.create(args, pg)
        eal = await controller.get_updatable_engines_and_lock()
        assert len(eal.rollout_engines) == 2  # actor's 2, not ref's 2
        assert eal.engine_gpu_counts == [1, 1]
        assert all(isinstance(api_client, SGLangApiClient) for api_client in eal.rollout_engines)
        assert await eal.rollout_engines[0].health_generate(timeout=5.0) is True

    async def test_returns_empty_when_no_updatable_model(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """If every model has ``update_weights=False`` (e.g. inference-only
        deployment), the returned EnginesAndLock has empty engines list and
        the lock handle is still present (callers always need a lock)."""
        args = _make_test_args(tmp_path, models=[("ref", False)])
        pg = placement_group_factory(2)

        controller = await InferenceController.create(args, pg)
        eal = await controller.get_updatable_engines_and_lock()
        assert eal.rollout_engines == []
        assert eal.engine_gpu_counts == []
        assert eal.has_new_engines is False
        assert eal.rollout_engine_lock is not None

    async def test_has_new_engines_flag_lifecycle(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """Lifecycle the trainer relies on: ``has_new_engines`` is True after
        init, False after ``clear_updatable_has_new_engines``, True again
        after ``start_cell`` spawns a fresh engine."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        controller = await InferenceController.create(args, pg)
        eal_init = await controller.get_updatable_engines_and_lock()
        assert eal_init.has_new_engines is True

        await controller.clear_updatable_has_new_engines()
        eal_cleared = await controller.get_updatable_engines_and_lock()
        assert eal_cleared.has_new_engines is False

        await controller.worker_manager.stop_cell("actor-0")
        await _sync_cells(controller)
        await controller.worker_manager.start_cell("actor-0")
        await _sync_cells(controller)
        eal_recovered = await controller.get_updatable_engines_and_lock()
        assert eal_recovered.has_new_engines is True

    async def test_clear_does_not_affect_non_updatable_server(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """``clear_updatable_has_new_engines`` must touch only the updatable
        server's flag; non-updatable (ref) servers keep their flag intact."""
        args = _make_test_args(tmp_path, models=[("actor", True), ("ref", False)])
        pg = placement_group_factory(4)

        controller = await InferenceController.create(args, pg)
        # Force ref's flag True so we can detect any erroneous clear.
        controller.servers["ref"].has_new_engines = True

        await controller.clear_updatable_has_new_engines()

        assert controller.servers["ref"].has_new_engines is True
        assert controller.servers["actor"].has_new_engines is False

    async def test_multiple_updatable_servers_raises_assertion(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """Production guards against misconfiguration where two models both set
        ``update_weights=True``; that's ambiguous for the trainer."""
        args = _make_test_args(tmp_path, models=[("actor1", True), ("actor2", True)])
        pg = placement_group_factory(4)

        controller = await InferenceController.create(args, pg)
        with pytest.raises(ValueError, match="Multiple servers"):
            await controller.get_updatable_engines_and_lock()


@pytest.mark.asyncio
class TestCheckWeights:
    async def test_check_weights_targets_only_updatable_model(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """``check_weights`` targets only the updatable model. The snapshot/reset/
        compare round-trip is meaningless for a frozen model (restored from disk,
        never re-synced via update_weights), so it must be skipped there."""
        args = _make_test_args(tmp_path, models=[("actor", True), ("ref", False)])
        pg = placement_group_factory(4)

        controller = await InferenceController.create(args, pg)
        await controller.get_updatable_engines_and_lock()  # wait for engines to be alive

        results = await controller.check_weights(action="pre_update")

        # Updatable server only: one flat entry per cell's primary engine.
        assert len(results) == 2
        for engine_result in results:
            assert engine_result == {"mock": True}

        updatable_cells = [
            cell for srv in controller.servers.values() if srv.update_weights for cell in srv.server_cells.values()
        ]
        frozen_cells = [
            cell for srv in controller.servers.values() if not srv.update_weights for cell in srv.server_cells.values()
        ]
        assert updatable_cells and frozen_cells

        for cell in updatable_cells:
            paths = ray.get(cell.primary_actor_handle.get_http_paths.remote())
            assert "/weights_checker" in paths, f"updatable engine {cell.addr_info.server_url} was not checked"
        for cell in frozen_cells:
            paths = ray.get(cell.primary_actor_handle.get_http_paths.remote())
            assert "/weights_checker" not in paths, f"frozen engine {cell.addr_info.server_url} must not be checked"


@pytest.mark.asyncio
class TestReconcileAfterAnExternalRestart:
    async def test_a_killed_engine_is_removed_then_recreated_by_reconcile(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """The controller never restarts an engine itself; it follows the manager."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        controller = await InferenceController.create(args, pg)
        actor0_before = _cells(controller)[0].primary_actor_handle

        await controller.worker_manager.stop_cell("actor-0")
        await _sync_cells(controller)
        assert "actor-0" not in controller.servers["actor"].server_cells

        await controller.worker_manager.start_cell("actor-0")
        await _sync_cells(controller)

        cell = controller.servers["actor"].server_cells["actor-0"]
        assert cell.is_alive
        assert cell.primary_actor_handle is not actor0_before
        assert isinstance(ray.get(cell.primary_actor_handle.get_calls.remote()), list)


@pytest.mark.asyncio
class TestRolloutFaultToleranceIsUnsupported:
    async def test_pausing_the_reconcile_loop_leaves_the_engines_attached(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """A paused controller must still serve the engine snapshot the trainer holds."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        controller = await InferenceController.create(args, pg)

        await controller.health_monitoring_pause()
        eal = await controller.get_updatable_engines_and_lock()
        assert len(eal.rollout_engines) == 2
        await controller.health_monitoring_resume()

    async def test_fault_injection_refuses_to_run(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """The injector depended on the deleted monitor to observe the crash."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        controller = await InferenceController.create(args, pg)

        with pytest.raises(NotImplementedError):
            await controller._try_ci_fault_injection()
