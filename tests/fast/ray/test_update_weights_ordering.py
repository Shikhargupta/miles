import asyncio
from argparse import Namespace
from unittest.mock import MagicMock

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout.inference_controller import InferenceController
from miles.utils.context_lock import ContextLock


class _ColocatedCellStub:
    def __init__(self) -> None:
        self.init_count = 0
        self.ready = False

    async def init(self) -> None:
        self.init_count += 1
        self.ready = True

    @property
    def is_uninitialized(self) -> bool:
        return not self.ready

    @property
    def is_pending_weights_or_serving(self) -> bool:
        return self.ready


class _ServerStub:
    def __init__(self, server_cells: dict[str, _ColocatedCellStub]) -> None:
        self.server_cells = server_cells


class _UpdatableServerStub:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.update_weights = True
        self.server_cells: dict[str, _ColocatedCellStub] = {}


class _UnreadableServerStub(_UpdatableServerStub):
    @property
    def api_clients(self) -> list[object]:
        raise RuntimeError("the engine set is not readable")


def _make_inference_controller(**arg_overrides: object) -> InferenceController:
    return InferenceController(make_args(**arg_overrides), engine_provider=None, router_providers=[])


@pytest.mark.asyncio
async def test_controller_pauses_health_checks_before_snapshotting_the_engines():
    """``start_update_weights`` pauses the health monitor, then readies the cells, then reads the engine set."""
    order: list[str] = []
    controller = _make_inference_controller()

    async def _record_pause() -> None:
        order.append("health_monitoring_pause")

    async def _record_ensure_cells_ready() -> None:
        order.append("ensure_cells_ready")

    def _record_snapshot() -> None:
        order.append("get_updatable_server")
        return None

    controller.context_lock = ContextLock("InferenceController")
    controller.args = Namespace(colocate=False)
    controller.servers = {}
    controller._health_monitoring_pause = _record_pause
    controller._ensure_cells_ready = _record_ensure_cells_ready
    controller._get_updatable_server = _record_snapshot

    await controller.start_update_weights()

    assert order == ["health_monitoring_pause", "ensure_cells_ready", "get_updatable_server"]


@pytest.mark.asyncio
async def test_start_update_weights_initializes_colocated_cells_before_snapshotting_the_engines():
    """A colocated cell is initialized inside the weight update window, before the engine snapshot is taken."""
    controller = _make_inference_controller(colocate=True)
    cell = _ColocatedCellStub()
    controller.servers = {"default": _ServerStub({"a": cell})}
    init_counts_at_snapshot: list[int] = []

    def _record_snapshot() -> None:
        init_counts_at_snapshot.append(cell.init_count)
        return None

    controller._get_updatable_server = _record_snapshot

    await controller.start_update_weights()

    assert cell.init_count == 1
    assert init_counts_at_snapshot == [1]


@pytest.mark.asyncio
async def test_aborting_the_window_releases_the_lock_and_resumes_health_checking():
    """A release that survives the orchestration script must not keep a failed update's lock forever."""
    controller = _make_inference_controller()
    controller.servers = {}

    info = await controller.start_update_weights()
    assert not controller._health_checker_activeness.get().active

    await controller.abort_update_weights(window_id=info.window_id)

    assert controller._health_checker_activeness.get().active
    assert not controller.context_lock.locked


@pytest.mark.asyncio
async def test_a_window_closed_by_an_abort_can_be_opened_again():
    """A hot restart opens a second window on the same release, which a leaked lock would block forever."""
    controller = _make_inference_controller()
    controller.servers = {}

    first = await controller.start_update_weights()
    await controller.abort_update_weights(window_id=first.window_id)
    second = await controller.start_update_weights()
    await controller.end_update_weights(window_id=second.window_id, snapshot_cell_id_to_hashes={})

    assert not controller.context_lock.locked


@pytest.mark.asyncio
async def test_every_window_is_numbered_apart_from_the_one_before_it():
    """The number is what tells a late caller's window from the one that replaced it, so it may never repeat."""
    controller = _make_inference_controller()
    controller.servers = {}

    first = await controller.start_update_weights()
    await controller.abort_update_weights(window_id=first.window_id)
    second = await controller.start_update_weights()
    await controller.abort_update_weights(window_id=second.window_id)

    assert first.window_id != second.window_id


@pytest.mark.asyncio
async def test_an_action_of_a_window_that_was_already_closed_is_refused():
    """A late abort of a replaced window would resume health checking while the next broadcast is running."""
    controller = _make_inference_controller()
    controller.servers = {}

    first = await controller.start_update_weights()
    await controller.abort_update_weights(window_id=first.window_id)
    await controller.start_update_weights()

    with pytest.raises(AssertionError, match="already closed"):
        await controller.abort_update_weights(window_id=first.window_id)

    assert not controller._health_checker_activeness.get().active


@pytest.mark.asyncio
async def test_an_action_carrying_no_open_window_is_refused():
    """Nothing is holding the lock, so closing a window would release a lock this caller never took."""
    controller = _make_inference_controller()
    controller.servers = {}

    with pytest.raises(AssertionError, match="already closed"):
        await controller.end_update_weights(window_id=1, snapshot_cell_id_to_hashes={})


@pytest.mark.asyncio
async def test_a_window_that_never_opened_resumes_the_health_checking_it_paused():
    """Opening the window fails after pausing, and nothing later gets the chance to resume it."""
    controller = _make_inference_controller()
    controller.servers = {}

    async def _time_out() -> None:
        raise TimeoutError("cells never became ready")

    controller._ensure_cells_ready = _time_out

    with pytest.raises(TimeoutError, match="never became ready"):
        await controller.start_update_weights()

    assert controller._health_checker_activeness.get().active
    await asyncio.wait_for(controller.prepare_eval(), timeout=1)


@pytest.mark.asyncio
async def test_a_window_refused_for_having_two_updatable_servers_resumes_the_health_checking_it_paused():
    """The engine snapshot throws after the window was numbered, which must not leave health checking paused."""
    controller = _make_inference_controller()
    controller.servers = {"a": _UpdatableServerStub("a"), "b": _UpdatableServerStub("b")}

    with pytest.raises(ValueError, match="Multiple servers"):
        await controller.start_update_weights()

    assert controller._health_checker_activeness.get().active
    assert controller._open_update_weights_window_id is None
    await asyncio.wait_for(controller.prepare_eval(), timeout=1)


@pytest.mark.asyncio
async def test_a_window_whose_engine_set_cannot_be_read_resumes_the_health_checking_it_paused():
    """Reading the engines is the last throw point of the window, and it is past the point that numbers it."""
    controller = _make_inference_controller()
    controller.servers = {"a": _UnreadableServerStub("a")}

    with pytest.raises(RuntimeError, match="not readable"):
        await controller.start_update_weights()

    assert controller._health_checker_activeness.get().active
    assert controller._open_update_weights_window_id is None
    await asyncio.wait_for(controller.prepare_eval(), timeout=1)


def test_fsdp_updater_flushes_only_after_every_engine_is_paused():
    """Each weight-update phase finishes on every engine before the next phase starts on any."""
    from unittest.mock import patch

    from miles.backends.fsdp_utils.update_weight_utils import UpdateWeightFromTensor

    order: list[str] = []
    pause_modes: list[str] = []

    class _Client:
        def __init__(self, index: int):
            self._index = index

        async def pause_generation(self, mode: str = "retract"):
            order.append(f"pause-{self._index}")
            pause_modes.append(mode)

        async def flush_cache(self):
            order.append(f"flush-{self._index}")

        async def begin_weight_update(self, selector: str = "all"):
            order.append(f"begin-{self._index}")

        async def end_weight_update(self):
            order.append(f"end-{self._index}")

        async def continue_generation(self):
            order.append(f"continue-{self._index}")

    updater = UpdateWeightFromTensor.__new__(UpdateWeightFromTensor)
    updater.args = Namespace(update_weight_buffer_size=1 << 30)
    updater.weight_version = 0
    updater.model = MagicMock()
    updater.model.state_dict.return_value = {}
    updater.rollout_engines = [_Client(0), _Client(1)]

    module = "miles.backends.fsdp_utils.update_weight_utils"
    with patch(f"{module}.dist") as dist_mock, patch(f"{module}.get_gloo_group", return_value=MagicMock()):
        dist_mock.get_rank.return_value = 0
        updater.update_weights()

    assert set(order[:2]) == {"pause-0", "pause-1"}
    assert set(order[2:4]) == {"flush-0", "flush-1"}
    assert set(order[4:6]) == {"begin-0", "begin-1"}
    assert set(order[6:8]) == {"end-0", "end-1"}
    assert set(order[8:]) == {"continue-0", "continue-1"}
    assert pause_modes == ["retract", "retract"]
