from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout.cell_state import CellAddrInfo, StatePendingWeights, StateServing, StateUninitialized
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata
from miles.utils.ft_utils.health_checker import NoopHealthChecker, SimpleHealthChecker


def _make_meta() -> ServerCellMetadata:
    return ServerCellMetadata(
        model_id="default",
        worker_type="regular",
        cell_id="inference-engine-0-0-0",
        num_gpus_per_engine=1,
        gpu_offset=0,
        sglang_api_key=None,
        worker_name="inference-engine-0-0-0-0",
        needs_offload=False,
        update_weights=True,
        workers_hash="pseudo-hash-0",
    )


def _make_cell(*, ft_components: list[str], global_activeness: bool = True) -> ServerCell:
    return ServerCell(
        args=make_args(ft_components=ft_components),
        meta=_make_meta(),
        router_api_client=MagicMock(),
        global_health_checker_activeness=lambda: global_activeness,
    )


def _addr_info() -> CellAddrInfo:
    return CellAddrInfo(server_url="http://10.0.0.1:30000", bootstrap_port=None, gate_url="http://10.0.0.1:31000")


class TestRolloutCellHealthCheckerGating:
    async def test_a_cell_gets_no_checker_when_rollout_ft_is_off(self):
        """Probing engines nobody will heal only produces noise and load."""
        cell = _make_cell(ft_components=["train"])
        assert isinstance(cell._health_checker, NoopHealthChecker)

    async def test_a_cell_gets_a_real_checker_when_rollout_ft_is_on(self):
        """Rollout healing needs liveness, so the checker must actually be wired up."""
        cell = _make_cell(ft_components=["rollout"])
        try:
            assert isinstance(cell._health_checker, SimpleHealthChecker)
        finally:
            cell._health_checker.stop()

    async def test_the_checker_never_waits_a_grace_period(self):
        """Activeness flips every weight update window, so a grace period would restart forever."""
        cell = _make_cell(ft_components=["rollout"])
        try:
            assert cell._health_checker._config.first_wait == 0.0
        finally:
            cell._health_checker.stop()


class TestRolloutCellHealthCheckerActiveness:
    @pytest.mark.parametrize(
        "state, expected",
        [
            (StateUninitialized(), False),
            (StatePendingWeights(addr_info=_addr_info()), True),
            (StateServing(addr_info=_addr_info()), True),
        ],
    )
    async def test_only_a_started_engine_is_probed(self, state, expected):
        """An engine whose process is not up yet would fail every probe and look unhealthy."""
        cell = _make_cell(ft_components=["rollout"])
        try:
            cell._state = state
            assert cell._health_checker._get_activeness() is expected
        finally:
            cell._health_checker.stop()

    async def test_the_global_flag_can_silence_a_serving_cell(self):
        """During a weight update the engine is offloaded, so probing it would kill a healthy cell."""
        cell = _make_cell(ft_components=["rollout"], global_activeness=False)
        try:
            cell._state = StateServing(addr_info=_addr_info())
            assert cell._health_checker._get_activeness() is False
        finally:
            cell._health_checker.stop()


class TestRolloutCellHealthCheckerDisposal:
    async def test_disposing_a_cell_stops_its_checker(self):
        """A removed cell whose loop keeps polling leaks the task and the whole cell it closes over."""
        cell = _make_cell(ft_components=["rollout"])
        assert cell._health_checker._task is not None

        await cell.dispose()

        assert cell._health_checker._task is None
