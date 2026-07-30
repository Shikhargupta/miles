from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.wiring import start_router, start_session_server
from miles.utils.workers.worker_spec import CellAddressing


class _FakeWorkerManager:
    """Runs the spec's own payload builder against synthetic addressing, like start_cell would."""

    def __init__(self) -> None:
        self.specs: dict[str, object] = {}
        self.started: list[str] = []
        self._payloads: dict[str, list[dict]] = {}
        self._next_port = 20000

    async def register_cells(self, specs) -> None:
        for spec in specs:
            assert spec.cell_id not in self.specs
            self.specs[spec.cell_id] = spec
        for spec in specs:
            await self.start_cell(spec.cell_id)

    async def start_cell(self, cell_id: str) -> None:
        spec = self.specs[cell_id]
        ports = {}
        for info in spec.worker.port_infos:
            if info.allow_dynamic:
                ports[info.name] = self._next_port
                self._next_port += 1
            else:
                ports[info.name] = info.static_port
        addressing = CellAddressing(node_ips=["127.0.0.1"], master_ports={}, per_worker_ports=[ports])
        self._payloads[cell_id] = spec.worker.build_member_payloads(addressing)
        self.started.append(cell_id)

    def cell_workers(self, cell_id: str):
        return [SimpleNamespace(payload=payload) for payload in self._payloads[cell_id]]


class TestStartRouter:
    def test_returns_existing_when_already_configured(self):
        """Preconfigured router addressing skips the manager entirely."""
        args = make_args(use_miles_router=False, sglang_router_ip="10.1.2.3", sglang_router_port=4567)
        ip, port = asyncio.run(start_router(args, None, model_name="actor", force_new=False))
        assert (ip, port) == ("10.1.2.3", 4567)

    def test_pd_disagg_with_miles_router_asserts(self):
        """The miles router cannot serve a PD-disaggregated model."""
        args = make_args(use_miles_router=True, sglang_router_ip=None, sglang_router_port=None)
        with pytest.raises(AssertionError, match="miles router does not support PD"):
            asyncio.run(start_router(args, _FakeWorkerManager(), model_name="actor", has_pd_disaggregation=True))

    def test_static_port_conflict_raises_runtime_error(self):
        """A stale process on the configured port must fail loud, not launch behind it."""
        args = make_args(use_miles_router=False, sglang_router_ip=None, sglang_router_port=3123)
        with patch("miles.ray.wiring.is_port_available", return_value=False):
            with pytest.raises(RuntimeError, match="already in use"):
                asyncio.run(start_router(args, _FakeWorkerManager(), model_name="actor"))

    def test_starts_one_managed_cell_and_returns_its_addressing(self):
        """The router runs as a manager cell; the returned address is the cell's payload."""
        args = make_args(use_miles_router=False, sglang_router_ip=None, sglang_router_port=None)
        manager = _FakeWorkerManager()

        ip, port = asyncio.run(start_router(args, manager, model_name="actor"))

        assert manager.started == ["router-actor"]
        assert ip == "127.0.0.1"
        assert port == 20000

    def test_force_new_ignores_the_configured_port(self):
        """A second model's router must not collide with the first model's static port."""
        args = make_args(use_miles_router=False, sglang_router_ip="10.0.0.1", sglang_router_port=3123)
        manager = _FakeWorkerManager()

        ip, port = asyncio.run(start_router(args, manager, model_name="critic", force_new=True))

        spec = manager.specs["router-critic"]
        port_info = next(info for info in spec.worker.port_infos if info.name == "port")
        assert port_info.allow_dynamic
        assert port != 3123


class TestStartSessionServer:
    def test_disabled_returns_silently(self):
        """use_session_server=False must not touch the manager at all."""
        args = make_args(use_session_server=False)
        asyncio.run(start_session_server(args, None))

    def test_enabled_without_hf_checkpoint_raises(self):
        """The session server needs the tokenizer, so a missing checkpoint is a config error."""
        args = make_args(use_session_server=True, hf_checkpoint=None)
        with pytest.raises(ValueError, match="hf-checkpoint"):
            asyncio.run(start_session_server(args, _FakeWorkerManager()))

    def test_enabled_port_conflict_raises_runtime_error(self):
        """A stale session server on a configured port must fail loud."""
        args = make_args(
            use_session_server=True,
            hf_checkpoint="/fake/model",
            sglang_router_ip="127.0.0.1",
            sglang_router_port=20000,
            session_server_ip="127.0.0.1",
            session_server_port=[20001],
        )
        with patch("miles.ray.wiring.is_port_available", return_value=False):
            with pytest.raises(RuntimeError, match="already in use"):
                asyncio.run(start_session_server(args, _FakeWorkerManager()))

    def test_one_cell_per_static_port_and_args_carry_the_addressing(self):
        """Each resolved port gets its own managed cell; args carry ip, ports and instance ids."""
        args = make_args(
            use_session_server=True,
            hf_checkpoint="/fake/model",
            sglang_router_ip="127.0.0.1",
            sglang_router_port=3000,
            session_server_ip=None,
            session_server_port=[5005, 5007],
            miles_router_timeout=None,
            chat_template_path=None,
            tito_model="default",
            apply_chat_template_kwargs=None,
            tito_allowed_append_roles=["tool"],
            use_rollout_indexer_replay=False,
        )
        manager = _FakeWorkerManager()

        with patch("miles.ray.wiring.is_port_available", return_value=True):
            asyncio.run(start_session_server(args, manager))

        assert manager.started == ["session-server-0", "session-server-1"]
        assert args.session_server_ip == "127.0.0.1"
        assert args.session_server_ports == [5005, 5006]
        assert set(args.session_server_instance_ids) == {5005, 5006}
