from __future__ import annotations

import shlex
import sys

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.specs.session import (
    SESSION_SERVER_PORT_NAME,
    compute_session_server_cell_specs,
    resolve_session_server_ports,
)
from miles.rollout.session.config import SessionServerConfig
from miles.utils.workers.argv_utils import parse_config_argv
from miles.utils.workers.worker_spec import CellAddressing, WorkerPlacement


def _session_args(**overrides):
    kwargs = dict(
        use_session_server=True,
        hf_checkpoint="/fake/model",
        sglang_router_ip="127.0.0.1",
        sglang_router_port=3000,
        session_server_port=None,
        miles_router_timeout=None,
        chat_template_path=None,
        tito_model="default",
        apply_chat_template_kwargs=None,
        tito_allowed_append_roles=["tool"],
        use_rollout_indexer_replay=False,
    )
    kwargs.update(overrides)
    return make_args(**kwargs)


def _addressing(port: int) -> CellAddressing:
    return CellAddressing(node_ips=["127.0.0.1"], master_ports={}, per_worker_ports=[{SESSION_SERVER_PORT_NAME: port}])


class TestResolveSessionServerPorts:
    def test_none_means_dynamic(self):
        """No configured port leaves the allocation to the worker manager."""
        assert resolve_session_server_ports(None) is None

    def test_single_value_is_a_single_server(self):
        """One value pins a single server on that port."""
        assert resolve_session_server_ports([30000]) == [30000]

    def test_two_values_expand_to_half_open_range(self):
        """A start/end pair expands to one server per port in [start, end)."""
        assert resolve_session_server_ports([30000, 30004]) == [30000, 30001, 30002, 30003]

    def test_empty_range_raises(self):
        """A reversed range would silently start zero servers, so it must fail."""
        with pytest.raises(ValueError, match="empty"):
            resolve_session_server_ports([30004, 30000])

    def test_more_than_two_values_raises(self):
        """Anything but one port or a range is a config error."""
        with pytest.raises(ValueError, match="one port or a start/end range"):
            resolve_session_server_ports([30000, 30001, 30002])


class TestSessionServerCellSpecs:
    def test_a_dynamic_setup_is_one_cell_with_a_dynamic_port(self):
        """No configured port means a single server on an allocator-chosen port."""
        (spec,) = compute_session_server_cell_specs(_session_args())
        (port_info,) = spec.worker.port_infos
        assert port_info.allow_dynamic

    def test_a_static_range_is_one_cell_per_port(self):
        """Each port of the range gets its own cell pinned to exactly that port."""
        specs = compute_session_server_cell_specs(_session_args(session_server_port=[5005, 5008]))
        assert [spec.cell_id for spec in specs] == ["session-server-0", "session-server-1", "session-server-2"]
        assert [spec.worker.port_infos[0].static_port for spec in specs] == [5005, 5006, 5007]
        assert not any(spec.worker.port_infos[0].allow_dynamic for spec in specs)

    def test_launch_plan_renders_a_parseable_config_with_the_router_backend(self):
        """The session server command's config parses back and dials the router."""
        (spec,) = compute_session_server_cell_specs(_session_args())
        placement = WorkerPlacement(local_index=0, global_rank=0, base_gpu_id=0)

        plan = spec.worker.build_launch_plan(placement, _addressing(5005))

        argv = shlex.split(plan.cmd)
        assert argv[:3] == [sys.executable, "-m", "miles.rollout.session.server"]
        config = parse_config_argv(SessionServerConfig, argv[3:])
        assert config.backend_url == "http://127.0.0.1:3000"
        assert config.host == "127.0.0.1"
        assert config.port == 5005

    def test_payload_instance_id_matches_the_launched_config(self):
        """The tracer matches servers by instance id, so payload and command must agree."""
        (spec,) = compute_session_server_cell_specs(_session_args())
        placement = WorkerPlacement(local_index=0, global_rank=0, base_gpu_id=0)

        plan = spec.worker.build_launch_plan(placement, _addressing(5005))
        (payload,) = spec.worker.build_member_payloads(_addressing(5005))

        config = parse_config_argv(SessionServerConfig, shlex.split(plan.cmd)[3:])
        assert payload["instance_id"] == config.instance_id
        assert payload["port"] == 5005
