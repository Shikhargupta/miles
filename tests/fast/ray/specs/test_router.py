from __future__ import annotations

import shlex
import sys

from tests.fast.ray.rollout.conftest import make_args

from miles.ray.specs.router import ROUTER_PORT_NAME, ROUTER_PROMETHEUS_PORT_NAME, compute_router_cell_spec
from miles.router.config import MilesRouterConfig
from miles.utils.workers.argv_utils import parse_config_argv
from miles.utils.workers.worker_spec import CellAddressing, WorkerPlacement


def _addressing(ports: dict[str, int]) -> CellAddressing:
    return CellAddressing(node_ips=["127.0.0.1"], master_ports={}, per_worker_ports=[ports])


def _placement() -> WorkerPlacement:
    return WorkerPlacement(local_index=0, global_rank=0, base_gpu_id=0)


class TestRouterCellSpec:
    def test_a_static_port_is_declared_non_dynamic(self):
        """A user-fixed router port must be handed out verbatim, never reallocated."""
        args = make_args(use_miles_router=False)
        spec = compute_router_cell_spec(args, cell_id="router-actor", has_pd_disaggregation=False, static_port=3123)
        port_info = next(info for info in spec.worker.port_infos if info.name == ROUTER_PORT_NAME)
        assert port_info.static_port == 3123
        assert not port_info.allow_dynamic

    def test_the_sglang_router_also_declares_a_prometheus_port(self):
        """The upstream router CLI requires a prometheus port, so the spec must allocate one."""
        args = make_args(use_miles_router=False)
        spec = compute_router_cell_spec(args, cell_id="router-actor", has_pd_disaggregation=False, static_port=None)
        assert {info.name for info in spec.worker.port_infos} == {ROUTER_PORT_NAME, ROUTER_PROMETHEUS_PORT_NAME}

    def test_the_miles_router_declares_no_prometheus_port(self):
        """The miles router serves no prometheus endpoint, so no port may be wasted on it."""
        args = make_args(use_miles_router=True)
        spec = compute_router_cell_spec(args, cell_id="router-actor", has_pd_disaggregation=False, static_port=None)
        assert {info.name for info in spec.worker.port_infos} == {ROUTER_PORT_NAME}

    def test_gpuless_scheduling(self):
        """The router needs no GPU, which is what routes it onto the head node."""
        args = make_args(use_miles_router=False)
        spec = compute_router_cell_spec(args, cell_id="router-actor", has_pd_disaggregation=False, static_port=None)
        assert spec.worker.scheduling.num_gpus_per_worker == 0
        assert spec.worker.ray_options.num_gpus == 0


class TestRouterLaunchPlan:
    def test_sgl_router_renders_the_native_cli_with_the_allocated_ports(self):
        """The sgl router runs as the upstream CLI with the addressing the allocator chose."""
        args = make_args(use_miles_router=False)
        spec = compute_router_cell_spec(args, cell_id="router-actor", has_pd_disaggregation=False, static_port=None)

        plan = spec.worker.build_launch_plan(
            _placement(), _addressing({ROUTER_PORT_NAME: 20000, ROUTER_PROMETHEUS_PORT_NAME: 4001})
        )

        argv = shlex.split(plan.cmd)
        assert argv[0] == sys.executable
        assert argv[1:3] == ["-m", "sglang_router.launch_router"]
        assert argv[argv.index("--port") + 1] == "20000"
        assert argv[argv.index("--prometheus-port") + 1] == "4001"

    def test_miles_router_renders_a_parseable_config(self):
        """The miles router command's config payload parses back losslessly."""
        args = make_args(
            use_miles_router=True,
            miles_router_max_connections=100,
            miles_router_timeout=None,
            miles_router_health_check_failure_threshold=3,
            rollout_health_check_interval=10.0,
        )
        spec = compute_router_cell_spec(args, cell_id="router-actor", has_pd_disaggregation=False, static_port=None)

        plan = spec.worker.build_launch_plan(_placement(), _addressing({ROUTER_PORT_NAME: 20000}))

        argv = shlex.split(plan.cmd)
        assert argv[:3] == [sys.executable, "-m", "miles.router.router"]
        config = parse_config_argv(MilesRouterConfig, argv[3:])
        assert config.host == "127.0.0.1"
        assert config.port == 20000
        assert config.max_connections == 100

    def test_member_payload_matches_the_launch_addressing(self):
        """Consumers dial the payload, so it must match what the command binds."""
        args = make_args(use_miles_router=True)
        spec = compute_router_cell_spec(args, cell_id="router-actor", has_pd_disaggregation=False, static_port=None)
        (payload,) = spec.worker.build_member_payloads(_addressing({ROUTER_PORT_NAME: 20000}))
        assert payload == {"host": "127.0.0.1", "port": 20000}
