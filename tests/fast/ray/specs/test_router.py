from __future__ import annotations

import shlex

import pytest
from tests.fast.ray.rollout.conftest import make_sglang_config_yaml
from tests.fast.ray.specs.conftest import make_args

from miles.ray.specs.router import compute_router_specs
from miles.router.config import MilesRouterConfig
from miles.utils.workers.argv_utils import parse_config_argv


class TestComputeRouterSpecs:
    def test_one_router_per_model(self, tmp_path):
        """Every configured model gets its own router."""
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            make_sglang_config_yaml(name="actor") + make_sglang_config_yaml(name="ref").replace("sglang:\n", "")
        )
        args = make_args(sglang_config=str(cfg_path), rollout_num_gpus=16)
        specs = compute_router_specs(args)
        assert [spec.name for spec in specs] == ["router-actor", "router-ref"]

    def test_single_cpu_worker_with_router_and_prometheus_ports(self):
        """The sgl router exposes its serving port plus the prometheus port."""
        (spec,) = compute_router_specs(make_args())
        assert spec.scheduling.num_cells == 1
        assert spec.scheduling.num_gpus_per_worker == 0
        assert [port_info.name for port_info in spec.port_infos] == ["router", "prometheus"]

    def test_miles_router_has_no_prometheus_port(self):
        """The miles router variant does not expose prometheus metrics."""
        (spec,) = compute_router_specs(make_args(use_miles_router=True))
        assert [port_info.name for port_info in spec.port_infos] == ["router"]

    def test_explicit_router_port_becomes_the_static_port(self):
        """--sglang-router-port pins the static router port."""
        (spec,) = compute_router_specs(make_args(sglang_router_port=4321))
        (router_port,) = [port_info for port_info in spec.port_infos if port_info.name == "router"]
        assert router_port.static_port == 4321

    def test_empty_when_debug_train_only(self):
        """debug_train_only launches no routers."""
        assert compute_router_specs(make_args(debug_train_only=True)) == []

    def test_external_rollout_still_gets_a_router(self):
        """External engines are routed through a miles-owned router."""
        specs = compute_router_specs(make_args(rollout_external=True))
        assert [spec.name for spec in specs] == ["router-default"]


class TestRouterLaunchCommand:
    def test_sgl_router_command_binds_the_static_ports(self):
        """The sgl router command carries the bind host and static ports."""
        (spec,) = compute_router_specs(make_args())
        assert spec.launch_command.startswith("python -m sglang_router.launch_router")
        assert "--host 0.0.0.0" in spec.launch_command
        assert "--port 30080" in spec.launch_command
        assert "--prometheus-port 30081" in spec.launch_command
        assert "--log-level warn" in spec.launch_command

    def test_explicit_router_port_lands_in_the_command(self):
        """--sglang-router-port flows into the rendered command."""
        (spec,) = compute_router_specs(make_args(sglang_router_port=4321))
        assert "--port 4321" in spec.launch_command

    def test_request_timeout_and_policy_come_from_args(self):
        """Timeout is always rendered and the policy only when non-default."""
        (spec,) = compute_router_specs(make_args(sglang_router_request_timeout_secs=123))
        assert "--request-timeout-secs 123" in spec.launch_command
        assert "--policy" not in spec.launch_command
        (spec,) = compute_router_specs(make_args(sglang_router_policy="power_of_two"))
        assert "--policy power_of_two" in spec.launch_command

    def test_pd_model_enables_pd_disaggregation(self):
        """A PD-disaggregated model turns the router flag on."""
        (spec,) = compute_router_specs(make_args(prefill_num_servers=2))
        assert "--pd-disaggregation" in spec.launch_command

    def test_miles_router_command_carries_a_parseable_config(self):
        """The miles router command's config payload parses back losslessly."""
        (spec,) = compute_router_specs(make_args(use_miles_router=True))
        argv = shlex.split(spec.launch_command)
        assert argv[:3] == ["python", "-m", "miles.router.router"]
        config = parse_config_argv(MilesRouterConfig, argv[3:])
        assert config.host == "0.0.0.0"
        assert config.port == 30080

    def test_miles_router_rejects_pd_disaggregation(self):
        """miles router cannot serve a PD-disaggregated model."""
        with pytest.raises(AssertionError, match="PD disaggregation"):
            compute_router_specs(make_args(use_miles_router=True, prefill_num_servers=2))
