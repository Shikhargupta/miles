from __future__ import annotations

import pytest
from tests.fast.ray.specs.conftest import make_args

from miles.ray.specs.session import compute_session_server_specs


class TestComputeSessionServerSpecs:
    def test_empty_when_session_server_disabled(self):
        """Without --use-session-server no session server workers exist."""
        assert compute_session_server_specs(make_args()) == []

    def test_launch_command_wires_the_config_via_a_placeholder(self):
        """The command binds its whole config through the config-json placeholder."""
        (spec,) = compute_session_server_specs(make_args(use_session_server=True))
        assert spec.launch_command.startswith("python -m miles.rollout.session.server")
        assert "--config-json {config_json}" in spec.launch_command

    def test_default_is_one_instance_with_a_dynamic_port(self):
        """No explicit port means one instance on an auto-allocated port."""
        (spec,) = compute_session_server_specs(make_args(use_session_server=True))
        assert spec.name == "session-server"
        assert spec.scheduling.num_cells == 1
        (port_info,) = spec.port_infos
        assert port_info.allow_dynamic is True

    def test_single_explicit_port_is_static(self):
        """One explicit port pins one instance to that port."""
        (spec,) = compute_session_server_specs(make_args(use_session_server=True, session_server_port=[5005]))
        assert spec.scheduling.num_cells == 1
        (port_info,) = spec.port_infos
        assert port_info.static_port == 5005
        assert port_info.allow_dynamic is False

    def test_port_range_spawns_one_instance_per_port(self):
        """A [start, end) range yields end - start instances."""
        (spec,) = compute_session_server_specs(make_args(use_session_server=True, session_server_port=[5000, 5004]))
        assert spec.scheduling.num_cells == 4

    def test_empty_port_range_is_rejected(self):
        """An empty range is a configuration error."""
        with pytest.raises(AssertionError, match="is empty"):
            compute_session_server_specs(make_args(use_session_server=True, session_server_port=[5004, 5000]))

    def test_too_many_port_values_are_rejected(self):
        """More than two values cannot be interpreted."""
        with pytest.raises(ValueError, match="one port or a start/end range"):
            compute_session_server_specs(make_args(use_session_server=True, session_server_port=[1, 2, 3]))
