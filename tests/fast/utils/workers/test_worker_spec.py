import pytest
from pydantic import ValidationError

from miles.utils.workers.worker_spec import (
    BaseWorkerSpec,
    CommandWorkerSpec,
    PortInfo,
    SchedulingSpec,
    ServeWorkerSpec,
)


def _make_port_info(**overrides) -> PortInfo:
    kwargs = dict(name="http", static_port=8000, mode="per_worker", allow_dynamic=False)
    kwargs.update(overrides)
    return PortInfo(**kwargs)


def _make_base_kwargs(**overrides) -> dict:
    kwargs = dict(
        name="demo-worker",
        port_infos=[_make_port_info()],
        env_var=lambda: {"DEMO": "1"},
        scheduling=SchedulingSpec(num_cells=2, num_workers_per_cell=4, num_gpus_per_worker=0.4),
    )
    kwargs.update(overrides)
    return kwargs


class TestPortInfo:
    def test_accepts_both_modes(self):
        """Both per_worker and master are valid modes."""
        assert _make_port_info(mode="per_worker").mode == "per_worker"
        assert _make_port_info(mode="master").mode == "master"

    def test_rejects_unknown_mode(self):
        """An unknown mode literal is rejected."""
        with pytest.raises(ValidationError):
            _make_port_info(mode="broadcast")

    def test_rejects_extra_field(self):
        """Unknown fields are forbidden."""
        with pytest.raises(ValidationError):
            _make_port_info(unknown_field=1)

    def test_is_frozen(self):
        """Field assignment after construction is rejected."""
        port_info = _make_port_info()
        with pytest.raises(ValidationError):
            port_info.static_port = 9000


class TestBaseWorkerSpec:
    def test_constructs_and_exposes_fields(self):
        """A spec keeps its name, ports, and scheduling as provided."""
        spec = BaseWorkerSpec(**_make_base_kwargs())
        assert spec.name == "demo-worker"
        assert spec.port_infos[0].static_port == 8000
        assert spec.scheduling.num_cells == 2

    def test_env_var_is_stored_uncalled(self):
        """The env_var callable is stored as-is and only evaluated on demand."""
        calls = []

        def env_var() -> dict[str, str]:
            calls.append(1)
            return {"A": "b"}

        spec = BaseWorkerSpec(**_make_base_kwargs(env_var=env_var))
        assert calls == []
        assert spec.env_var() == {"A": "b"}

    def test_rejects_extra_field(self):
        """Unknown fields are forbidden."""
        with pytest.raises(ValidationError):
            BaseWorkerSpec(**_make_base_kwargs(unknown_field=1))

    def test_is_frozen(self):
        """Field assignment after construction is rejected."""
        spec = BaseWorkerSpec(**_make_base_kwargs())
        with pytest.raises(ValidationError):
            spec.name = "other"


class TestCommandWorkerSpec:
    def test_constructs_with_launch_command(self):
        """A command spec carries the launch command besides base fields."""
        spec = CommandWorkerSpec(**_make_base_kwargs(), launch_command="python -m sglang.launch_server")
        assert spec.launch_command == "python -m sglang.launch_server"
        assert isinstance(spec, BaseWorkerSpec)


class TestServeWorkerSpec:
    def test_constructs_with_worker_class(self):
        """A serve spec carries the worker class path besides base fields."""
        spec = ServeWorkerSpec(
            **_make_base_kwargs(),
            worker_class="miles.ray.rollout.inference_controller.InferenceController",
            ctor_kwargs=lambda: {},
        )
        assert spec.worker_class == "miles.ray.rollout.inference_controller.InferenceController"
        assert isinstance(spec, BaseWorkerSpec)

    def test_ctor_kwargs_is_stored_uncalled(self):
        """The ctor_kwargs callable is stored as-is and only evaluated on demand."""
        calls = []

        def ctor_kwargs() -> dict:
            calls.append(1)
            return {"x": 1}

        spec = ServeWorkerSpec(
            **_make_base_kwargs(),
            worker_class="miles.demo.Worker",
            ctor_kwargs=ctor_kwargs,
        )
        assert calls == []
        assert spec.ctor_kwargs() == {"x": 1}
