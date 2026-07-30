import pytest
from pydantic import ValidationError

from miles.utils.workers.worker_spec import (
    BaseWorkerSpec,
    CellAddressing,
    CommandWorkerSpec,
    PortInfo,
    RayActorOptions,
    SchedulingSpec,
    ServeWorkerSpec,
    WorkerLaunchPlan,
    WorkerPlacement,
)


def _make_placement(**overrides) -> WorkerPlacement:
    kwargs = dict(local_index=0, global_rank=3, base_gpu_id=4)
    kwargs.update(overrides)
    return WorkerPlacement(**kwargs)


def _make_port_info(**overrides) -> PortInfo:
    kwargs = dict(name="http", static_port=8000, mode="per_worker", allow_dynamic=False)
    kwargs.update(overrides)
    return PortInfo(**kwargs)


def _make_base_kwargs(**overrides) -> dict:
    kwargs = dict(
        name="demo-worker",
        port_infos=[_make_port_info()],
        env_var=lambda placement: {"DEMO": "1"},
        scheduling=SchedulingSpec(num_cells=2, num_workers_per_cell=4, num_gpus_per_worker=0.4),
        ray_options=RayActorOptions(num_cpus=0.2, num_gpus=0.2),
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

    def test_num_consecutive_defaults_to_one(self):
        """A port reserves a single slot unless a block is requested."""
        assert _make_port_info().num_consecutive == 1
        assert _make_port_info(num_consecutive=32).num_consecutive == 32

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

    def test_env_var_is_stored_uncalled_and_sees_the_placement(self):
        """The env_var callable is stored as-is and only evaluated once a worker is placed."""
        seen = []

        def env_var(placement: WorkerPlacement) -> dict[str, str]:
            seen.append(placement)
            return {"A": "b"}

        spec = BaseWorkerSpec(**_make_base_kwargs(env_var=env_var))
        assert seen == []
        assert spec.env_var(_make_placement()) == {"A": "b"}
        assert seen == [_make_placement()]

    def test_rejects_extra_field(self):
        """Unknown fields are forbidden."""
        with pytest.raises(ValidationError):
            BaseWorkerSpec(**_make_base_kwargs(unknown_field=1))

    def test_is_frozen(self):
        """Field assignment after construction is rejected."""
        spec = BaseWorkerSpec(**_make_base_kwargs())
        with pytest.raises(ValidationError):
            spec.name = "other"


def _make_command_kwargs(**overrides) -> dict:
    kwargs = dict(
        build_launch_plan=lambda placement, addressing: WorkerLaunchPlan(cmd="true"),
        build_member_payloads=lambda addressing: [],
        wait_cell_ready=_noop_wait,
    )
    kwargs.update(overrides)
    return kwargs


async def _noop_wait(addressing, is_worker_alive):
    return None


class TestCommandWorkerSpec:
    def test_constructs_with_the_launch_contract(self):
        """A command spec carries the launch callables besides base fields."""
        spec = CommandWorkerSpec(**_make_base_kwargs(), **_make_command_kwargs())
        assert isinstance(spec, BaseWorkerSpec)

    def test_build_launch_plan_is_stored_uncalled_and_sees_placement_and_addressing(self):
        """The plan builder is stored as-is and turns placement plus addressing into a command."""
        seen = []

        def build_launch_plan(placement: WorkerPlacement, addressing: CellAddressing) -> WorkerLaunchPlan:
            seen.append((placement, addressing))
            return WorkerLaunchPlan(cmd=f"serve --rank {placement.global_rank}", envs={"A": "1"})

        spec = CommandWorkerSpec(**_make_base_kwargs(), **_make_command_kwargs(build_launch_plan=build_launch_plan))
        assert seen == []
        addressing = CellAddressing(node_ips=["10.0.0.1"], master_ports={}, per_worker_ports=[{}])
        plan = spec.build_launch_plan(_make_placement(), addressing)
        assert plan.cmd == "serve --rank 3"
        assert plan.envs == {"A": "1"}

    def test_build_member_payloads_turns_addressing_into_one_payload_per_worker(self):
        """The payload builder is what keeps the manager free of worker knowledge."""
        spec = CommandWorkerSpec(
            **_make_base_kwargs(),
            **_make_command_kwargs(
                build_member_payloads=lambda addressing: [{"host": ip} for ip in addressing.node_ips],
            ),
        )
        addressing = CellAddressing(node_ips=["10.0.0.1", "10.0.0.2"], master_ports={}, per_worker_ports=[{}, {}])
        assert spec.build_member_payloads(addressing) == [{"host": "10.0.0.1"}, {"host": "10.0.0.2"}]

    async def test_wait_cell_ready_is_awaitable_with_the_liveness_probe(self):
        """Readiness is the spec's job, so the manager only supplies a liveness probe."""
        waited = []

        async def wait_cell_ready(addressing, is_worker_alive):
            waited.append(is_worker_alive())

        spec = CommandWorkerSpec(**_make_base_kwargs(), **_make_command_kwargs(wait_cell_ready=wait_cell_ready))
        addressing = CellAddressing(node_ips=["10.0.0.1"], master_ports={}, per_worker_ports=[{}])
        await spec.wait_cell_ready(addressing, lambda: True)
        assert waited == [True]


def _make_class_kwargs(**overrides) -> dict:
    kwargs = dict(
        worker_class="miles.demo.Worker",
        ctor_kwargs=lambda placement: {},
        build_init_payloads=lambda addressing: [],
    )
    kwargs.update(overrides)
    return kwargs


class TestServeWorkerSpec:
    def test_constructs_with_worker_class(self):
        """A serve spec carries the worker class path besides base fields."""
        spec = ServeWorkerSpec(**_make_base_kwargs(), **_make_class_kwargs())
        assert spec.worker_class == "miles.demo.Worker"
        assert isinstance(spec, BaseWorkerSpec)

    def test_ctor_kwargs_is_stored_uncalled_and_sees_the_placement(self):
        """The ctor_kwargs callable is stored as-is and turns a placement into arguments."""
        seen = []

        def ctor_kwargs(placement: WorkerPlacement) -> dict:
            seen.append(placement)
            return {"rank": placement.global_rank}

        spec = ServeWorkerSpec(**_make_base_kwargs(), **_make_class_kwargs(ctor_kwargs=ctor_kwargs))
        assert seen == []
        assert spec.ctor_kwargs(_make_placement()) == {"rank": 3}

    def test_build_init_payloads_turns_addressing_into_one_payload_per_worker(self):
        """The payload builder is what keeps the manager free of worker knowledge."""
        spec = ServeWorkerSpec(
            **_make_base_kwargs(),
            **_make_class_kwargs(
                build_init_payloads=lambda addressing: [{"host": ip} for ip in addressing.node_ips],
            ),
        )
        addressing = CellAddressing(node_ips=["10.0.0.1", "10.0.0.2"], master_ports={}, per_worker_ports=[{}, {}])
        assert spec.build_init_payloads(addressing) == [{"host": "10.0.0.1"}, {"host": "10.0.0.2"}]
