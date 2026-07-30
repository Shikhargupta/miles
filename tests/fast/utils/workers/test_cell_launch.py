from __future__ import annotations

from tests.fast.ray.rollout.conftest import fake_engine

import miles.utils.workers.cell_launch as cell_launch
from miles.utils.workers.addr_allocator import PortAllocator
from miles.utils.workers.cell_launch import (
    _resolve_worker_class,
    allocate_cell_ports,
    cell_worker_placements,
    probe_node_ips,
)
from miles.utils.workers.command_actor import CommandActor
from miles.utils.workers.worker_spec import (
    BaseCellSpec,
    CommandWorkerSpec,
    PortInfo,
    RayActorOptions,
    SchedulingSpec,
    ServeWorkerSpec,
    WorkerLaunchPlan,
    WorkerPlacement,
)


def _port_infos(*, with_master: bool = True) -> list[PortInfo]:
    infos = [
        PortInfo(name="alpha", static_port=30000, mode="per_worker", allow_dynamic=True),
        PortInfo(name="beta", static_port=30500, mode="per_worker", allow_dynamic=True),
    ]
    if with_master:
        infos.append(PortInfo(name="master", static_port=31500, mode="master", allow_dynamic=True, num_consecutive=5))
    return infos


class TestAllocateCellPorts:
    def test_every_worker_gets_all_per_worker_ports(self, patch_ray_get):
        """Each worker's dict covers exactly the per-worker port names."""
        actors = [fake_engine(host="10.0.0.1", port_seed=0), fake_engine(host="10.0.0.2", port_seed=0)]
        addressing = allocate_cell_ports(
            port_allocator=PortAllocator(),
            port_infos=_port_infos(),
            actors=actors,
            node_ips=["10.0.0.1", "10.0.0.2"],
        )
        assert [set(ports) for ports in addressing.per_worker_ports] == [{"alpha", "beta"}, {"alpha", "beta"}]

    def test_master_ports_are_allocated_once_on_the_first_worker_node(self, patch_ray_get):
        """The master block lives on the primary worker's node, before its per-worker ports."""
        actors = [fake_engine(host="10.0.0.1", port_seed=15000), fake_engine(host="10.0.0.2", port_seed=15000)]
        addressing = allocate_cell_ports(
            port_allocator=PortAllocator(),
            port_infos=_port_infos(),
            actors=actors,
            node_ips=["10.0.0.1", "10.0.0.2"],
        )
        assert set(addressing.master_ports) == {"master"}
        assert addressing.master_ports["master"] < min(addressing.per_worker_ports[0].values())

    def test_consecutive_blocks_move_the_cursor_past_the_whole_block(self, patch_ray_get):
        """A num_consecutive=5 master reservation must not overlap the next allocations."""
        actor = fake_engine(host="10.0.0.1", port_seed=15000)
        addressing = allocate_cell_ports(
            port_allocator=PortAllocator(),
            port_infos=_port_infos(),
            actors=[actor],
            node_ips=["10.0.0.1"],
        )
        assert addressing.per_worker_ports[0]["alpha"] >= addressing.master_ports["master"] + 5

    def test_no_master_infos_yield_empty_master_ports(self, patch_ray_get):
        """Worker kinds without a master endpoint allocate nothing extra."""
        actor = fake_engine(host="10.0.0.1", port_seed=0)
        addressing = allocate_cell_ports(
            port_allocator=PortAllocator(),
            port_infos=_port_infos(with_master=False),
            actors=[actor],
            node_ips=["10.0.0.1"],
        )
        assert addressing.master_ports == {}

    def test_ports_across_workers_on_one_node_never_collide(self, patch_ray_get):
        """A shared allocator hands out disjoint ports for co-located workers."""
        actors = [fake_engine(host="10.0.0.1", port_seed=0) for _ in range(3)]
        allocator = PortAllocator()
        addressing = allocate_cell_ports(
            port_allocator=allocator,
            port_infos=_port_infos(),
            actors=actors,
            node_ips=["10.0.0.1"] * 3,
        )
        all_ports = list(addressing.master_ports.values()) + [
            port for ports in addressing.per_worker_ports for port in ports.values()
        ]
        assert len(all_ports) == len(set(all_ports))


class TestProbeNodeIps:
    async def test_returns_one_ip_per_actor_in_order(self):
        """The addressing stage relies on node_ips being parallel to the actor list."""
        actors = [fake_engine(host=f"10.0.0.{i}", port_seed=0) for i in (1, 2, 3)]
        assert await probe_node_ips(actors) == ["10.0.0.1", "10.0.0.2", "10.0.0.3"]


def _class_cell_spec(
    *, num_workers: int, num_gpus_per_worker: float, rank_offset: int, gpu_offset: int
) -> BaseCellSpec:
    worker = ServeWorkerSpec(
        name="demo",
        port_infos=[],
        env_var=lambda placement: {},
        scheduling=SchedulingSpec(
            num_cells=1, num_workers_per_cell=num_workers, num_gpus_per_worker=num_gpus_per_worker
        ),
        ray_options=RayActorOptions(num_cpus=0.2, num_gpus=0.2),
        worker_class="miles.demo.Worker",
        ctor_kwargs=lambda placement: {},
        build_init_payloads=lambda addressing: [],
    )
    return BaseCellSpec(worker=worker, cell_id="cell-0", rank_offset=rank_offset, gpu_offset=gpu_offset)


class TestCreateCellWorkerActors:
    def _record_creations(self, monkeypatch) -> list[dict]:
        created: list[dict] = []

        def _create(*, worker, placement, pg_handle, bundle_index):
            created.append(dict(placement=placement, bundle_index=bundle_index, pg_handle=pg_handle))
            return f"actor-{placement.local_index}"

        monkeypatch.setattr(cell_launch, "create_cell_worker_actor", _create)
        return created

    def test_ranks_and_gpus_come_from_the_cell_offsets(self, monkeypatch):
        """A cell owns the workers its offsets point at, so the placement is derived, never passed in."""
        created = self._record_creations(monkeypatch)
        pg_handle = object()
        spec = _class_cell_spec(num_workers=2, num_gpus_per_worker=1, rank_offset=3, gpu_offset=2)

        actors = cell_launch.create_cell_worker_actors(spec=spec, pg=(pg_handle, [10, 11, 12, 13], [4, 5, 6, 7]))

        assert actors == ["actor-0", "actor-1"]
        assert [(c["placement"].global_rank, c["placement"].base_gpu_id, c["bundle_index"]) for c in created] == [
            (3, 6, 12),
            (4, 7, 13),
        ]
        assert all(c["pg_handle"] is pg_handle for c in created)

    def test_a_multi_gpu_worker_strides_the_gpu_index_by_its_gpu_count(self, monkeypatch):
        """One worker per node of a multi-node engine claims a whole node of gpus."""
        created = self._record_creations(monkeypatch)
        spec = _class_cell_spec(num_workers=2, num_gpus_per_worker=8, rank_offset=0, gpu_offset=0)

        cell_launch.create_cell_worker_actors(spec=spec, pg=(object(), list(range(16)), list(range(100, 116))))

        assert [(c["placement"].global_rank, c["placement"].base_gpu_id, c["bundle_index"]) for c in created] == [
            (0, 100, 0),
            (1, 108, 8),
        ]


async def _noop_wait(addressing, is_worker_alive):
    return None


def _command_worker(**overrides) -> CommandWorkerSpec:
    kwargs = dict(
        name="command-worker",
        port_infos=[],
        env_var=lambda placement: {"RANK": str(placement.global_rank)},
        scheduling=SchedulingSpec(num_cells=1, num_workers_per_cell=1, num_gpus_per_worker=1),
        ray_options=RayActorOptions(num_cpus=0.2, num_gpus=0.2),
        build_launch_plan=lambda placement, addressing: WorkerLaunchPlan(cmd="true"),
        build_member_payloads=lambda addressing: [],
        wait_cell_ready=_noop_wait,
    )
    kwargs.update(overrides)
    return CommandWorkerSpec(**kwargs)


class TestCellWorkerPlacements:
    def test_placements_follow_the_cell_offsets_and_stride(self):
        """Placement is derived from the cell's own offsets, never passed in."""
        spec = _class_cell_spec(num_workers=2, num_gpus_per_worker=2, rank_offset=3, gpu_offset=2)
        placements = cell_worker_placements(spec=spec, pg=(object(), [], [0, 1, 4, 5, 6, 7]))
        assert [(p.local_index, p.global_rank, p.base_gpu_id) for p in placements] == [(0, 3, 4), (1, 4, 6)]


class TestCreateCellWorkerActorKinds:
    def test_a_command_worker_becomes_a_command_actor_with_empty_ctor_kwargs(self, monkeypatch):
        """A command spec launches the generic CommandActor; the command comes later."""
        captured: dict = {}

        def _create_pg(**kwargs):
            captured.update(kwargs)
            return "actor"

        monkeypatch.setattr(cell_launch, "create_pg_worker_actor", _create_pg)
        placement = WorkerPlacement(local_index=0, global_rank=7, base_gpu_id=0)

        cell_launch.create_cell_worker_actor(
            worker=_command_worker(), placement=placement, pg_handle=object(), bundle_index=1
        )

        assert captured["worker_cls"] is CommandActor
        assert captured["ctor_kwargs"] == {}
        assert captured["env_vars"] == {"RANK": "7"}

    def test_a_serve_worker_resolves_its_class_and_placement_ctor_kwargs(self, monkeypatch):
        """A serve spec resolves its dotted class path and builds ctor kwargs from the placement."""
        captured: dict = {}

        def _create_pg(**kwargs):
            captured.update(kwargs)
            return "actor"

        monkeypatch.setattr(cell_launch, "create_pg_worker_actor", _create_pg)
        spec = _class_cell_spec(num_workers=1, num_gpus_per_worker=1)
        worker = spec.worker.model_copy(
            update=dict(
                worker_class="miles.utils.workers.worker_spec.PortInfo",
                ctor_kwargs=lambda placement: {"rank": placement.global_rank},
            )
        )
        placement = WorkerPlacement(local_index=0, global_rank=5, base_gpu_id=0)

        cell_launch.create_cell_worker_actor(worker=worker, placement=placement, pg_handle=object(), bundle_index=0)

        assert captured["worker_cls"] is PortInfo
        assert captured["ctor_kwargs"] == {"rank": 5}


class TestResolveWorkerClass:
    def test_resolves_a_dotted_path_to_the_class(self):
        """The spec names its worker class by path, so the class is looked up at launch time."""
        assert _resolve_worker_class("miles.utils.workers.worker_spec.PortInfo") is PortInfo
