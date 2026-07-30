from __future__ import annotations

from tests.fast.ray.rollout.conftest import fake_engine

from miles.utils.workers.addr_allocator import PortAllocator
from miles.utils.workers.cell_launch import allocate_cell_ports, probe_node_ips
from miles.utils.workers.worker_spec import PortInfo


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
