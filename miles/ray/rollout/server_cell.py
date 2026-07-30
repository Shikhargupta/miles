import asyncio
import dataclasses
import functools
import logging
import os
from dataclasses import dataclass
from typing import Any, Literal

import ray
from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient, wait_server_healthy
from miles.backends.sglang_utils.sglang_engine import build_server_url, compute_engine_launch_plan, format_v6_uri
from miles.backends.sglang_utils.sglang_router_api_client import SGLangRouterApiClient, use_legacy_router_api
from miles.ray.rollout.cell_state import (
    AddrInfo,
    CellState,
    StateAllocatedAlive,
    StateAllocatedBase,
    StateAllocatedUninitialized,
    StateStopped,
)
from miles.ray.specs.inference import (
    ENGINE_DISAGGREGATION_BOOTSTRAP_PORT_NAME,
    ENGINE_DIST_INIT_PORT_NAME,
    ENGINE_INFO_BOOTSTRAP_PORT_NAME,
    ENGINE_NCCL_PORT_NAME,
    ENGINE_SERVER_PORT_NAME,
    InferenceCellSpec,
    InferenceWorkerSpec,
)
from miles.utils.workers.addr_allocator import PortAllocator
from miles.utils.workers.cell_launch import CellAddressing, allocate_cell_ports, create_pg_worker_actor, probe_node_ips
from miles.utils.workers.command_actor import CommandActor

logger = logging.getLogger(__name__)

SHUTDOWN_TIMEOUT = 30


@dataclass
class ServerCell:
    args: Any
    spec: InferenceCellSpec
    update_weights: bool = True
    pg: Any = None  # (placement_group, reordered_bundle_indices, reordered_gpu_ids)
    _state: CellState = dataclasses.field(default_factory=StateStopped)

    # ============================= temporary spec pass-throughs =============================
    # These keep the old attribute names alive while the callers still read them off the cell.
    # They go away as the callers move to the spec.

    @property
    def cell_id(self) -> str:
        return self.spec.cell_id

    @property
    def worker_type(self) -> Literal["regular", "prefill", "decode"]:
        return self.spec.worker.worker_type

    @property
    def num_nodes(self) -> int:
        return self.spec.worker.scheduling.num_workers_per_cell

    @property
    def num_gpus_per_engine(self) -> int:
        return self.spec.worker.num_gpus_per_engine

    @property
    def rank_offset(self) -> int:
        return self.spec.rank_offset

    @property
    def gpu_offset(self) -> int:
        return self.spec.gpu_offset

    @property
    def sglang_overrides(self) -> dict:
        return self.spec.worker.sglang_overrides

    @property
    def needs_offload(self) -> bool:
        return self.spec.worker.needs_offload

    @property
    def model_path(self) -> str | None:
        return self.spec.worker.model_path

    # ======================= end of temporary spec pass-throughs ===========================

    @property
    def is_allocated(self) -> bool:
        return isinstance(self._state, StateAllocatedBase)

    @property
    def is_alive(self) -> bool:
        return isinstance(self._state, StateAllocatedAlive)

    @property
    def actor_handles(self) -> list[ray.actor.ActorHandle]:
        assert isinstance(self._state, StateAllocatedBase)
        return self._state.actor_handles

    @property
    def primary_actor_handle(self) -> ray.actor.ActorHandle:
        return self.actor_handles[0]

    @property
    def engine_gpu_ids(self) -> list[list[int]]:
        _, _, reordered_gpu_ids = self.pg
        gpus_on_node = min(self.num_gpus_per_engine, self.args.num_gpus_per_node)
        bases = [
            int(reordered_gpu_ids[self.gpu_offset + local_index * gpus_on_node])
            for local_index in range(self.num_nodes)
        ]
        return [list(range(base, base + gpus_on_node)) for base in bases]

    @property
    def addr_infos(self) -> list[AddrInfo]:
        assert isinstance(self._state, StateAllocatedBase)
        assert self._state.addr_infos is not None, f"{self._state=}"
        return self._state.addr_infos

    @property
    def addr_info(self) -> AddrInfo:
        return self.addr_infos[0]

    @property
    def api_client(self) -> SGLangApiClient:
        return SGLangApiClient(server_url=self.addr_info.server_url)

    async def start_engines(self, port_allocator: PortAllocator) -> None:
        assert not ({"host", "port"} & set(self.sglang_overrides)), (
            f"sglang_overrides must not override host/port ({self.sglang_overrides=}): the rollout process derives "
            f"each engine's url from the addr allocator, so an override would make it talk to the wrong endpoint"
        )
        assert not self.is_allocated, "the caller starts only stopped cells"

        if self.args.rollout_external:
            raise NotImplementedError(
                "external rollout address allocation was removed and a new implementation is coming"
            )

        num_gpu_per_engine = int(self.spec.worker.scheduling.num_gpus_per_worker)

        actor_handles = [
            launch_sglang_ray_actor(
                args=self.args,
                pg=self.pg,
                spec=self.spec.worker,
                global_rank=self.rank_offset + local_index,
                gpu_index=self.gpu_offset + local_index * num_gpu_per_engine,
            )
            for local_index in range(self.num_nodes)
        ]

        self._mark_allocated_uninitialized(actor_handles)

        addressing = allocate_cell_ports(
            port_allocator=port_allocator,
            port_infos=self.spec.worker.port_infos,
            actors=actor_handles,
            node_ips=await probe_node_ips(actor_handles),
        )
        addr_and_ports = build_engine_addr_and_ports(addressing=addressing)

        self._mark_addressing(
            [
                AddrInfo(
                    server_url=build_server_url(host=entry["host"], port=entry["port"]),
                    bootstrap_port=entry.get("disaggregation_bootstrap_port"),
                )
                for entry in addr_and_ports
            ]
        )

        global_ranks = [self.rank_offset + local_index for local_index in range(self.num_nodes)]

        if env_report := self.args.env_report:
            await asyncio.gather(
                *[
                    actor._collect_env_report.remote(role="rollout", rank=rank, partial_env_report=env_report)
                    for rank, actor in zip(global_ranks, actor_handles, strict=True)
                ]
            )

        plans = [
            compute_engine_launch_plan(
                self.args,
                rank=rank,
                worker_type=self.worker_type,
                base_gpu_id=self.engine_gpu_ids[local_index][0],
                sglang_overrides=self.sglang_overrides,
                num_gpus_per_engine=self.num_gpus_per_engine,
                addr_and_ports=entry,
            )
            for local_index, (rank, entry) in enumerate(zip(global_ranks, addr_and_ports, strict=True))
        ]

        await asyncio.gather(
            *[actor.run.remote(cmd=plan.cmd, envs={}) for actor, plan in zip(actor_handles, plans, strict=True)]
        )

        await wait_server_healthy(
            server_url=self.addr_info.server_url,
            api_key=plans[0].api_key,
            is_process_alive=functools.partial(_engine_actor_is_alive, self.primary_actor_handle),
        )

    async def start(
        self, port_allocator: PortAllocator, router_api_client: SGLangRouterApiClient, recover: bool = False
    ) -> None:
        await self.start_engines(port_allocator)

        if recover and self.needs_offload:
            await self.api_client.release_memory_occupation()
            if self.update_weights or self.model_path:
                await self.api_client.resume_memory_occupation(tags=[GPU_MEMORY_TYPE_WEIGHTS])

        self._mark_alive()

        await self.register(router_api_client)

    async def stop(self, router_api_client: SGLangRouterApiClient) -> None:
        if self.is_allocated:
            try:
                await asyncio.wait_for(self.unregister(router_api_client), timeout=SHUTDOWN_TIMEOUT)
            except Exception as e:
                logger.warning(f"Unregistering cell {self.cell_id} from the router failed, tearing down anyway ({e})")

            for local_index, actor_handle in enumerate(self.actor_handles):
                logger.info(f"Cell {self.cell_id}: shutting down and killing engine at cell-local index {local_index}")
                try:
                    ray.get(actor_handle.shutdown.remote(), timeout=SHUTDOWN_TIMEOUT)
                except Exception as e:
                    logger.warning(
                        f"Cell {self.cell_id}: graceful shutdown of engine at cell-local index {local_index} "
                        f"failed, killing anyway ({e})"
                    )
                try:
                    ray.kill(actor_handle)
                    logger.info(f"Cell {self.cell_id}: killed engine at cell-local index {local_index}")
                except Exception as e:
                    logger.warning(f"Cell {self.cell_id}: fail to kill engine at cell-local index {local_index} ({e})")
        else:
            logger.info(f"Cell {self.cell_id} is already stopped")
        self._mark_stopped()

    def _mark_allocated_uninitialized(self, actor_handles: list[ray.actor.ActorHandle]) -> None:
        self._change_state(
            "mark_allocated_uninitialized", StateStopped, StateAllocatedUninitialized(actor_handles=actor_handles)
        )

    def _mark_addressing(self, addr_infos: list[AddrInfo]) -> None:
        self._change_state(
            "mark_addressing",
            StateAllocatedUninitialized,
            StateAllocatedUninitialized(actor_handles=self.actor_handles, addr_infos=addr_infos),
        )

    def _mark_alive(self) -> None:
        self._change_state(
            "mark_alive",
            StateAllocatedUninitialized,
            StateAllocatedAlive(actor_handles=self.actor_handles, addr_infos=self.addr_infos),
        )

    def _mark_stopped(self) -> None:
        self._change_state("mark_stopped", (StateStopped, StateAllocatedBase), StateStopped())

    # TODO: unify w/ trainer `change_state`
    def _change_state(
        self,
        debug_name: str,
        old_state_cls: type[CellState] | tuple[type[CellState], ...],
        new_state: CellState,
    ) -> None:
        logger.info(f"Cell {self.cell_id} {debug_name} start old={self._state}")
        assert isinstance(self._state, old_state_cls), f"{self._state=}"
        self._state = new_state
        logger.info(f"Cell {self.cell_id} {debug_name} end new={self._state}")

    async def offload(self, tags: list[str] | None):
        return await self.api_client.release_memory_occupation(tags=tags)

    async def onload(self, tags: list[str] | None):
        return await self.api_client.resume_memory_occupation(tags=tags)

    async def check_weights(self, action: str, allow_quant_error: bool, selector: str, skip_list: list[str] | None):
        return await self.api_client.check_weights(
            action=action, allow_quant_error=allow_quant_error, selector=selector, skip_list=skip_list
        )

    async def register(self, router_api_client: SGLangRouterApiClient) -> None:
        await router_api_client.add_worker(
            worker_url=self.addr_info.server_url,
            worker_type=self.worker_type,
            use_legacy_api=use_legacy_router_api(self.args),
            bootstrap_port=self.addr_info.bootstrap_port,
        )

    async def unregister(self, router_api_client: SGLangRouterApiClient) -> None:
        await router_api_client.remove_worker(
            worker_url=self.addr_info.server_url,
            use_legacy_api=use_legacy_router_api(self.args),
        )


def build_engine_addr_and_ports(*, addressing: CellAddressing) -> list[dict[str, Any]]:
    assert set(addressing.master_ports) == {ENGINE_DIST_INIT_PORT_NAME}, f"{addressing.master_ports=}"
    dist_init_addr = f"{format_v6_uri(addressing.node_ips[0])}:{addressing.master_ports[ENGINE_DIST_INIT_PORT_NAME]}"

    payloads: list[dict[str, Any]] = []
    for node_ip, ports in zip(addressing.node_ips, addressing.per_worker_ports, strict=True):
        payload = dict(
            host=format_v6_uri(node_ip),
            port=ports[ENGINE_SERVER_PORT_NAME],
            nccl_port=ports[ENGINE_NCCL_PORT_NAME],
            engine_info_bootstrap_port=ports[ENGINE_INFO_BOOTSTRAP_PORT_NAME],
            dist_init_addr=dist_init_addr,
        )
        if ENGINE_DISAGGREGATION_BOOTSTRAP_PORT_NAME in ports:
            payload["disaggregation_bootstrap_port"] = ports[ENGINE_DISAGGREGATION_BOOTSTRAP_PORT_NAME]
        payloads.append(payload)
    return payloads


def launch_sglang_ray_actor(
    *,
    args: Any,
    pg: Any,
    spec: InferenceWorkerSpec,
    global_rank: int,
    gpu_index: int,
) -> ray.actor.ActorHandle:
    pg, reordered_bundle_indices, _ = pg

    num_gpus = 0.2
    num_cpus = num_gpus

    env_vars = spec.env_var()
    # TODO: this is hacky. Use env var SGLANG_DG_CACHE_DIR_PER_PROCESS=1
    # to enable this isolation.
    env_vars["SGLANG_DG_CACHE_DIR"] = os.environ.get(
        "SGLANG_DG_CACHE_DIR", f"/tmp/sglang_deep_gemm/{spec.worker_type}_rank_{global_rank}"
    )

    return create_pg_worker_actor(
        worker_cls=CommandActor,
        pg_handle=pg,
        bundle_index=reordered_bundle_indices[gpu_index],
        env_vars=env_vars,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        ctor_kwargs={},
    )


def _engine_actor_is_alive(actor_handle: ray.actor.ActorHandle) -> bool:
    try:
        ray.get(actor_handle._get_node_ip.remote(), timeout=30)
        return True
    except Exception:
        return False
