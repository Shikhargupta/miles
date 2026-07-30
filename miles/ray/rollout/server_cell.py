import dataclasses
import logging
from dataclasses import dataclass
from typing import Any, Literal

import ray
from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.sglang_utils.sglang_engine import build_server_url
from miles.backends.sglang_utils.sglang_router_api_client import SGLangRouterApiClient, use_legacy_router_api
from miles.ray.rollout.cell_state import (
    AddrInfo,
    CellState,
    StateAllocatedAlive,
    StateAllocatedBase,
    StateAllocatedUninitialized,
    StateStopped,
)
from miles.ray.specs.inference import InferenceCellSpec
from miles.utils.workers.worker_provider.base import CellInfo, CellMember

logger = logging.getLogger(__name__)


@dataclass
class ServerCell:
    args: Any
    spec: InferenceCellSpec
    update_weights: bool = True
    attached_members: list[CellMember] | None = None
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
        assert self.attached_members is not None, f"cell {self.cell_id} has no workers to report gpus for"
        gpus_on_node = min(self.num_gpus_per_engine, self.args.num_gpus_per_node)
        return [
            list(range(member.placement.base_gpu_id, member.placement.base_gpu_id + gpus_on_node))
            for member in self.attached_members
        ]

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

    def attach(self, cell_info: CellInfo) -> None:
        """Adopt the workers the infrastructure layer reports for this cell."""
        assert not self.is_allocated, "a cell must be detached before it attaches to new workers"

        self._mark_allocated_uninitialized([member.handle for member in cell_info.members])
        self._mark_addressing(
            [
                AddrInfo(
                    server_url=build_server_url(host=member.payload["host"], port=member.payload["port"]),
                    bootstrap_port=member.payload.get("disaggregation_bootstrap_port"),
                )
                for member in cell_info.members
            ]
        )
        self.attached_members = list(cell_info.members)

    async def release_offloaded_memory(self) -> None:
        """Give back the GPU memory a freshly attached engine holds."""
        await self.api_client.release_memory_occupation()
        if self.update_weights or self.model_path:
            await self.api_client.resume_memory_occupation(tags=[GPU_MEMORY_TYPE_WEIGHTS])

    def mark_alive(self) -> None:
        self._mark_alive()

    def detach(self) -> None:
        self._mark_stopped()
        self.attached_members = None

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
