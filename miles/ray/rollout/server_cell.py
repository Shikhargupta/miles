import dataclasses
import logging
from dataclasses import dataclass
from typing import Any

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
)
from miles.ray.specs.inference import InferenceCellSpec
from miles.utils.workers.worker_provider.base import CellInfo, CellMember

logger = logging.getLogger(__name__)


@dataclass
class ServerCell:
    args: Any
    spec: InferenceCellSpec
    update_weights: bool = True
    attached_members: list[CellMember] = dataclasses.field(default_factory=list)
    _state: CellState | None = None

    @property
    def cell_id(self) -> str:
        return self.spec.cell_id

    @property
    def needs_offload(self) -> bool:
        return self.spec.worker.needs_offload

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
        gpus_on_node = min(self.spec.worker.num_gpus_per_engine, self.args.num_gpus_per_node)
        return [
            list(range(member.placement.base_gpu_id, member.placement.base_gpu_id + gpus_on_node))
            for member in self.attached_members
        ]

    @property
    def addr_infos(self) -> list[AddrInfo]:
        assert isinstance(self._state, StateAllocatedBase)
        return self._state.addr_infos

    @property
    def addr_info(self) -> AddrInfo:
        return self.addr_infos[0]

    @property
    def api_client(self) -> SGLangApiClient:
        return SGLangApiClient(server_url=self.addr_info.server_url)

    @classmethod
    def attach(cls, *, args: Any, spec: InferenceCellSpec, update_weights: bool, cell_info: CellInfo) -> "ServerCell":
        """A cell is born attached: it exists exactly as long as its observed workers do."""
        assert cell_info.cell_id == spec.cell_id, f"{cell_info.cell_id=} does not name {spec.cell_id=}"

        addr_infos = [
            AddrInfo(
                server_url=build_server_url(host=member.payload["host"], port=member.payload["port"]),
                bootstrap_port=member.payload.get("disaggregation_bootstrap_port"),
            )
            for member in cell_info.members
        ]
        return cls(
            args=args,
            spec=spec,
            update_weights=update_weights,
            attached_members=list(cell_info.members),
            _state=StateAllocatedUninitialized(
                actor_handles=[member.handle for member in cell_info.members], addr_infos=addr_infos
            ),
        )

    async def release_offloaded_memory(self) -> None:
        """Give back the GPU memory a freshly attached engine holds."""
        await self.api_client.release_memory_occupation()
        if self.update_weights or self.spec.worker.model_path:
            await self.api_client.resume_memory_occupation(tags=[GPU_MEMORY_TYPE_WEIGHTS])

    def mark_alive(self) -> None:
        self._change_state(
            "mark_alive",
            StateAllocatedUninitialized,
            StateAllocatedAlive(actor_handles=self.actor_handles, addr_infos=self.addr_infos),
        )

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
            worker_type=self.spec.worker.worker_type,
            use_legacy_api=use_legacy_router_api(self.args),
            bootstrap_port=self.addr_info.bootstrap_port,
        )

    async def unregister(self, router_api_client: SGLangRouterApiClient) -> None:
        await router_api_client.remove_worker(
            worker_url=self.addr_info.server_url,
            use_legacy_api=use_legacy_router_api(self.args),
        )
