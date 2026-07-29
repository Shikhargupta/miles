import asyncio
import dataclasses
import logging
from dataclasses import dataclass
from typing import Any, Literal

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
from miles.utils.workers.naming import compute_worker_name
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_provider.base import BaseWorkerProvider

logger = logging.getLogger(__name__)


@dataclass
class ServerCell:
    args: Any
    worker_type: Literal["regular", "prefill", "decode"]
    cell_id: str
    num_nodes: int = 1
    num_gpus_per_engine: int = 1
    gpu_offset: int = 0
    needs_offload: bool = False
    model_path: str | None = None
    update_weights: bool = True
    spec_name: str = ""
    cell_index: int = 0
    provider: BaseWorkerProvider | None = None
    observed_members_hash: str | None = None
    failed_attach_members_hash: str | None = None
    _state: CellState = dataclasses.field(default_factory=StateStopped)

    @property
    def is_allocated(self) -> bool:
        return isinstance(self._state, StateAllocatedBase)

    @property
    def is_alive(self) -> bool:
        return isinstance(self._state, StateAllocatedAlive)

    @property
    def worker_handles(self) -> list[BaseWorkerHandle]:
        assert isinstance(self._state, StateAllocatedBase)
        return self._state.worker_handles

    @property
    def primary_worker_handle(self) -> BaseWorkerHandle:
        return self.worker_handles[0]

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

    async def start(self, router_api_client: SGLangRouterApiClient) -> None:
        await self.start_engines()
        await self.promote_to_alive(router_api_client)

    async def start_engines(self) -> None:
        assert not self.is_allocated, "the caller starts only stopped cells"
        assert self.provider is not None, f"cell {self.cell_id} was built without a worker provider"
        self.observed_members_hash = None

        worker_names = [
            compute_worker_name(spec_name=self.spec_name, cell_index=self.cell_index, worker_index=worker_index)
            for worker_index in range(self.num_nodes)
        ]
        handles = [await self.provider.get_handle(worker_name) for worker_name in worker_names]
        self._mark_allocated_uninitialized(handles)

        addr_and_ports = await asyncio.gather(*[handle.get_addr_and_ports() for handle in handles])
        self._mark_addressing(
            [
                AddrInfo(
                    server_url=build_server_url(host=addr_ports["server_addr"], port=addr_ports["server_port"]),
                    bootstrap_port=addr_ports.get("disaggregation_bootstrap_port"),
                )
                for addr_ports in addr_and_ports
            ]
        )

        await asyncio.gather(*[handle.init() for handle in handles])

    async def attach_unsynced(self) -> None:
        try:
            await self.start_engines()

            if self.needs_offload:
                await self.api_client.release_memory_occupation()
                if self.update_weights or self.model_path:
                    await self.api_client.resume_memory_occupation(tags=[GPU_MEMORY_TYPE_WEIGHTS])
        except Exception:
            if self.is_allocated:
                self._mark_stopped()
            raise

    async def promote_to_alive(self, router_api_client: SGLangRouterApiClient) -> None:
        await self.register(router_api_client)
        self._mark_alive()

    def _mark_allocated_uninitialized(self, worker_handles: list[BaseWorkerHandle]) -> None:
        self._change_state(
            "mark_allocated_uninitialized", StateStopped, StateAllocatedUninitialized(worker_handles=worker_handles)
        )

    def _mark_addressing(self, addr_infos: list[AddrInfo]) -> None:
        self._change_state(
            "mark_addressing",
            StateAllocatedUninitialized,
            StateAllocatedUninitialized(worker_handles=self.worker_handles, addr_infos=addr_infos),
        )

    def _mark_alive(self) -> None:
        self._change_state(
            "mark_alive",
            StateAllocatedUninitialized,
            StateAllocatedAlive(worker_handles=self.worker_handles, addr_infos=self.addr_infos),
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
