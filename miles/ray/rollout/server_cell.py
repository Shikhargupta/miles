import asyncio
import dataclasses
import logging
import time
from collections.abc import Awaitable, Callable, Collection
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError
from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient, probe_server_healthy
from miles.backends.sglang_utils.sglang_engine import build_server_url
from miles.backends.sglang_utils.sglang_router_api_client import SGLangRouterApiClient, use_legacy_router_api
from miles.ray.rollout.cell_state import (
    CellAddrInfo,
    CellState,
    StateDisposed,
    StateInitializing,
    StatePendingWeights,
    StateServing,
    StateUninitialized,
)
from miles.ray.rollout.engine_env_reporter import EngineEnvReporter
from miles.utils.ft_utils.api_server.models import CellCondition, CellStatus, TriState
from miles.utils.ft_utils.health_checker import (
    ActiveAndEpoch,
    BaseHealthChecker,
    NoopHealthChecker,
    SimpleHealthChecker,
    SimpleHealthCheckerConfig,
)
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.launch_gate import (
    GATE_PORT_NAME,
    GATE_TIMEOUT_META_KEY,
    LAUNCH_GATE_TIMEOUT_SECONDS,
    activate_launch_gate,
)
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo

logger = logging.getLogger(__name__)

SHUTDOWN_TIMEOUT = 30
CELL_CALL_TIMEOUT_SECONDS = 60.0
SERVING_DEADLINE_SECONDS = 1800.0
STUCK_WARNING_INTERVAL_SECONDS = 60.0


class ServerCellMetadata(FrozenStrictBaseModel):
    model_id: str
    worker_type: Literal["regular", "prefill", "decode"]
    cell_id: str
    num_gpus_per_engine: int
    gpu_offset: int
    sglang_api_key: str | None
    worker_name: str
    needs_offload: bool
    update_weights: bool
    workers_hash: str
    launch_gate_timeout_seconds: float = LAUNCH_GATE_TIMEOUT_SECONDS


@dataclass
class ServerCell:
    args: Any
    meta: ServerCellMetadata
    router_api_client: SGLangRouterApiClient
    provider: BaseWorkerProvider
    global_health_checker_activeness: Callable[[], ActiveAndEpoch] = lambda: ActiveAndEpoch(active=True, epoch=0)
    observed_info: CellInfo | None = None
    clock: Callable[[], float] = time.monotonic
    serving_deadline_seconds: float = SERVING_DEADLINE_SECONDS
    _health_checker: BaseHealthChecker = dataclasses.field(init=False)
    _env_reporter: EngineEnvReporter = dataclasses.field(init=False)
    _state: CellState = dataclasses.field(default_factory=StateUninitialized)
    _state_entered_at: float = dataclasses.field(init=False, default=0.0)
    _stuck_warned_at: float | None = dataclasses.field(init=False, default=None)
    _reached_router: bool = dataclasses.field(init=False, default=False)
    _router_lock: asyncio.Lock = dataclasses.field(init=False, default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        self._state_entered_at = self.clock()
        self._env_reporter = EngineEnvReporter(interval_seconds=self.args.env_report_interval_seconds)
        self._health_checker = create_rollout_cell_health_checker(
            args=self.args,
            name=f"rollout-cell-{self.meta.cell_id}",
            get_api_client=lambda: self.api_client,
            get_activeness=self._get_health_checker_active_and_epoch,
        )
        self._health_checker.start()

    def _get_health_checker_active_and_epoch(self) -> ActiveAndEpoch:
        controller_active_and_epoch = self.global_health_checker_activeness()
        cell_active = isinstance(self._state, (StatePendingWeights, StateServing))
        return ActiveAndEpoch(
            active=cell_active and controller_active_and_epoch.active, epoch=controller_active_and_epoch.epoch
        )

    def __del__(self) -> None:
        assert isinstance(self._state, StateDisposed), (
            f"ServerCell {self.meta.cell_id} was garbage collected without dispose() ({self._state=}); "
            "every cell must be disposed so its health checker task is stopped"
        )

    def cell_status(self) -> CellStatus:
        match self._state:
            case StateUninitialized() | StateInitializing():
                return compute_pending_rollout_cell_status()

            case StatePendingWeights() | StateServing():
                return CellStatus(
                    phase="Running",
                    conditions=[
                        CellCondition.allocated(TriState.TRUE),
                        CellCondition.from_health_checker_status(self._health_checker.status),
                        CellCondition.serving(TriState.TRUE if self.is_serving else TriState.FALSE),
                    ],
                )

            case StateDisposed():
                return CellStatus(
                    phase="Suspended",
                    conditions=[CellCondition.allocated(TriState.FALSE)],
                )

            case _:
                raise NotImplementedError(f"Unknown state: {self._state}")

    @property
    def is_uninitialized(self) -> bool:
        return isinstance(self._state, StateUninitialized)

    @property
    def is_initializing(self) -> bool:
        return isinstance(self._state, StateInitializing)

    @property
    def is_pending_weights_or_serving(self) -> bool:
        return isinstance(self._state, (StatePendingWeights, StateServing))

    @property
    def is_pending_weights(self) -> bool:
        return isinstance(self._state, StatePendingWeights)

    @property
    def is_serving(self) -> bool:
        return isinstance(self._state, StateServing)

    @property
    def is_unreachable(self) -> bool:
        return self.unreachable_reason is not None

    @property
    def seconds_in_state(self) -> float:
        return self.clock() - self._state_entered_at

    @property
    def unreachable_reason(self) -> str | None:
        if self.is_pending_weights_or_serving:
            return "it failed its health checks" if self._health_checker.status is TriState.FALSE else None
        if not self.is_initializing or self.seconds_in_state < self.serving_deadline_seconds:
            return None
        return (
            f"its engine did not start serving within {self.serving_deadline_seconds:.0f}s of answering its launch "
            f"gate"
        )

    @property
    def addr_info(self) -> CellAddrInfo:
        assert isinstance(self._state, (StateInitializing, StatePendingWeights, StateServing))
        return self._state.addr_info

    @property
    def server_url(self) -> str:
        return self.addr_info.server_url

    @property
    def api_client(self) -> SGLangApiClient:
        return SGLangApiClient(server_url=self.server_url, api_key=self.meta.sglang_api_key)

    async def init(self) -> None:
        addr_info = await self._compute_addr_info()
        if (gate_url := addr_info.gate_url) is not None:
            await activate_launch_gate(gate_url=gate_url, timeout=self.meta.launch_gate_timeout_seconds)
        self._change_state("init", StateUninitialized, StateInitializing(addr_info=addr_info))

    async def tick(self) -> None:
        if isinstance(self._state, StateDisposed):
            return
        if isinstance(self._state, StateInitializing):
            await self._tick_when_initializing()
        await self._report_env_if_due()

    def _warn_if_stuck(self) -> None:
        elapsed = self.seconds_in_state
        if self._stuck_warned_at is not None and elapsed - self._stuck_warned_at < STUCK_WARNING_INTERVAL_SECONDS:
            return
        self._stuck_warned_at = elapsed
        logger.warning(
            f"Cell {self.meta.cell_id} answered its launch gate {elapsed:.0f}s ago but its engine still does not "
            f"answer /health_generate at {self._state.addr_info.server_url}; after "
            f"{self.serving_deadline_seconds:.0f}s it is handed back to its provider so this run stops waiting for it"
        )

    async def _report_env_if_due(self) -> None:
        if not self.is_pending_weights_or_serving:
            return
        await self._env_reporter.report_if_due(
            cell_id=self.meta.cell_id, server_url=self.server_url, api_client=self.api_client
        )

    async def _tick_when_initializing(self) -> None:
        addr_info = self._state.addr_info
        if not await probe_server_healthy(server_url=addr_info.server_url, api_key=self.meta.sglang_api_key):
            self._warn_if_stuck()
            return
        if not self._is_still_initializing():
            return

        if self.args.check_weight_update_equal and self.meta.update_weights:
            await self.check_weights(action="snapshot", allow_quant_error=False, selector="all", skip_list=None)

        if self.meta.needs_offload:
            api_client = SGLangApiClient(server_url=addr_info.server_url)
            await _with_timeout(api_client.release_memory_occupation(), what="releasing the memory of")
            await _with_timeout(
                api_client.resume_memory_occupation(tags=[GPU_MEMORY_TYPE_WEIGHTS]), what="resuming the memory of"
            )

        if not self._is_still_initializing():
            return

        serve_without_weight_update: bool = not self.meta.update_weights or self.args.debug_rollout_only
        if serve_without_weight_update:
            await self._register_with_router(addr_info=addr_info)
            if not self._is_still_initializing():
                await self._unregister_from_router(server_url=addr_info.server_url)
                return

        self._change_state("mark_pending_weights", StateInitializing, StatePendingWeights(addr_info=addr_info))

        if serve_without_weight_update:
            self._mark_serving()
        elif self.args.check_weight_update_equal:
            await self.check_weights(
                action="reset_tensors",
                allow_quant_error=False,
                selector="all",
                skip_list=self.args.check_weight_update_skip_list,
            )

    async def mark_weights_ready(self) -> None:
        assert isinstance(self._state, StatePendingWeights), f"{self._state=}"
        await self._register_with_router(addr_info=self._state.addr_info)
        self._mark_serving()

    def _is_still_initializing(self) -> bool:
        if isinstance(self._state, StateInitializing):
            return True
        logger.warning(
            f"Cell {self.meta.cell_id} left {StateInitializing.__name__} while it was being probed, so this probe "
            f"is dropped and whatever moved it decides what this run holds"
        )
        return False

    async def _register_with_router(self, addr_info: CellAddrInfo) -> None:
        async with self._router_lock:
            self._reached_router = True
            await _with_timeout(
                self.router_api_client.add_worker(
                    worker_url=addr_info.server_url,
                    worker_type=self.meta.worker_type,
                    use_legacy_api=use_legacy_router_api(self.args),
                    bootstrap_port=addr_info.bootstrap_port,
                ),
                what="registering with the router",
            )

    async def dispose(self) -> None:
        self._health_checker.stop()

        if self._reached_router:
            await self._unregister_from_router(server_url=self.server_url)

        self._change_state(
            "dispose",
            (StateUninitialized, StateInitializing, StatePendingWeights, StateServing, StateDisposed),
            StateDisposed(),
        )

    async def _unregister_from_router(self, *, server_url: str) -> None:
        async with self._router_lock:
            if not self._reached_router:
                return
            self._reached_router = False
            try:
                await asyncio.wait_for(
                    self.router_api_client.remove_worker(
                        worker_url=server_url,
                        use_legacy_api=use_legacy_router_api(self.args),
                    ),
                    timeout=SHUTDOWN_TIMEOUT,
                )
            except Exception as e:
                logger.warning(
                    f"Unregistering cell {self.meta.cell_id} from the router failed, tearing down anyway ({e})"
                )

    async def _compute_addr_info(self) -> CellAddrInfo:
        master_addrs = await self.provider.get_addrs(worker_name=self.meta.worker_name)
        primary = master_addrs["primary"]
        gate = master_addrs.get(GATE_PORT_NAME)
        return CellAddrInfo(
            server_url=build_server_url(host=primary.host, port=primary.port),
            bootstrap_port=x.port if (x := master_addrs.get("disaggregation_bootstrap")) else None,
            gate_url=build_server_url(host=gate.host, port=gate.port) if gate else None,
        )

    def _mark_serving(self) -> None:
        self._change_state("mark_serving", StatePendingWeights, StateServing(addr_info=self.addr_info))

    # TODO: unify w/ trainer `change_state`
    def _change_state(
        self,
        debug_name: str,
        old_state_cls: type[CellState] | tuple[type[CellState], ...],
        new_state: CellState,
    ) -> None:
        logger.info(f"Cell {self.meta.cell_id} {debug_name} start old={self._state}")
        assert isinstance(self._state, old_state_cls), f"{self._state=}"
        self._state = new_state
        self._state_entered_at = self.clock()
        self._stuck_warned_at = None
        logger.info(f"Cell {self.meta.cell_id} {debug_name} end new={self._state}")

    async def probe_and_mark_dead(self) -> None:
        if not self.is_allocated:
            return
        try:
            await asyncio.wait_for(self.api_client.get_weight_version(), timeout=60)
        except Exception as e:
            logger.warning(f"Cell unreachable ({e!r}); marking stopped for recovery")
            self._mark_stopped()

    async def offload(self, tags: list[str] | None):
        return await _with_timeout(self.api_client.release_memory_occupation(tags=tags), what="offloading")

    async def onload(self, tags: list[str] | None):
        return await _with_timeout(self.api_client.resume_memory_occupation(tags=tags), what="onloading")

    async def check_weights(self, action: str, allow_quant_error: bool, selector: str, skip_list: list[str] | None):
        return await _with_timeout(
            self.api_client.check_weights(
                action=action, allow_quant_error=allow_quant_error, selector=selector, skip_list=skip_list
            ),
            what="checking the weights of",
        )


async def _with_timeout(call: Awaitable[Any], *, what: str, timeout: float | None = None) -> Any:
    if timeout is None:
        timeout = CELL_CALL_TIMEOUT_SECONDS
    try:
        return await asyncio.wait_for(call, timeout=timeout)
    except TimeoutError as e:
        raise TimeoutError(f"{what} an engine of this run took longer than {timeout:.0f}s") from e


# TODO may move and generalize later
def compute_server_cell_meta_from_info(info: CellInfo) -> ServerCellMetadata:
    return ServerCellMetadata(
        model_id=info.meta["model_id"],
        worker_type=info.meta["worker_type"],
        cell_id=info.cell_id,
        num_gpus_per_engine=info.meta["num_gpus_per_engine"],
        gpu_offset=info.meta["gpu_offset"],
        sglang_api_key=info.meta["sglang_api_key"],
        worker_name=info.worker_names[0],
        needs_offload=info.meta["needs_offload"],
        update_weights=info.meta["update_weights"],
        workers_hash=info.workers_hash,
        launch_gate_timeout_seconds=info.meta.get(GATE_TIMEOUT_META_KEY, LAUNCH_GATE_TIMEOUT_SECONDS),
    )


def compute_server_cell_refusal_reason(info: CellInfo, *, model_ids: Collection[str]) -> str | None:
    try:
        cell_meta = compute_server_cell_meta_from_info(info)
    except KeyError as e:
        return (
            f"its metadata carries no {e.args[0]!r}, and every cell this run serves is built from that field; what "
            f"it did carry is {sorted(info.meta)}, so the two deployments run different versions of miles"
        )
    except ValidationError as e:
        return (
            f"its metadata does not describe an engine this run can serve ({_summarize_validation_error(e)}), so "
            f"the two deployments run different versions of miles"
        )
    if cell_meta.model_id not in model_ids:
        return (
            f"it serves model {cell_meta.model_id!r} and this run serves {sorted(model_ids)}, so no request of this "
            f"run would ever reach it"
        )
    return None


def _summarize_validation_error(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in one['loc'])}: {one['msg']}" for one in error.errors(include_url=False)
    )


def compute_nodes_per_engine(*, num_gpus_per_engine: int, num_gpus_per_node: int) -> int:
    return max(1, num_gpus_per_engine // num_gpus_per_node)


def create_rollout_cell_health_checker(
    *,
    args: Any,
    name: str,
    get_api_client: Callable[[], SGLangApiClient],
    get_activeness: Callable[[], ActiveAndEpoch],
) -> BaseHealthChecker:
    if "rollout" not in args.ft_components:
        return NoopHealthChecker()

    config = SimpleHealthCheckerConfig.from_args(args, prefix="rollout_health_check")

    async def _check() -> None:
        await get_api_client().health_generate(timeout=config.timeout)

    return SimpleHealthChecker(name=name, check_fn=_check, get_activeness=get_activeness, config=config)


def compute_pending_rollout_cell_status() -> CellStatus:
    return CellStatus(
        phase="Pending",
        conditions=[CellCondition.allocated(TriState.TRUE)],
    )
