import logging
from collections.abc import Awaitable, Callable

from miles.utils.function_registry import load_function
from miles.utils.workers.naming import parse_worker_name
from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_spec import RPC_PORT_NAME, NamedHostAndPorts

logger = logging.getLogger(__name__)


async def apply_cell_observation(
    *,
    cell_id: str,
    observed: CellInfo | None,
    actual: CellInfo | None,
    add: Callable[[str, CellInfo], Awaitable[None]],
    remove: Callable[[str], Awaitable[None]],
) -> None:
    if observed is not None and actual is None:
        await add(cell_id, observed)
    elif observed is None and actual is not None:
        await remove(cell_id)
    elif observed is not None and actual is not None and actual != observed:
        await remove(cell_id)
        await add(cell_id, observed)


def warn_static_membership(cell_id: str, *, provider: str) -> None:
    logger.warning(
        f"{provider} was asked to forget cell {cell_id}, but it serves the membership this run was launched with, "
        f"so nothing observes that cell anew and it stays in this run until the run ends"
    )


def build_rpc_handle(*, worker_class: type, addrs: NamedHostAndPorts, pool_id: str) -> BaseWorkerHandle:
    assert RPC_PORT_NAME in addrs, f"spec {pool_id} has no {RPC_PORT_NAME!r} port to be called through"
    return RpcWorkerHandle(worker_class, server_url=addrs[RPC_PORT_NAME].addr, require_stable_boot_uuid=True)


def attach_rpc_handle(info: WorkerInfo) -> WorkerInfo:
    if info.handle is not None or info.worker_class is None:
        return info

    pool_id, _cell_index, _worker_in_cell_index = parse_worker_name(info.name)
    handle = build_rpc_handle(worker_class=load_function(info.worker_class), addrs=info.self_addrs, pool_id=pool_id)
    return info.model_copy(update=dict(handle=handle))
