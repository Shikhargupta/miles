from dataclasses import dataclass

from miles.utils.workers.ray_worker_manager.state import ActorState
from miles.utils.workers.worker_spec import BaseWorkerSpec


@dataclass(frozen=True)
class WorkerAddressing:
    addr_port_kwargs: dict[str, str | int]
    url: str | None


def compute_worker_addressings(*, spec: BaseWorkerSpec, workers: list[ActorState]) -> dict[str, WorkerAddressing]:
    master = workers[0]

    addressings: dict[str, WorkerAddressing] = {}
    for worker in workers:
        addr_port_kwargs: dict[str, str | int] = {}
        url: str | None = None
        for port_info in spec.port_infos:
            owner = master if port_info.mode == "master" else worker
            port = owner.owned_ports[port_info.name]
            addr_port_kwargs[f"{port_info.name}_addr"] = owner.node_ip
            addr_port_kwargs[f"{port_info.name}_port"] = port
            if port_info.url_scheme is not None:
                url = f"{port_info.url_scheme}://{owner.node_ip}:{port}"
        addressings[worker.name] = WorkerAddressing(addr_port_kwargs=addr_port_kwargs, url=url)

    return addressings
