from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

from miles.backends.sglang_utils.sglang_config import resolve_sglang_config
from miles.ray.specs.inference import (
    INFERENCE_CONTROLLER_POOL_ID,
    compute_router_pool_id,
    compute_router_worker_name,
    inference_controller_worker_name,
)
from miles.ray.specs.static_addrs import (
    INFERENCE_CONTROLLER_ADDRS_FLAG,
    INFERENCE_ROUTER_ADDRS_FLAG,
    TRAINER_CONTROLLER_ADDRS_FLAG,
)
from miles.ray.specs.train import compute_deployed_trainer_instances, trainer_controller_worker_name
from miles.ray.specs.trainer_identity import compute_trainer_controller_pool_id
from miles.utils.workers.types import DeployComponent
from miles.utils.workers.worker_spec import RPC_PORT_NAME, HostAndPort

ROUTER_PRIMARY_PORT_NAME = "primary"


class AddressedWorker(NamedTuple):
    flag: str
    key: str | None
    pool_id: str
    worker_name: str
    port_name: str


def compute_addressed_workers(args, *, component: DeployComponent) -> list[AddressedWorker]:
    if component is DeployComponent.INFERENCE:
        return [
            AddressedWorker(
                flag=INFERENCE_CONTROLLER_ADDRS_FLAG,
                key=None,
                pool_id=INFERENCE_CONTROLLER_POOL_ID,
                worker_name=inference_controller_worker_name(),
                port_name=RPC_PORT_NAME,
            ),
            *[
                AddressedWorker(
                    flag=INFERENCE_ROUTER_ADDRS_FLAG,
                    key=model_cfg.name,
                    pool_id=compute_router_pool_id(model_idx),
                    worker_name=compute_router_worker_name(model_idx),
                    port_name=ROUTER_PRIMARY_PORT_NAME,
                )
                for model_idx, model_cfg in enumerate(resolve_sglang_config(args).models)
            ],
        ]

    assert component is DeployComponent.TRAINER, (
        f"only the {DeployComponent.TRAINER.value} and {DeployComponent.INFERENCE.value} deployments are reached "
        f"by address, not {component.value}"
    )
    return [
        AddressedWorker(
            flag=TRAINER_CONTROLLER_ADDRS_FLAG,
            key=role,
            pool_id=compute_trainer_controller_pool_id(role),
            worker_name=trainer_controller_worker_name(role),
            port_name=RPC_PORT_NAME,
        )
        for role in [instance.role for instance in compute_deployed_trainer_instances(args)]
    ]


def format_addressed_workers(entries: Sequence[tuple[AddressedWorker, HostAndPort]]) -> str:
    values_by_flag: dict[str, list[str]] = {}
    for worker, addr in entries:
        value = f"{addr.host}:{addr.port}" if worker.key is None else f"{worker.key}={addr.host}:{addr.port}"
        values_by_flag.setdefault(worker.flag, []).append(value)
    return " ".join(f"{flag} {' '.join(values)}" for flag, values in values_by_flag.items())
