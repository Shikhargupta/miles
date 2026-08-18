import json
import logging
from dataclasses import dataclass

import requests

from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import (
    BOOT_UUID_READ,
    POD_KIND,
    WORKLOAD_KINDS,
    ClusterSnapshot,
    PodFact,
    WorkloadFact,
)

from miles.ray.specs.train import compute_trainer_controller_pool_id
from miles.utils.external_utils.command_utils.common import run_process
from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import RESTART_AT_ANNOTATION
from miles.utils.external_utils.command_utils.helm_backend.naming import RunNames
from miles.utils.workers.rpc.common.protocol import BOOT_UUID_HEADER, HEALTH_PATH
from miles.utils.workers.worker_provider.kubernetes.helm.env import INSTANCE_LABEL
from miles.utils.workers.worker_provider.kubernetes.helm.naming import static_worker_host
from miles.utils.workers.worker_spec import DEFAULT_RPC_PORT

logger = logging.getLogger(__name__)

KUBECTL_TIMEOUT_SECONDS: float = 60.0
BOOT_UUID_TIMEOUT_SECONDS: float = 10.0


def compute_trainer_rpc_url(*, release: str, namespace: str, trainer_id: str) -> str:
    host = RunNames.service_fqdn(
        name=static_worker_host(release, compute_trainer_controller_pool_id(trainer_id), 0), namespace=namespace
    )
    return f"http://{host}:{DEFAULT_RPC_PORT}{HEALTH_PATH}"


@dataclass(frozen=True)
class SnapshotAttempt:
    snapshot: ClusterSnapshot | None
    reads_attempted: tuple[str, ...]
    reads_failed: tuple[str, ...]


def read_cluster_snapshot(*, release: str, namespace: str, trainer_rpc_url: str) -> SnapshotAttempt:
    payload_of_kind = {
        kind: _read_objects(kind=kind, release=release, namespace=namespace) for kind in (POD_KIND, *WORKLOAD_KINDS)
    }
    boot_uuid = read_boot_uuid(trainer_rpc_url)

    attempted = (*payload_of_kind, BOOT_UUID_READ)
    failed = tuple(
        [kind for kind, payload in payload_of_kind.items() if payload is None]
        + ([BOOT_UUID_READ] if boot_uuid is None else [])
    )

    pods = payload_of_kind[POD_KIND]
    if pods is None:
        return SnapshotAttempt(snapshot=None, reads_attempted=attempted, reads_failed=failed)

    workloads = tuple(
        fact
        for kind in WORKLOAD_KINDS
        if (payload := payload_of_kind[kind]) is not None
        for fact in parse_workload_facts(payload, kind=kind)
    )
    snapshot = ClusterSnapshot(
        pods=parse_pod_facts(pods),
        workloads=tuple(sorted(workloads, key=lambda one: (one.kind, one.name))),
        trainer_boot_uuid=boot_uuid,
        reads_missing=failed,
    )
    return SnapshotAttempt(snapshot=snapshot, reads_attempted=attempted, reads_failed=failed)


def read_boot_uuid(url: str) -> str | None:
    try:
        response = requests.get(url, timeout=BOOT_UUID_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.headers.get(BOOT_UUID_HEADER)
    except Exception:
        logger.warning(f"Failed to read the rpc server boot uuid from {url}", exc_info=True)
        return None


def _read_objects(*, kind: str, release: str, namespace: str) -> dict | None:
    try:
        result = run_process(
            [
                "kubectl",
                "get",
                kind,
                "--namespace",
                namespace,
                "--selector",
                f"{INSTANCE_LABEL}={release}",
                "--output",
                "json",
            ],
            capture_output=True,
            check=True,
            timeout=KUBECTL_TIMEOUT_SECONDS,
        )
        return json.loads(result.stdout)
    except Exception:
        logger.warning(f"Failed to list the {kind} of {release} in {namespace}", exc_info=True)
        return None


def parse_pod_facts(payload: dict) -> tuple[PodFact, ...]:
    facts = [
        PodFact(
            name=item["metadata"]["name"],
            uid=item["metadata"]["uid"],
            restart_count=sum(
                int(one.get("restartCount", 0)) for one in item.get("status", {}).get("containerStatuses", [])
            ),
        )
        for item in payload["items"]
    ]
    return tuple(sorted(facts, key=lambda one: one.name))


def parse_workload_facts(payload: dict, *, kind: str) -> tuple[WorkloadFact, ...]:
    facts = [
        WorkloadFact(
            kind=kind,
            name=item["metadata"]["name"],
            generation=int(item["metadata"]["generation"]),
            restart_at=_read_restart_at(item),
        )
        for item in payload["items"]
    ]
    return tuple(sorted(facts, key=lambda one: one.name))


def _read_restart_at(item: dict) -> str | None:
    spec = item.get("spec", {})
    leader_worker_template = spec.get("leaderWorkerTemplate", {})
    templates = [
        spec.get("template"),
        leader_worker_template.get("leaderTemplate"),
        leader_worker_template.get("workerTemplate"),
    ]
    stamps = {
        stamp
        for template in templates
        if template is not None
        if (stamp := template.get("metadata", {}).get("annotations", {}).get(RESTART_AT_ANNOTATION)) is not None
    }
    assert len(stamps) <= 1, (
        f"{item['metadata']['name']} carries the restart stamps {sorted(stamps)} on the templates of one object, "
        f"and a hot restart writes one stamp per object it replaces"
    )
    return next(iter(stamps), None)
