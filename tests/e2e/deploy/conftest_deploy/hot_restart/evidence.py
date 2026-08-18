import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

from miles.ray.specs.rollout import ROLLOUT_EXECUTOR_POOL_ID
from miles.utils.external_utils.command_utils.helm_backend.naming import ORCHESTRATOR_COMPONENT
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.worker_provider.kubernetes.helm.naming import component_name

logger = logging.getLogger(__name__)

POD_KIND: str = "pods"
STATEFUL_SET_KIND: str = "statefulsets"
LEADER_WORKER_SET_KIND: str = "leaderworkersets.leaderworkerset.x-k8s.io"
BOOT_UUID_READ: str = "trainer-boot-uuid"
WORKLOAD_KINDS: tuple[str, ...] = (STATEFUL_SET_KIND, LEADER_WORKER_SET_KIND)


class PodFact(FrozenStrictBaseModel):
    name: str
    uid: str
    restart_count: int


class WorkloadFact(FrozenStrictBaseModel):
    kind: str
    name: str
    generation: int
    restart_at: str | None


class ClusterSnapshot(FrozenStrictBaseModel):
    pods: tuple[PodFact, ...]
    workloads: tuple[WorkloadFact, ...]
    trainer_boot_uuid: str | None
    reads_missing: tuple[str, ...] = ()

    @property
    def workload_names(self) -> tuple[str, ...]:
        return tuple(one.name for one in self.workloads)

    @property
    def describes_the_whole_release(self) -> bool:
        return not ({POD_KIND, *WORKLOAD_KINDS} & set(self.reads_missing))

    @property
    def describes_a_release_that_is_gone(self) -> bool:
        return not self.pods or not self.workloads


class ObservationCounts(FrozenStrictBaseModel):
    attempts_of_read: dict[str, int] = {}
    failures_of_read: dict[str, int] = {}

    def success_ratio_of(self, read: str) -> float:
        attempts = self.attempts_of_read.get(read, 0)
        if attempts == 0:
            return 0.0
        return 1.0 - self.failures_of_read.get(read, 0) / attempts


class HotRestartRecord(FrozenStrictBaseModel):
    index: int
    saved_iteration_at_trigger: int
    finished_rollout_id_at_trigger: int


class HotRestartEvidence(FrozenStrictBaseModel):
    records: tuple[HotRestartRecord, ...]
    snapshots: tuple[ClusterSnapshot, ...]
    release: str
    observations: ObservationCounts = ObservationCounts()

    def write(self, *, dump_dir: str) -> None:
        path = evidence_path(dump_dir=dump_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
        logger.info(f"Wrote what {len(self.records)} hot restart(s) left behind to {path}")

    @classmethod
    def load(cls, *, dump_dir: str) -> "HotRestartEvidence":
        path = evidence_path(dump_dir=dump_dir)
        assert path.is_file(), (
            f"{path} does not exist, so nothing recorded which steps this run redid and which pods outlived the "
            f"orchestration script; run the target side before comparing"
        )
        return cls.model_validate_json(path.read_text())


def evidence_path(*, dump_dir: str) -> Path:
    return Path(dump_dir) / "hot_restart" / "evidence.json"


def compute_hot_restart_workloads(release: str) -> frozenset[str]:
    return frozenset(
        component_name(release, component) for component in (ORCHESTRATOR_COMPONENT, ROLLOUT_EXECUTOR_POOL_ID)
    )


def compute_workload_of_pod(pod_name: str, *, workloads: Iterable[str]) -> str | None:
    candidates = [one for one in workloads if pod_name.startswith(f"{one}-")]
    return max(candidates, key=len) if candidates else None


def compute_unattributed_pod_names(snapshots: Sequence[ClusterSnapshot]) -> set[str]:
    return {
        pod.name
        for snapshot in snapshots
        if snapshot.describes_the_whole_release
        for pod in snapshot.pods
        if compute_workload_of_pod(pod.name, workloads=snapshot.workload_names) is None
    }


def compute_workloads_whose_pods_were_replaced(snapshots: Sequence[ClusterSnapshot]) -> dict[str, list[str]]:
    workloads = sorted({name for snapshot in snapshots for name in snapshot.workload_names})
    uids_of_pod: dict[str, set[str]] = {}
    restart_counts_of_pod: dict[str, set[int]] = {}
    pod_names_of_workload: dict[str, set[frozenset[str]]] = {}

    for snapshot in snapshots:
        seen_of_workload: dict[str, set[str]] = {one: set() for one in workloads}
        for pod in snapshot.pods:
            if (workload := compute_workload_of_pod(pod.name, workloads=workloads)) is None:
                continue
            seen_of_workload[workload].add(pod.name)
            uids_of_pod.setdefault(pod.name, set()).add(pod.uid)
            restart_counts_of_pod.setdefault(pod.name, set()).add(pod.restart_count)
        for workload, seen in seen_of_workload.items():
            pod_names_of_workload.setdefault(workload, set()).add(frozenset(seen))

    reasons_of_workload: dict[str, list[str]] = {}
    for pod_name in sorted(uids_of_pod):
        workload = compute_workload_of_pod(pod_name, workloads=workloads)
        assert workload is not None
        if len(uids := uids_of_pod[pod_name]) > 1:
            reasons_of_workload.setdefault(workload, []).append(f"pod {pod_name} was recreated as {sorted(uids)}")
        if len(counts := restart_counts_of_pod[pod_name]) > 1:
            reasons_of_workload.setdefault(workload, []).append(
                f"pod {pod_name} restarted a container: restartCount went through {sorted(counts)}"
            )
    for workload in workloads:
        if len(name_sets := pod_names_of_workload.get(workload, set())) > 1:
            reasons_of_workload.setdefault(workload, []).append(
                f"the pods of {workload} came and went: {sorted(sorted(one) for one in name_sets)}"
            )

    return {workload: sorted(reasons) for workload, reasons in sorted(reasons_of_workload.items())}


def compute_workloads_whose_template_changed(snapshots: Sequence[ClusterSnapshot]) -> set[str]:
    generations_of_workload: dict[str, set[int]] = {}
    stamps_of_workload = compute_restart_stamps_of_workload(snapshots)
    for snapshot in snapshots:
        for workload in snapshot.workloads:
            generations_of_workload.setdefault(workload.name, set()).add(workload.generation)
    return {
        name
        for name, generations in generations_of_workload.items()
        if len(generations) > 1 or len(stamps_of_workload.get(name, set())) > 1
    }


def compute_restart_stamps_of_workload(snapshots: Sequence[ClusterSnapshot]) -> dict[str, set[str]]:
    stamps_of_workload: dict[str, set[str]] = {}
    for snapshot in snapshots:
        for workload in snapshot.workloads:
            stamps = stamps_of_workload.setdefault(workload.name, set())
            if workload.restart_at is not None:
                stamps.add(workload.restart_at)
    return stamps_of_workload


def compute_trainer_boot_uuids(snapshots: Sequence[ClusterSnapshot]) -> set[str]:
    return {one.trainer_boot_uuid for one in snapshots if one.trainer_boot_uuid is not None}
