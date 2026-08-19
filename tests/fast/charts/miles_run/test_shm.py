import json
from typing import Any

from tests.fast.charts.utils import objects_of_kind, render_run, requires_helm, with_object_names

TRAINER = [
    {
        "name": "trainer-engine-actor",
        "command": ["python", "-m", "miles.utils.workers.process_supervisor"],
        "resources": {"limits": {"nvidia.com/gpu": 4}},
    }
]


def _pod_spec_of_the_only_pool(*args: str) -> dict[str, Any]:
    rendered = render_run("--set-json", f"run.trainerEngines={json.dumps(with_object_names(TRAINER))}", *args)
    pool = objects_of_kind(rendered, "LeaderWorkerSet")[0]
    return pool["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]


def _shm_volume(pod_spec: dict[str, Any]) -> dict[str, Any]:
    [mount] = _shm_mounts(pod_spec)
    [volume] = [volume for volume in pod_spec["volumes"] if volume["name"] == mount["name"]]
    return volume


def _shm_mounts(pod_spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        mount
        for container in pod_spec["containers"]
        for mount in container.get("volumeMounts", [])
        if mount["mountPath"] == "/dev/shm"
    ]


@requires_helm
class TestPoolPodsGetSharedMemoryNcclCanUse:
    def test_a_pool_container_mounts_dev_shm(self):
        """Kubernetes' default 64Mi of /dev/shm is less than NCCL asks for per peer it cannot reach over p2p."""
        assert len(_shm_mounts(_pod_spec_of_the_only_pool())) == 1

    def test_that_mount_is_backed_by_memory_rather_than_the_node_disk(self):
        """A disk-backed tmpfs would let NCCL rendezvous succeed while running at disk speed."""
        assert _shm_volume(_pod_spec_of_the_only_pool())["emptyDir"]["medium"] == "Memory"

    def test_the_volume_is_bounded_so_it_cannot_eat_the_node(self):
        """A memory-backed emptyDir without a limit is charged to the node until it is out of ram."""
        assert _shm_volume(_pod_spec_of_the_only_pool())["emptyDir"]["sizeLimit"] == "32Gi"

    def test_a_run_that_needs_more_or_less_of_it_may_say_so(self):
        """The default is copied from the docker quick start, not measured, so it has to be overridable."""
        pod_spec = _pod_spec_of_the_only_pool("--set", "run.shmSize=8Gi")

        assert _shm_volume(pod_spec)["emptyDir"]["sizeLimit"] == "8Gi"

    def test_a_suffixless_size_still_reaches_kubernetes_as_a_quantity(self):
        """An unquoted 32 is a yaml integer, and a resource quantity that is not a string is rejected."""
        pod_spec = _pod_spec_of_the_only_pool("--set-string", "run.shmSize=32")

        assert _shm_volume(pod_spec)["emptyDir"]["sizeLimit"] == "32"

    def test_every_mounted_volume_is_declared_by_the_pod_that_mounts_it(self):
        """A container naming a volume the pod does not declare makes the whole manifest invalid."""
        pod_spec = _pod_spec_of_the_only_pool()
        declared = {volume["name"] for volume in pod_spec["volumes"]}
        mounted = {mount["name"] for container in pod_spec["containers"] for mount in container["volumeMounts"]}

        assert mounted <= declared, f"{mounted - declared} is mounted but never declared"

    def test_a_pod_that_runs_no_collective_is_left_alone(self):
        """The orchestrator only talks to the apiserver, so shared memory it never uses is wasted ram."""
        rendered = render_run("--set-json", f"run.trainerEngines={json.dumps(with_object_names(TRAINER))}")
        [orchestrator] = [
            obj for obj in objects_of_kind(rendered, "StatefulSet") if "orchestrator" in obj["metadata"]["name"]
        ]

        assert not _shm_mounts(orchestrator["spec"]["template"]["spec"])
