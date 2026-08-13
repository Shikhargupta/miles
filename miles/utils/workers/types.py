from enum import Enum

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class ClusterBackend(Enum):
    RAY = "ray"
    KUBERNETES = "kubernetes"


class WorkerCommBackend(Enum):
    RAY = "ray"
    RPC = "rpc"


class DeployComponent(Enum):
    ALL = "all"
    PRIMARY = "primary"
    TRAINER = "trainer"
    INFERENCE = "inference"

    def selects(self, component: "DeployComponent") -> bool:
        assert component is not DeployComponent.ALL, "`all` is a selector over components, never a component itself"
        return self is DeployComponent.ALL or self is component

    def deploys_orchestration_script(self) -> bool:
        return self.selects(DeployComponent.PRIMARY)

    def is_split(self) -> bool:
        return self is not DeployComponent.ALL

    def takes_instance(self) -> bool:
        return self in (DeployComponent.TRAINER, DeployComponent.INFERENCE)


class HotRestartComponent(Enum):
    ORCHESTRATION = "orchestration"
    ROLLOUT_EXECUTOR = "rollout_executor"


HOT_RESTART_SEPARATOR = ","


def parse_hot_restart(value: str) -> frozenset[HotRestartComponent]:
    names = [name.strip() for name in value.split(HOT_RESTART_SEPARATOR) if name.strip()]
    supported = [one.value for one in HotRestartComponent]
    for name in names:
        assert name in supported, (
            f"--hot-restart {value!r} names {name!r}, and a hot restart replaces only {supported}: every other "
            f"component of a run stays up and is taken over by the new orchestration script"
        )
    components = frozenset(HotRestartComponent(name) for name in names)
    assert not components or components == frozenset(HotRestartComponent), (
        f"--hot-restart {value!r} names {sorted(one.value for one in components)}, and the two components are "
        f"replaced together or not at all: the new orchestration script cannot drive the executor its predecessor "
        f"initialized, and an executor replaced under a live script kills the run it belongs to"
    )
    return components


class DeploymentIdentity(FrozenStrictBaseModel):
    run_uuid: str
    deploy_component: str
    deploy_instance: str | None = None


_SUPPORTED_WORKER_COMM_BACKENDS = {
    ClusterBackend.RAY: (WorkerCommBackend.RAY, WorkerCommBackend.RPC),
    ClusterBackend.KUBERNETES: (WorkerCommBackend.RPC,),
}


def resolve_worker_comm_backend(*, cluster_backend: ClusterBackend, requested: str | None) -> WorkerCommBackend:
    if requested is None:
        return _SUPPORTED_WORKER_COMM_BACKENDS[cluster_backend][0]

    backend = WorkerCommBackend(requested)
    supported = _SUPPORTED_WORKER_COMM_BACKENDS[cluster_backend]
    assert backend in supported, (
        f"--worker-comm-backend {backend.value} is not available under --cluster-backend {cluster_backend.value}, "
        f"which speaks {[one.value for one in supported]}"
    )
    return backend
