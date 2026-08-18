from tests.fast.cluster_backends import RUN_NAMESPACE_ENV_VAR, create_backend_for_run

from miles.utils.external_utils import command_utils
from miles.utils.workers.types import ClusterBackend


def is_cluster_ready_for_helm_runs(config: command_utils.ExecuteTrainConfig) -> bool:
    if (reason := _compute_unconfigured_reason(config)) is not None:
        print(f"SKIPPED: {reason}")
        return False

    create_backend_for_run(config)
    return True


def _compute_unconfigured_reason(config: command_utils.ExecuteTrainConfig) -> str | None:
    if (backend := config.cluster_backend) is not ClusterBackend.KUBERNETES:
        return (
            f"these tests install a run as helm releases, which only the {ClusterBackend.KUBERNETES.value} "
            f"backend does, and this environment declares the {backend.value} backend"
        )
    if not config.namespace:
        return (
            f"{RUN_NAMESPACE_ENV_VAR} is unset, so this environment declares no namespace to install the "
            f"releases of a run into; see docs/advanced/cluster-backend.md"
        )
    return None
