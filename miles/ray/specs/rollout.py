from miles.utils.workers.naming import compute_cell_id, compute_worker_name
from miles.utils.workers.types import ClusterBackend
from miles.utils.workers.worker_spec import SchedulingSpec, ServeWorkerSpec

ROLLOUT_EXECUTOR_SPEC_NAME = "rollout-executor"
ROLLOUT_EXECUTOR_WORKER_CLASS = "miles.ray.rollout.rollout_executor.RolloutExecutor"
ROLLOUT_EXECUTOR_RPC_CLASS = "miles.ray.rollout.rollout_executor_api.RolloutExecutorApi"


def spec_rollout_executor(args) -> ServeWorkerSpec:
    return ServeWorkerSpec(
        name=ROLLOUT_EXECUTOR_SPEC_NAME,
        port_infos=[],
        env_var=lambda _ctx: {},
        scheduling=SchedulingSpec(
            num_cells=1 if ClusterBackend(args.cluster_backend) is ClusterBackend.KUBERNETES else 0,
            num_workers_per_cell=1,
            num_gpus_per_worker=0,
            num_cpus_per_worker=1,
            pin_to_head=args.pin_rollout_manager_to_head,
        ),
        worker_class=ROLLOUT_EXECUTOR_WORKER_CLASS,
        rpc_class=ROLLOUT_EXECUTOR_RPC_CLASS,
        ctor_kwargs=lambda _ctx: dict(args=args),
    )


def rollout_executor_worker_name() -> str:
    return compute_worker_name(spec_name=ROLLOUT_EXECUTOR_SPEC_NAME)


def rollout_executor_cell_id() -> str:
    return compute_cell_id(spec_name=ROLLOUT_EXECUTOR_SPEC_NAME, cell_index=0)
