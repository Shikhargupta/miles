from miles.utils.workers.worker_spec import SchedulingSpec, ServeWorkerSpec


def spec_rollout_executor(args) -> ServeWorkerSpec:
    return ServeWorkerSpec(
        name="rollout-executor",
        port_infos=[],
        env_var=lambda: {},
        scheduling=SchedulingSpec(num_cells=1, num_workers_per_cell=1, num_gpus_per_worker=0),
        worker_class="miles.ray.rollout.rollout_executor.RolloutExecutor",
        ctor_kwargs=lambda: dict(args=args),
    )
