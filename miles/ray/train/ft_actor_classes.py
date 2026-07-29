from typing import Any


def __getattr__(name: str) -> Any:
    if name == "MegatronTrainRayActorFt":
        from miles.backends.megatron_utils.actor import MegatronTrainRayActor
        from miles.ray.train.actor_factory import _with_ft_concurrency_groups

        return _with_ft_concurrency_groups(MegatronTrainRayActor)
    if name == "FSDPTrainRayActorFt":
        from miles.backends.experimental.fsdp_utils.actor import FSDPTrainRayActor
        from miles.ray.train.actor_factory import _with_ft_concurrency_groups

        return _with_ft_concurrency_groups(FSDPTrainRayActor)
    raise AttributeError(name)
