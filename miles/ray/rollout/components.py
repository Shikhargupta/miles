from typing import NamedTuple

from miles.ray.rollout.executor_handle import create_rollout_executor_handle
from miles.ray.rollout.inference_controller import InferenceController
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_provider.factory import ProviderFactory


class RolloutComponents(NamedTuple):
    inference_controller: InferenceController
    rollout_executor: BaseWorkerHandle
    num_rollout_per_epoch: int | None


async def create_rollout_components(args, *, providers: ProviderFactory) -> RolloutComponents:
    inference_controller = await InferenceController.create(args, providers=providers)

    rollout_executor = create_rollout_executor_handle(args, providers=providers)

    # calculate num_rollout from num_epoch
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        num_rollout_per_epoch = await rollout_executor.get_num_rollout_per_epoch()
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
        assert args.num_rollout > 0

    return RolloutComponents(
        inference_controller=inference_controller,
        rollout_executor=rollout_executor,
        num_rollout_per_epoch=num_rollout_per_epoch,
    )
