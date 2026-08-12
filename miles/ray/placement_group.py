import asyncio
import logging
import socket
from typing import NamedTuple

import ray
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.ray.rollout.router_manager import resolve_router_addrs, wait_session_server_ready
from miles.ray.specs.inference import (
    SESSION_SERVER_POOL_ID,
    compute_router_providers,
    create_inference_controller_handle,
)
from miles.ray.specs.rollout import create_rollout_executor_handle
from miles.ray.specs.static_addrs import (
    INFERENCE_CONTROLLER_ADDRS_FLAG,
    TRAINER_CONTROLLER_ADDRS_FLAG,
    assert_deployment_names_this_run,
    assert_routers_belong_to_inference_deployment,
    inference_controller_urls,
    trainer_controller_urls,
)
from miles.ray.specs.train import compute_critic_args, create_trainer_controller_handle
from miles.ray.wiring import get_backend_capability
from miles.utils.audit_utils.checksum_utils import flatten_inference_engine_checksums
from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent
from miles.utils.ft_utils.api_server.server import start_api_server
from miles.utils.workers.types import DeployComponent
from miles.utils.workers.worker_handle import BaseWorkerHandle

logger = logging.getLogger(__name__)

UPDATE_WEIGHTS_STOP_CONFIRMATION_TIMEOUT_SECONDS = 600.0


@ray.remote(num_gpus=1)
class InfoActor:
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]


def sort_key(x):
    index, node_identifier, gpu_id = x
    # Sort by node IP number and then by GPU ID
    try:
        # try to parse it as an IP address.
        ip_address = node_identifier
        node_ip_parts = list(map(int, ip_address.split(".")))
    except ValueError:
        # Try to resolve the hostname to an IP address.
        try:
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            # Instead, we convert each character of the original identifier string
            # to its ASCII value. This provides a stable and consistent numerical
            # representation that allows for sorting.
            node_ip_parts = [ord(c) for c in node_identifier]

    return (node_ip_parts, gpu_id)


class PlacementGroupInfo(NamedTuple):
    pg: PlacementGroup
    pg_reordered_bundle_indices: list[int]
    pg_reordered_gpu_ids: list[int]


def _create_placement_group(num_gpus) -> PlacementGroupInfo:
    """Create a placement group with the specified number of GPUs."""
    if num_gpus == 0:
        return None, [], []

    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")
    num_bundles = len(bundles)

    ray.get(pg.ready())
    # use info actor to get the GPU id
    info_actors = []
    for i in range(num_bundles):
        info_actors.append(
            InfoActor.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                )
            ).remote()
        )
    gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
    for actor in info_actors:
        ray.kill(actor)

    bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]
    sorted_bundle_infos = sorted(bundle_infos, key=sort_key)
    pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
    # Map from logical index -> physical GPU ID
    pg_reordered_gpu_ids = [gpu_ids[info[0]][1] for info in sorted_bundle_infos]

    for i in range(num_bundles):
        actual_bundle_index = pg_reordered_bundle_indices[i]
        logger.info(
            f"  bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, "
            f"node: {gpu_ids[actual_bundle_index][0]}, gpu: {gpu_ids[actual_bundle_index][1]}"
        )

    return PlacementGroupInfo(pg, pg_reordered_bundle_indices, pg_reordered_gpu_ids)


def _get_placement_group_layout(args) -> tuple[int, int]:
    actor_num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node

    component = DeployComponent(args.deploy_component)
    if component is DeployComponent.PRIMARY:
        return 0, 0
    if component is DeployComponent.TRAINER:
        return actor_num_gpus, 0
    if component is DeployComponent.INFERENCE:
        if args.debug_train_only or args.rollout_external:
            return 0, 0
        return args.rollout_num_gpus + args.eval_num_gpus, 0

    if args.debug_train_only:
        return actor_num_gpus, 0
    if args.rollout_external:
        if args.debug_rollout_only:
            return 0, 0
        return actor_num_gpus, actor_num_gpus
    if args.debug_rollout_only:
        return args.rollout_num_gpus, 0
    if args.colocate:
        return max(actor_num_gpus, args.rollout_num_gpus), 0
    return actor_num_gpus + args.rollout_num_gpus + args.eval_num_gpus, actor_num_gpus


def create_placement_groups(args) -> dict[str, PlacementGroupInfo]:
    """Create placement groups for actor and rollout engines."""

    num_gpus, rollout_offset = _get_placement_group_layout(args)

    logger.info(f"Creating placement group with {num_gpus} GPUs...")
    pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids = _create_placement_group(num_gpus)

    rollout_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[rollout_offset:]
    rollout_pg_reordered_gpu_ids = actor_pg_reordered_gpu_ids[rollout_offset:]
    ans = {
        "actor": PlacementGroupInfo(pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids),
        "rollout": PlacementGroupInfo(pg, rollout_pg_reordered_bundle_indices, rollout_pg_reordered_gpu_ids),
    }
    if args.use_critic:
        ans["critic"] = ans["actor"]
    return ans


# TODO: move (when reorganizing files)
async def create_training_models(
    args, rollout_executor: BaseWorkerHandle
) -> tuple[BaseWorkerHandle, BaseWorkerHandle | None]:
    capability = get_backend_capability(args)

    actor_model = create_trainer_controller_handle(args, capability=capability, role="actor")
    await _assert_trainer_names_this_run(args, trainer_controller=actor_model, role="actor")
    actor_start_rollout_ids = await actor_model.init(args)

    if args.use_critic:
        critic_model = create_trainer_controller_handle(args, capability=capability, role="critic")
        await _assert_trainer_names_this_run(args, trainer_controller=critic_model, role="critic")
        critic_start_rollout_ids = await critic_model.init(compute_critic_args(args))
    else:
        critic_model = None

    start_rollout_ids = critic_start_rollout_ids if args.use_critic else actor_start_rollout_ids

    assert len(set(start_rollout_ids)) == 1
    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_ids[0]

    await rollout_executor.set_train_parallel_config(await actor_model.get_train_parallel_config())
    await rollout_executor.load(args.start_rollout_id - 1)

    return actor_model, critic_model


# TODO: move (when reorganizing files)
async def update_weights(
    args,
    *,
    actor_model: BaseWorkerHandle,
    rollout_executor: BaseWorkerHandle,
    inference_controller: BaseWorkerHandle,
    rollout_id: int | None = None,
) -> None:
    """Sequence the weight update: the controllers never call each other, the orchestration script does."""
    info = await inference_controller.start_update_weights()
    try:
        weight_version = await actor_model.update_weights(info=info, rollout_id=rollout_id)
    except BaseException:
        await _abort_update_weights(
            actor_model=actor_model, inference_controller=inference_controller, window_id=info.window_id
        )
        raise
    await inference_controller.end_update_weights(
        window_id=info.window_id, snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes
    )

    await _maybe_log_inference_engine_weight_checksums(
        args, inference_controller=inference_controller, rollout_id=rollout_id
    )

    if weight_version is not None:
        await rollout_executor.set_weight_version(weight_version)


async def _abort_update_weights(
    *, actor_model: BaseWorkerHandle, inference_controller: BaseWorkerHandle, window_id: int
) -> None:
    try:
        confirmed = await asyncio.wait_for(
            actor_model.wait_update_weights_finished(window_id=window_id),
            timeout=UPDATE_WEIGHTS_STOP_CONFIRMATION_TIMEOUT_SECONDS,
        )
    except BaseException:
        logger.exception(
            f"The trainer never confirmed that it stopped broadcasting in update weights window {window_id}, so the "
            f"inference controller keeps the window's lock and its paused health checking rather than let an engine "
            f"the broadcast may still be writing to be restarted"
        )
        return
    if not confirmed:
        logger.error(
            f"The trainer answered that it is still broadcasting in update weights window {window_id}, so the "
            f"inference controller keeps the window's lock and its paused health checking rather than let an engine "
            f"the broadcast is still writing to be restarted"
        )
        return
    await inference_controller.abort_update_weights(window_id=window_id)


async def _assert_trainer_names_this_run(args, *, trainer_controller: BaseWorkerHandle, role: str) -> None:
    if trainer_controller_urls(args, role=role) is None:
        return
    assert_deployment_names_this_run(
        await trainer_controller.get_deployment_identity(), args=args, flag=TRAINER_CONTROLLER_ADDRS_FLAG
    )


async def _assert_inference_names_this_run(args, *, inference_controller: BaseWorkerHandle) -> None:
    if inference_controller_urls(args) is None:
        return
    identity = await inference_controller.get_deployment_identity()
    assert_deployment_names_this_run(identity, args=args, flag=INFERENCE_CONTROLLER_ADDRS_FLAG)
    assert_routers_belong_to_inference_deployment(identity, args=args)


async def _maybe_log_inference_engine_weight_checksums(
    args, *, inference_controller: BaseWorkerHandle, rollout_id: int | None
) -> None:
    if not is_event_logger_initialized():
        return
    if args.debug_train_only or args.debug_rollout_only:
        return

    check_weights_result = await inference_controller.check_weights(action="checksum")
    get_event_logger().log(
        InferenceEngineWeightChecksumEvent,
        dict(rollout_id=rollout_id, engine_checksums=flatten_inference_engine_checksums(check_weights_result)),
    )


# TODO: move (when reorganizing files)
def maybe_start_api_server(args, *, actor_model: BaseWorkerHandle, inference_controller: BaseWorkerHandle) -> None:
    if not args.api_server_port:
        return

    start_api_server(
        args=args,
        actor_model=actor_model,
        inference_controller=inference_controller,
        port=args.api_server_port,
        ft_components=args.ft_components,
        cell_operations=get_backend_capability(args).cell_operations(),
    )


class RolloutComponents(NamedTuple):
    inference_controller: BaseWorkerHandle
    rollout_executor: BaseWorkerHandle
    num_rollout_per_epoch: int | None


# TODO: move (when reorganizing files)
async def create_rollout_components(args) -> RolloutComponents:
    capability = get_backend_capability(args)

    if not args.debug_train_only:
        await resolve_router_addrs(args, router_providers=compute_router_providers(args, capability=capability))

        session_server_provider = (
            capability.static_worker_provider(pool_id=SESSION_SERVER_POOL_ID) if args.use_session_server else None
        )
        await wait_session_server_ready(args, provider=session_server_provider)

    inference_controller = create_inference_controller_handle(args, capability=capability)
    await inference_controller.init()
    await _assert_inference_names_this_run(args, inference_controller=inference_controller)

    rollout_executor = create_rollout_executor_handle(capability=capability)
    await rollout_executor.init()

    # calculate num_rollout from num_epoch
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        num_rollout_per_epoch = await rollout_executor.get_num_rollout_per_epoch()
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
        assert args.num_rollout > 0

    if (eval_fleet := await inference_controller.get_eval_fleet()) is not None:
        await rollout_executor.set_eval_fleet(eval_fleet)

    return RolloutComponents(
        inference_controller=inference_controller,
        rollout_executor=rollout_executor,
        num_rollout_per_epoch=num_rollout_per_epoch,
    )
