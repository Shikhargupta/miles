"""Translate torchtitan's ParallelDims into miles' shared ParallelState.

ParallelState is what every shared helper in ``training_utils`` reads (data
iteration, loss normalization, logging), so a backend's only topology
responsibility is producing one. torchtitan already owns the mesh construction;
this is a pure mapping over its named meshes.
"""

import logging
from argparse import Namespace

import torch.distributed as dist

from miles.backends.training_utils.parallel import ParallelState
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.ft_utils.process_group_utils import GroupInfo

logger = logging.getLogger(__name__)

# titan mesh name -> miles ParallelState field. titan's "batch" mesh is
# dp_replicate x dp_shard (the sample-parallel view) and "loss" additionally
# folds in cp, which is exactly miles' intra_dp / intra_dp_cp split.
_MESH_TO_FIELD = {
    "batch": "intra_dp",
    "loss": "intra_dp_cp",
    "cp": "cp",
    "tp": "tp",
    "pp": "pp",
    "ep": "ep",
}


def build_parallel_dims(args: Namespace):
    """Construct titan ParallelDims from miles arguments."""
    from torchtitan.distributed import ParallelDims

    world_size = dist.get_world_size()
    dp_replicate = args.dp_replicate_size
    non_dp = args.titan_cp_size * args.titan_tp_size * args.titan_pp_size
    if world_size % (dp_replicate * non_dp):
        raise ValueError(
            f"world_size({world_size}) is not divisible by "
            f"dp_replicate({dp_replicate}) * cp({args.titan_cp_size}) * "
            f"tp({args.titan_tp_size}) * pp({args.titan_pp_size})"
        )
    dp_shard = world_size // (dp_replicate * non_dp)

    return ParallelDims(
        dp_replicate=dp_replicate,
        dp_shard=dp_shard,
        cp=args.titan_cp_size,
        tp=args.titan_tp_size,
        pp=args.titan_pp_size,
        ep=args.titan_ep_size,
        world_size=world_size,
    )


def create_titan_parallel_state(parallel_dims) -> ParallelState:
    """Map titan's meshes onto ParallelState.

    Axes titan leaves at degree 1 have no mesh; they become trivial single-rank
    groups, which is what the shared helpers expect for a disabled dimension.
    """
    rank = dist.get_rank()
    self_group = dist.new_group([rank])
    trivial = GroupInfo(rank=0, size=1, group=self_group)

    fields: dict[str, GroupInfo] = {}
    for mesh_name, field in _MESH_TO_FIELD.items():
        mesh = parallel_dims.get_optional_mesh(mesh_name)
        if mesh is None:
            fields[field] = trivial
            continue
        group = mesh.get_group()
        fields[field] = GroupInfo(
            rank=dist.get_rank(group=group),
            size=dist.get_world_size(group=group),
            group=group,
            # the DP-CP group is the one shared helpers reduce metrics over, and
            # some of those reductions are object-based (gloo, not nccl).
            gloo_group=get_gloo_group() if field == "intra_dp_cp" else None,
        )

    meshes = {name: parallel_dims.get_mesh(name) for name in ("fsdp",) if parallel_dims.get_optional_mesh(name)}

    state = ParallelState(
        intra_dp=fields["intra_dp"],
        intra_dp_cp=fields["intra_dp_cp"],
        cp=fields["cp"],
        tp=fields["tp"],
        pp=fields["pp"],
        ep=fields["ep"],
        # titan has no separate expert-tensor axis; its EP region uses "efsdp".
        etp=trivial,
        indep_dp=trivial,
        meshes=meshes,
        is_pp_last_stage=True,
        vpp_size=1,
    )
    logger.info(
        f"[Rank {rank}] titan ParallelState: dp={state.intra_dp.size} dp_cp={state.intra_dp_cp.size} "
        f"cp={state.cp.size} tp={state.tp.size} pp={state.pp.size} ep={state.ep.size}"
    )
    return state
