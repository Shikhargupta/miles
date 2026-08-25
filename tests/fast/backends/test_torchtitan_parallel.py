"""torchtitan ParallelDims -> miles ParallelState mapping.

ParallelState is what every shared helper reads (data iteration, loss
normalization, metric reduction), so a wrong axis here does not crash -- it
silently normalizes the loss over the wrong group size. Worth pinning even
though the mapping is short.
"""

from unittest.mock import patch

import pytest

from miles.backends.torchtitan_utils import parallel as tp


class _Mesh:
    def __init__(self, size: int, rank: int = 0):
        self._size = size
        self._rank = rank
        self._group = f"pg(size={size})"

    def get_group(self):
        return self._group


class _ParallelDims:
    """Only the surface create_titan_parallel_state touches."""

    def __init__(self, meshes: dict[str, int]):
        self._meshes = {name: _Mesh(size) for name, size in meshes.items()}

    def get_optional_mesh(self, name):
        return self._meshes.get(name)

    def get_mesh(self, name):
        mesh = self._meshes.get(name)
        if mesh is None:
            raise ValueError(f"mesh {name} not available")
        return mesh


@pytest.fixture
def dist_stub():
    """dist calls resolve against the fake meshes' recorded sizes.

    GroupInfo validates that an attached group reports the declared size, so the
    stub has to answer for the gloo group as well as for each mesh group.
    """
    sizes: dict = {}

    def get_world_size(group=None):
        return sizes.get(group, sizes.get("__world__", 1))

    with (
        patch.object(tp.dist, "get_rank", lambda group=None: 0),
        patch.object(tp.dist, "get_world_size", get_world_size),
        patch.object(tp.dist, "new_group", lambda ranks, backend=None: "self_group" if backend is None else f"gloo_sub{tuple(ranks)}"),
        patch.object(tp.dist, "get_process_group_ranks", lambda group: list(range(sizes.get(group, 1)))),
        patch.object(tp.dist, "all_gather_object", lambda out, obj, group=None: out.__setitem__(slice(None), [obj] * len(out))),
        patch.object(tp, "get_gloo_group", lambda: "gloo"),
    ):
        yield sizes


def _state(dist_stub, meshes: dict[str, int], world: int | None = None, **kwargs):
    for size in meshes.values():
        dist_stub[f"pg(size={size})"] = size
    world = world if world is not None else meshes.get("loss", 1)
    dist_stub["__world__"] = world
    dist_stub["gloo"] = world
    dist_stub["self_group"] = 1
    dp_cp = meshes.get("loss", 1)
    dist_stub[f"gloo_sub{tuple(range(dp_cp))}"] = dp_cp
    return tp.create_titan_parallel_state(_ParallelDims(meshes), **kwargs)


def test_batch_and_loss_meshes_map_to_the_dp_axes(dist_stub):
    """titan's 'batch' mesh is dp_replicate x dp_shard; 'loss' folds cp in as
    well -- which is exactly miles' intra_dp / intra_dp_cp split."""
    state = _state(dist_stub, {"batch": 4, "loss": 8, "cp": 2})
    assert state.intra_dp.size == 4
    assert state.intra_dp_cp.size == 8
    assert state.cp.size == 2


def test_absent_axes_become_trivial_single_rank_groups(dist_stub):
    """titan gives no mesh for a degree-1 axis; the shared helpers expect a
    trivial group there, not None."""
    state = _state(dist_stub, {"batch": 2, "loss": 2})
    for axis in (state.tp, state.pp, state.ep, state.etp, state.indep_dp):
        assert axis.size == 1
        assert axis.rank == 0


def test_model_parallel_axes_are_carried_through(dist_stub):
    state = _state(dist_stub, {"batch": 4, "loss": 8, "tp": 4, "pp": 2, "ep": 8})
    assert (state.tp.size, state.pp.size, state.ep.size) == (4, 2, 8)


def test_a_dp_cp_group_narrower_than_the_world_gets_its_own_gloo_subgroup(dist_stub):
    """Under model parallelism DP-CP does not span the world, so the world-wide
    gloo group cannot be attached (GroupInfo checks sizes); a congruent gloo
    subgroup is built by enumeration instead."""
    state = _state(dist_stub, {"batch": 2, "loss": 2, "tp": 4}, world=8)
    assert state.intra_dp_cp.gloo_group == "gloo_sub(0, 1)"


def test_the_dp_cp_group_gets_a_gloo_group(dist_stub):
    """Some shared reductions over this axis are object-based, so they need a
    CPU-capable group rather than the nccl one."""
    state = _state(dist_stub, {"batch": 2, "loss": 2})
    assert state.intra_dp_cp.gloo_group == "gloo"
    assert state.intra_dp.gloo_group is None


def test_titan_has_no_expert_tensor_axis(dist_stub):
    """titan carves its EP region out with an 'efsdp' axis instead of a separate
    expert-tensor-parallel dimension, so etp stays trivial."""
    state = _state(dist_stub, {"batch": 4, "loss": 4, "ep": 4, "efsdp": 2})
    assert state.etp.size == 1
    assert state.ep.size == 4


def test_vpp_is_disabled(dist_stub):
    """Virtual pipeline parallelism is a Megatron scheduling concept; the data
    layer keys microbatch grouping off this, so it must be 1 here."""
    state = _state(dist_stub, {"batch": 2, "loss": 2})
    assert state.vpp_size == 1
    assert state.is_pp_last_stage is True


def test_pp_last_stage_comes_from_the_trainer_not_the_mesh(dist_stub):
    """Interleaved schedules can place the last stage on any rank, so the flag
    is the trainer's stage placement, not a mesh-rank comparison."""
    state = _state(dist_stub, {"batch": 2, "loss": 2, "pp": 2}, is_pp_last_stage=False)
    assert state.is_pp_last_stage is False
