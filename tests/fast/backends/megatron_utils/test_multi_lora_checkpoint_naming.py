"""Megatron-native adapter shards are keyed by parallel coordinates.

Expert-parallel ranks that share a ``(tp, pp)`` coordinate hold *different* local
experts, so their adapter shards are different files. Without the ep suffix they
overwrite each other and a resume loads one EP rank's experts onto every rank.

Resume must also wait for exactly the shards that exist. The realized
``(tp, pp, ep)`` coordinates are not the cross product of the group sizes: with
expert tensor parallelism smaller than tensor parallelism — what expert multi-LoRA
requires — only a subset of ``(tp, ep)`` pairs is occupied, so enumerating the
cross product would wait for shards no rank ever writes.
"""

from miles.backends.megatron_utils.multi_lora_utils import all_megatron_checkpoints_exist, megatron_shard_name


def _names(coords, ep_size):
    return {megatron_shard_name(*coord, ep_size) for coord in coords}


def test_shard_name_omits_ep_suffix_without_expert_parallelism():
    # Checkpoints written before expert adapters existed must stay loadable.
    assert megatron_shard_name(0, 0, 0, ep_size=1) == "adapter_megatron_tp0_pp0.pt"
    assert megatron_shard_name(1, 2, 0, ep_size=1) == "adapter_megatron_tp1_pp2.pt"


def test_shard_name_is_unique_per_expert_parallel_rank():
    names = {megatron_shard_name(0, 0, ep, ep_size=4) for ep in range(4)}
    assert len(names) == 4
    assert megatron_shard_name(0, 0, 2, ep_size=4) == "adapter_megatron_tp0_pp0_ep2.pt"


def test_completeness_check_requires_every_realized_shard(tmp_path):
    coords = [(0, 0, 0), (0, 0, 1), (0, 0, 2)]
    for coord in coords[:2]:
        (tmp_path / megatron_shard_name(*coord, 3)).touch()

    assert not all_megatron_checkpoints_exist(tmp_path, _names(coords, 3))

    (tmp_path / megatron_shard_name(*coords[2], 3)).touch()
    assert all_megatron_checkpoints_exist(tmp_path, _names(coords, 3))


def test_completeness_ignores_unrealized_coordinates(tmp_path):
    # TP=2, EP=2 with ETP=1: expert-parallel ranks are carved out of the
    # tensor-parallel dimension, so (tp=0, ep=1) and (tp=1, ep=0) do not exist.
    # A cross-product check would demand four shards and never resume.
    coords = [(0, 0, 0), (1, 0, 1)]
    for coord in coords:
        (tmp_path / megatron_shard_name(*coord, 2)).touch()

    assert all_megatron_checkpoints_exist(tmp_path, _names(coords, 2))


def test_completeness_check_with_single_shard(tmp_path):
    (tmp_path / "adapter_megatron_tp0_pp0.pt").touch()
    assert all_megatron_checkpoints_exist(tmp_path, _names([(0, 0, 0)], 1))
