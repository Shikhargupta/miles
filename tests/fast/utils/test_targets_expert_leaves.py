"""targets_expert_leaves decides whether adapters can land inside MoE experts.

It gates the MoE-specific multi-LoRA handling (turning off permute fusion, the
expert-parallel validations), so a false negative means those are silently skipped
and expert tokens get routed against a permutation the adapter cannot replay.
"""

from miles.utils.multi_lora import targets_expert_leaves


def test_mlp_leaf_names_target_experts():
    # These names match the dense MLP and the routed experts alike.
    assert targets_expert_leaves(["gate_proj", "up_proj", "down_proj"])
    assert targets_expert_leaves(["linear_fc1"])
    assert targets_expert_leaves(["linear_fc2"])


def test_expert_scoped_wildcards_target_experts():
    assert targets_expert_leaves(["*.layers.*.mlp.experts.linear_fc1"])


def test_attention_only_targets_do_not():
    assert not targets_expert_leaves(["linear_qkv", "linear_proj"])
    assert not targets_expert_leaves(["q_proj", "k_proj", "v_proj", "o_proj"])


def test_bulk_aliases_target_experts():
    # "all-linear" is expanded to concrete names during argument validation, but
    # "all" is only resolved later by the target-module conversion, so the alias
    # itself has to count.
    for alias in ("all", "all-linear", "all_linear", "ALL"):
        assert targets_expert_leaves([alias]), alias


def test_bare_string_is_accepted():
    assert targets_expert_leaves("gate_proj")
    assert not targets_expert_leaves("linear_qkv")


def test_empty_targets_do_not():
    assert not targets_expert_leaves(None)
    assert not targets_expert_leaves([])
