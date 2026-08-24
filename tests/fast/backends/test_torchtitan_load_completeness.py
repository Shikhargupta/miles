from miles.backends.torchtitan_utils.model import unloaded_parameters

WRAPPED = "layers.0._checkpoint_wrapped_module.moe.experts.w1_EFD"
UNWRAPPED = "layers.0.moe.experts.w1_EFD"


def test_missing_buffer_is_not_a_failure():
    # No HF export contains expert_bias_E; init_weights already set it up.
    assert unloaded_parameters(["layers.0.moe.expert_bias_E"], {UNWRAPPED}) == []


def test_missing_parameter_is_reported():
    assert unloaded_parameters([UNWRAPPED], {UNWRAPPED}) == [UNWRAPPED]


def test_wrapped_missing_key_against_unwrapped_parameter_names():
    # Activation checkpointing inserts _checkpoint_wrapped_module on one side only.
    # Comparing the two conventions directly matches nothing, which is what turned
    # this check into one that could never fail.
    assert unloaded_parameters([WRAPPED], {UNWRAPPED}) == [WRAPPED]


def test_unwrapped_missing_key_against_wrapped_parameter_names():
    assert unloaded_parameters([UNWRAPPED], {WRAPPED}) == [UNWRAPPED]


def test_moe_shape_separates_expert_params_from_the_bias_buffer():
    missing = [
        "layers.0._checkpoint_wrapped_module.moe.experts.w1_EFD",
        "layers.0._checkpoint_wrapped_module.moe.experts.w2_EDF",
        "layers.0.moe.expert_bias_E",
    ]
    parameter_names = {
        "layers.0.moe.experts.w1_EFD",
        "layers.0.moe.experts.w2_EDF",
        "layers.0.attention.wq.weight",
    }
    assert unloaded_parameters(missing, parameter_names) == missing[:2]


def test_nothing_missing():
    assert unloaded_parameters([], {UNWRAPPED}) == []
