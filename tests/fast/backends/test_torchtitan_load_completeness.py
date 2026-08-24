from miles.backends.torchtitan_utils.engine import unloaded_parameters

WRAPPED_BIAS = "layers.0._checkpoint_wrapped_module.moe.expert_bias_E"
BIAS = "layers.0.moe.expert_bias_E"
EXPERT = "layers.0.moe.experts.w1_EFD"


def test_missing_buffer_is_not_a_failure():
    # No HF export contains expert_bias_E; init_states already set it up.
    assert unloaded_parameters([BIAS], {BIAS}) == []


def test_missing_key_fails_closed_when_it_names_no_buffer():
    # The check compares against named_buffers(), not named_parameters(): a
    # fused module's state-dict aliases (FusedQKVLinear presents wq/wk/wv for
    # its wqkv parameter) never appear as parameters, so an allowlist of
    # parameters would silently pass an unloaded attention projection.
    assert unloaded_parameters(["layers.0.attention.wq.weight"], {BIAS}) == ["layers.0.attention.wq.weight"]


def test_wrapped_missing_key_against_unwrapped_buffer_names():
    # Activation checkpointing inserts _checkpoint_wrapped_module on one side
    # only; comparing the two conventions directly would match nothing and turn
    # every missing key into a failure (or none, depending on direction).
    assert unloaded_parameters([WRAPPED_BIAS], {BIAS}) == []


def test_unwrapped_missing_key_against_wrapped_buffer_names():
    assert unloaded_parameters([BIAS], {WRAPPED_BIAS}) == []


def test_moe_shape_separates_expert_params_from_the_bias_buffer():
    missing = [
        "layers.0._checkpoint_wrapped_module.moe.experts.w1_EFD",
        "layers.0._checkpoint_wrapped_module.moe.experts.w2_EDF",
        BIAS,
    ]
    assert unloaded_parameters(missing, {BIAS}) == missing[:2]


def test_nothing_missing():
    assert unloaded_parameters([], {BIAS}) == []
    assert unloaded_parameters([], set()) == []


def test_missing_expert_weight_is_reported_even_with_no_known_buffers():
    assert unloaded_parameters([EXPERT], set()) == [EXPERT]
