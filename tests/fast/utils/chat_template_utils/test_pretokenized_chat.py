"""
Unit tests for the pretokenized chat completion path.

Tests that using pretokenized_token_ids + pretokenized_num_message produces
identical token IDs as the standard apply_chat_template path.

Ported from sglang test/unit/test_pretokenized_chat.py.
"""

from copy import deepcopy

import pytest

from miles.utils.chat_template_utils import TITOTokenizerType, resolve_fixed_chat_template
from miles.utils.chat_template_utils.template import load_hf_chat_template
from miles.utils.test_utils.chat_template_verify import (
    ALL_CASES,
    CaseSpec,
    assert_pretokenized_equals_standard,
    enable_thinking_variants,
    format_case_id,
    select_cases,
    simulate_pretokenized_path,
)
from miles.utils.test_utils.mock_trajectories import (
    MultiTurnTrajectory,
    MultiUserTurnThinkingTrajectory,
    SingleToolTrajectory,
    last_user_index,
)


def _load_fixed(tito_model: TITOTokenizerType) -> str:
    path, _kwargs = resolve_fixed_chat_template(tito_model)
    assert path is not None, f"resolve_fixed_chat_template should resolve {tito_model.value}"
    with open(path) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Registered fixed-template matrix
# ---------------------------------------------------------------------------
#
# Each entry is one production renderer contract:
# (name, content, supports_thinking, extra_template_kwargs).
# Append capabilities do not filter this broad prefix-regression corpus.
# Unsupported raw renders remain parametrized as explicit expected rejections.

_TEMPLATES: list[tuple[str, str, bool, dict]] = [
    (
        "qwen3_fixed",
        _load_fixed(TITOTokenizerType.QWEN3),
        True,
        {"clear_thinking": False},
    ),
    (
        "qwen3.5_fixed",
        _load_fixed(TITOTokenizerType.QWEN35),
        True,
        {"clear_thinking": False},
    ),
    (
        "qwen3_next_thinking_fixed",
        _load_fixed(TITOTokenizerType.QWENNEXT),
        True,
        {"clear_thinking": False},
    ),
    (
        "glm47_flash",
        load_hf_chat_template("zai-org/GLM-4.7-Flash"),
        True,
        {"clear_thinking": False},
    ),
    (
        "kimi_k25_fixed",
        _load_fixed(TITOTokenizerType.KIMI25),
        True,
        {"preserve_thinking": True},
    ),
    (
        "kimi_k26",
        load_hf_chat_template("moonshotai/Kimi-K2.6"),
        True,
        {"preserve_thinking": True},
    ),
    (
        "minimax_m25_fixed",
        _load_fixed(TITOTokenizerType.MINIMAX_M25),
        True,
        {"clear_thinking": False},
    ),
    (
        "minimax_m27_fixed",
        _load_fixed(TITOTokenizerType.MINIMAX_M27),
        True,
        {"clear_thinking": False},
    ),
    (
        "nemotron3",
        load_hf_chat_template("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"),
        True,
        {"truncate_history_thinking": False},
    ),
]

# Original (unfixed) HF templates referenced by negative tests
_ORIGINAL_TEMPLATES = {
    "qwen3_original": load_hf_chat_template("Qwen/Qwen3-0.6B"),
    "qwen3_thinking_2507": load_hf_chat_template("Qwen/Qwen3-4B-Thinking-2507"),
    "qwen3_next_thinking": load_hf_chat_template("Qwen/Qwen3-Next-80B-A3B-Thinking"),
}


def test_case_append_roles_are_derived_from_each_request_boundary():
    by_name = {case.case_name: case for case in ALL_CASES}

    assert by_name["multi_user_tool_chain-N3"].append_roles == frozenset({"tool"})
    assert by_name["multi_user_tool_chain-N7"].append_roles == frozenset({"tool"})
    assert by_name["retry_system-N6"].append_roles == frozenset({"tool"})
    assert by_name["retry_system-N6"].prior_append_roles == frozenset({"tool", "system"})
    assert by_name["multi_role_sequence-N4"].append_roles == frozenset({"user"})
    assert sum(not case.has_appendix for case in ALL_CASES) == 6


def _build_pretokenized_params():
    params = []
    for name, content, supports_thinking, extra_kwargs in _TEMPLATES:
        cases = select_cases(is_thinking=None if supports_thinking else False)
        variants = enable_thinking_variants("both" if supports_thinking else "off")
        for case in cases:
            for variant in variants:
                kwargs = {**variant, **extra_kwargs}
                ident = f"{name}-{format_case_id(case, kwargs)}"
                expected_system_rejection = name == "qwen3.5_fixed" and any(
                    message["role"] == "system" for message in case.request_messages[1:]
                )
                if expected_system_rejection:
                    ident += "-EXPECTED_REJECT"
                params.append(pytest.param(content, case, kwargs, expected_system_rejection, id=ident))
    return params


# ===========================================================================
# Core tests: every (template, case, kwargs) tuple satisfies append-only
# ===========================================================================


@pytest.mark.parametrize(
    "chat_template, case, kwargs, expected_system_rejection",
    _build_pretokenized_params(),
)
def test_pretokenized(
    chat_template: str,
    case: CaseSpec,
    kwargs: dict,
    expected_system_rejection: bool,
):
    if not case.has_appendix:
        assert case.append_roles == frozenset()
        assert case.append_end == case.pretokenize_n
        return

    if expected_system_rejection:
        with pytest.raises(ValueError, match="System message must be at the beginning"):
            assert_pretokenized_equals_standard(
                chat_template=chat_template,
                messages=deepcopy(case.request_messages),
                pretokenized_num_message=case.pretokenize_n,
                tools=case.tools,
                **kwargs,
            )
        return

    assert_pretokenized_equals_standard(
        chat_template=chat_template,
        messages=deepcopy(case.request_messages),
        pretokenized_num_message=case.pretokenize_n,
        tools=case.tools,
        **kwargs,
    )


# ===========================================================================
# Negative tests: original (unfixed) templates fail prefix invariant
# ===========================================================================

# (chat_template, trajectory_cls, pretokenize_n)
_MISMATCH_CASES = [
    pytest.param(_ORIGINAL_TEMPLATES["qwen3_original"], SingleToolTrajectory, 3, id="qwen3_original-single_tool"),
    pytest.param(_ORIGINAL_TEMPLATES["qwen3_original"], MultiTurnTrajectory, 3, id="qwen3_original-multi_turn"),
    pytest.param(
        _ORIGINAL_TEMPLATES["qwen3_thinking_2507"], SingleToolTrajectory, 3, id="qwen3_thinking_2507-single_tool"
    ),
    pytest.param(
        _ORIGINAL_TEMPLATES["qwen3_next_thinking"], SingleToolTrajectory, 3, id="qwen3_next_thinking-single_tool"
    ),
    pytest.param(
        _ORIGINAL_TEMPLATES["qwen3_next_thinking"], MultiTurnTrajectory, 3, id="qwen3_next_thinking-multi_turn"
    ),
]


@pytest.mark.parametrize("chat_template,trajectory_cls,pretokenize_n", _MISMATCH_CASES)
def test_original_template_prefix_mismatch(chat_template, trajectory_cls, pretokenize_n):
    """Original templates with loop.last cause prefix mismatch (our fix resolves this)."""
    with pytest.raises(ValueError, match="Prefix mismatch"):
        simulate_pretokenized_path(
            chat_template,
            deepcopy(trajectory_cls.MESSAGES),
            pretokenize_n,
            tools=trajectory_cls.TOOLS,
        )


# ===========================================================================
# Negative test: cross-user-turn thinking compression breaks prefix invariant
# ===========================================================================

# Pretokenizing BEFORE the last user turn in a multi-user-turn thinking
# trajectory fails because templates compress reasoning_content from earlier
# turns.  This is a known template limitation, not a bug in the fixed templates.
_CROSS_USER_THINKING_N = last_user_index(MultiUserTurnThinkingTrajectory.MESSAGES)


def _unique_thinking_templates():
    return [pytest.param(template, id=name) for name, template in _ORIGINAL_TEMPLATES.items()]


@pytest.mark.parametrize("chat_template", _unique_thinking_templates())
@pytest.mark.parametrize("enable_thinking", [True, False], ids=["thinking_on", "thinking_off"])
def test_cross_user_turn_thinking_prefix_mismatch(chat_template, enable_thinking):
    """Thinking templates compress reasoning_content from earlier user turns, breaking prefix invariant."""
    with pytest.raises(ValueError, match="Prefix mismatch"):
        simulate_pretokenized_path(
            chat_template,
            deepcopy(MultiUserTurnThinkingTrajectory.MESSAGES),
            _CROSS_USER_THINKING_N,
            tools=MultiUserTurnThinkingTrajectory.TOOLS,
            enable_thinking=enable_thinking,
        )
