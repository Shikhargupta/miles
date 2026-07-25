"""Independent raw-render capability audit for every fixed-template family.

Expected role surfaces and renderer sources are declared here rather than
derived from ``FixedTemplate.allowed_append_roles``.  Every matrix item renders
the old history and the appended history directly, then classifies the result
as PASS or one concrete rejection mode.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from functools import cache
from itertools import product
from typing import Any

import pytest
from tests.ci.ci_register import register_cpu_ci

from miles.utils.chat_template_utils import TEMPLATE_DIR, TITOTokenizerType, deepseek
from miles.utils.chat_template_utils.template import apply_chat_template_from_str, load_hf_chat_template

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])


class RendererKind(Enum):
    BUNDLED_JINJA = "bundled-jinja"
    HF_NATIVE = "hf-native"
    DEEPSEEK_ENCODER = "deepseek-encoder"


class ProbeOutcome(Enum):
    PASS = "PASS"
    REJECT_EXCEPTION = "REJECT_EXCEPTION"
    REJECT_PREFIX = "REJECT_PREFIX"
    REJECT_DROPPED_MESSAGE = "REJECT_DROPPED_MESSAGE"


@dataclass(frozen=True)
class RendererSource:
    kind: RendererKind
    location: str
    registered_kwargs: Mapping[str, Any]
    render_messages: Callable[..., str] | None = None


@dataclass(frozen=True)
class RenderMode:
    name: str
    kwargs: Mapping[str, Any]


@dataclass(frozen=True)
class AppendShape:
    name: str
    messages: tuple[dict[str, Any], ...]

    @property
    def required_roles(self) -> frozenset[str]:
        return frozenset(message["role"] for message in self.messages)

    @property
    def markers(self) -> tuple[str, ...]:
        return tuple(message["content"] for message in self.messages)


@dataclass(frozen=True)
class ProbeResult:
    outcome: ProbeOutcome
    detail: str = ""
    missing_markers: tuple[str, ...] = ()


_ALL_ROLES = frozenset({"tool", "user", "system", "assistant"})
_NO_SYSTEM = frozenset({"tool", "user", "assistant"})

EXPECTED_FIXED_TEMPLATE_CAPABILITIES = {
    TITOTokenizerType.QWEN3: _ALL_ROLES,
    TITOTokenizerType.QWEN35: _NO_SYSTEM,
    TITOTokenizerType.QWENNEXT: _ALL_ROLES,
    TITOTokenizerType.GLM47: _ALL_ROLES,
    TITOTokenizerType.NEMOTRON3: _ALL_ROLES,
    TITOTokenizerType.KIMI25: _ALL_ROLES,
    TITOTokenizerType.KIMI26: _ALL_ROLES,
    TITOTokenizerType.MINIMAX_M25: _NO_SYSTEM,
    TITOTokenizerType.MINIMAX_M27: _NO_SYSTEM,
    TITOTokenizerType.DEEPSEEKV32: _ALL_ROLES,
    TITOTokenizerType.DEEPSEEKV4: _ALL_ROLES,
}

_EXPECTED_UNSUPPORTED_OUTCOMES = {
    TITOTokenizerType.QWEN35: ProbeOutcome.REJECT_EXCEPTION,
    TITOTokenizerType.MINIMAX_M25: ProbeOutcome.REJECT_DROPPED_MESSAGE,
    TITOTokenizerType.MINIMAX_M27: ProbeOutcome.REJECT_DROPPED_MESSAGE,
}

_RENDERER_SOURCES = {
    TITOTokenizerType.QWEN3: RendererSource(
        RendererKind.BUNDLED_JINJA,
        "qwen3_fixed.jinja",
        {"clear_thinking": False},
    ),
    TITOTokenizerType.QWEN35: RendererSource(
        RendererKind.BUNDLED_JINJA,
        "qwen3.5_fixed.jinja",
        {"clear_thinking": False},
    ),
    TITOTokenizerType.QWENNEXT: RendererSource(
        RendererKind.BUNDLED_JINJA,
        "qwen3_thinking_2507_and_next_fixed.jinja",
        {"clear_thinking": False},
    ),
    TITOTokenizerType.GLM47: RendererSource(
        RendererKind.HF_NATIVE,
        "zai-org/GLM-4.7-Flash",
        {"clear_thinking": False},
    ),
    TITOTokenizerType.NEMOTRON3: RendererSource(
        RendererKind.HF_NATIVE,
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
        {"truncate_history_thinking": False},
    ),
    TITOTokenizerType.KIMI25: RendererSource(
        RendererKind.BUNDLED_JINJA,
        "kimi_k25_fixed.jinja",
        {"preserve_thinking": True},
    ),
    TITOTokenizerType.KIMI26: RendererSource(
        RendererKind.HF_NATIVE,
        "moonshotai/Kimi-K2.6",
        {"preserve_thinking": True},
    ),
    TITOTokenizerType.MINIMAX_M25: RendererSource(
        RendererKind.BUNDLED_JINJA,
        "minimax_m25_fixed.jinja",
        {"clear_thinking": False},
    ),
    TITOTokenizerType.MINIMAX_M27: RendererSource(
        RendererKind.BUNDLED_JINJA,
        "minimax_m27_fixed.jinja",
        {"clear_thinking": False},
    ),
    TITOTokenizerType.DEEPSEEKV32: RendererSource(
        RendererKind.DEEPSEEK_ENCODER,
        "miles.utils.chat_template_utils.templates.encoding_dsv32.encode_messages",
        {"drop_thinking": False},
        deepseek.V32.render_messages,
    ),
    TITOTokenizerType.DEEPSEEKV4: RendererSource(
        RendererKind.DEEPSEEK_ENCODER,
        "sglang.srt.entrypoints.openai.encoding_dsv4.encode_messages",
        {"drop_thinking": False},
        deepseek.V4.render_messages,
    ),
}

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "capability_probe",
            "description": "Return a capability marker.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        },
    }
]

_HISTORY = [
    {"role": "system", "content": "BASE_SYSTEM_CAP"},
    {"role": "user", "content": "BASE_USER_CAP"},
    {
        "role": "assistant",
        "content": "",
        "reasoning_content": "BASE_REASONING_CAP",
        "tool_calls": [
            {
                "id": "call_capability_probe",
                "type": "function",
                "function": {
                    "name": "capability_probe",
                    "arguments": '{"value":"base"}',
                },
            }
        ],
    },
]

_SYSTEM_MARKER = "APPENDED_SYSTEM_CAP"

_APPEND_SHAPES = (
    AppendShape(
        "tool",
        (
            {
                "role": "tool",
                "content": "APPENDED_TOOL_CAP",
                "tool_call_id": "call_capability_probe",
            },
        ),
    ),
    AppendShape(
        "user",
        ({"role": "user", "content": "APPENDED_USER_CAP"},),
    ),
    AppendShape(
        "system",
        ({"role": "system", "content": _SYSTEM_MARKER},),
    ),
    AppendShape(
        "assistant",
        (
            {
                "role": "assistant",
                "content": "APPENDED_ASSISTANT_CAP",
                "reasoning_content": "",
            },
        ),
    ),
    AppendShape(
        "tool_user",
        (
            {
                "role": "tool",
                "content": "APPENDED_TOOL_CAP",
                "tool_call_id": "call_capability_probe",
            },
            {"role": "user", "content": "APPENDED_USER_CAP"},
        ),
    ),
    AppendShape(
        "tool_system",
        (
            {
                "role": "tool",
                "content": "APPENDED_TOOL_CAP",
                "tool_call_id": "call_capability_probe",
            },
            {"role": "system", "content": _SYSTEM_MARKER},
        ),
    ),
    AppendShape(
        "system_user",
        (
            {"role": "system", "content": _SYSTEM_MARKER},
            {"role": "user", "content": "APPENDED_USER_CAP"},
        ),
    ),
    AppendShape(
        "assistant_user",
        (
            {
                "role": "assistant",
                "content": "APPENDED_ASSISTANT_CAP",
                "reasoning_content": "",
            },
            {"role": "user", "content": "APPENDED_USER_CAP"},
        ),
    ),
    AppendShape(
        "consecutive_assistant",
        (
            {
                "role": "assistant",
                "content": "APPENDED_ASSISTANT_ONE_CAP",
                "reasoning_content": "",
            },
            {
                "role": "assistant",
                "content": "APPENDED_ASSISTANT_TWO_CAP",
                "reasoning_content": "",
            },
        ),
    ),
    AppendShape(
        "consecutive_assistant_user",
        (
            {
                "role": "assistant",
                "content": "APPENDED_ASSISTANT_ONE_CAP",
                "reasoning_content": "",
            },
            {
                "role": "assistant",
                "content": "APPENDED_ASSISTANT_TWO_CAP",
                "reasoning_content": "",
            },
            {"role": "user", "content": "APPENDED_USER_CAP"},
        ),
    ),
)

_RENDER_MODES = (
    RenderMode("registered", {}),
    RenderMode("thinking_disabled", {"enable_thinking": False}),
)

_NON_DEFAULT_FAMILIES = set(TITOTokenizerType) - {TITOTokenizerType.DEFAULT}
assert set(EXPECTED_FIXED_TEMPLATE_CAPABILITIES) == _NON_DEFAULT_FAMILIES
assert set(_RENDERER_SOURCES) == _NON_DEFAULT_FAMILIES
assert {source.kind for source in _RENDERER_SOURCES.values()} == set(RendererKind)
assert len(_APPEND_SHAPES) == 10
assert len(_RENDER_MODES) == 2
assert all(len(shape.messages) == len(shape.markers) for shape in _APPEND_SHAPES)
assert all(marker.startswith("APPENDED_") for shape in _APPEND_SHAPES for marker in shape.markers)


@cache
def _load_jinja(kind: RendererKind, location: str) -> str:
    if kind is RendererKind.BUNDLED_JINJA:
        return (TEMPLATE_DIR / location).read_text(encoding="utf-8")
    if kind is RendererKind.HF_NATIVE:
        return load_hf_chat_template(location)
    raise AssertionError(f"{kind.value} does not identify a Jinja template")


def _render(
    family: TITOTokenizerType,
    messages: list[dict[str, Any]],
    mode: RenderMode,
    *,
    add_generation_prompt: bool,
) -> str:
    source = _RENDERER_SOURCES[family]
    kwargs = dict(source.registered_kwargs)
    kwargs.update(mode.kwargs)

    if source.kind in {RendererKind.BUNDLED_JINJA, RendererKind.HF_NATIVE}:
        return apply_chat_template_from_str(
            _load_jinja(source.kind, source.location),
            messages,
            add_generation_prompt=add_generation_prompt,
            tools=_TOOLS,
            **kwargs,
        )

    assert source.kind is RendererKind.DEEPSEEK_ENCODER
    assert source.render_messages is not None
    return source.render_messages(
        messages,
        add_generation_prompt=add_generation_prompt,
        tools=_TOOLS,
        **kwargs,
    )


def _probe(family: TITOTokenizerType, shape: AppendShape, mode: RenderMode) -> ProbeResult:
    try:
        before = _render(
            family,
            list(_HISTORY),
            mode,
            add_generation_prompt=False,
        )
        after = _render(
            family,
            [*_HISTORY, *shape.messages],
            mode,
            add_generation_prompt=True,
        )
    except Exception as error:
        return ProbeResult(
            ProbeOutcome.REJECT_EXCEPTION,
            detail=f"{type(error).__name__}: {error}",
        )

    if not after.startswith(before):
        return ProbeResult(
            ProbeOutcome.REJECT_PREFIX,
            detail=f"before_len={len(before)}, after_len={len(after)}",
        )

    missing_markers = tuple(marker for marker in shape.markers if marker not in after)
    if missing_markers:
        return ProbeResult(
            ProbeOutcome.REJECT_DROPPED_MESSAGE,
            detail=f"missing_markers={missing_markers!r}",
            missing_markers=missing_markers,
        )

    return ProbeResult(ProbeOutcome.PASS)


def _expected_outcome(family: TITOTokenizerType, shape: AppendShape) -> ProbeOutcome:
    if shape.required_roles <= EXPECTED_FIXED_TEMPLATE_CAPABILITIES[family]:
        return ProbeOutcome.PASS
    return _EXPECTED_UNSUPPORTED_OUTCOMES[family]


_AUDIT_PARAMS = tuple(
    product(
        EXPECTED_FIXED_TEMPLATE_CAPABILITIES,
        _APPEND_SHAPES,
        _RENDER_MODES,
    )
)
assert len(_AUDIT_PARAMS) == 11 * 10 * 2 == 220
assert Counter(_expected_outcome(family, shape) for family, shape, _mode in _AUDIT_PARAMS) == {
    ProbeOutcome.PASS: 202,
    ProbeOutcome.REJECT_EXCEPTION: 6,
    ProbeOutcome.REJECT_DROPPED_MESSAGE: 12,
}


@pytest.mark.parametrize(
    "family,shape,mode",
    _AUDIT_PARAMS,
    ids=[
        f"{family.value}-{shape.name}-{mode.name}-{_expected_outcome(family, shape).value}"
        for family, shape, mode in _AUDIT_PARAMS
    ],
)
def test_fixed_template_raw_render_capability(
    family: TITOTokenizerType,
    shape: AppendShape,
    mode: RenderMode,
) -> None:
    registered_roles = TITOTokenizerType.get_tokenizer_class(family).FIXED_TEMPLATE.allowed_append_roles
    assert registered_roles == EXPECTED_FIXED_TEMPLATE_CAPABILITIES[family]

    expected = _expected_outcome(family, shape)
    actual = _probe(family, shape, mode)

    assert actual.outcome is expected, (
        f"{family.value}/{shape.name}/{mode.name} via {_RENDERER_SOURCES[family].location}: "
        f"expected {expected.value}, got {actual.outcome.value}; "
        f"{actual.detail}"
    )
    if expected is ProbeOutcome.REJECT_DROPPED_MESSAGE:
        assert actual.missing_markers == (_SYSTEM_MARKER,)
    elif expected is ProbeOutcome.REJECT_EXCEPTION:
        assert "System message must be at the beginning." in actual.detail
    else:
        assert actual.missing_markers == ()
