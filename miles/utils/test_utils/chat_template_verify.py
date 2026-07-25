"""Verify that a chat template satisfies the append-only invariant.

The append-only invariant means: rendering the first N messages (without
generation prompt) produces a string that is an exact prefix of rendering
all messages (with generation prompt).  This is required by sglang's
pretokenized prefix mechanism for agentic workflows.

Core functions are used by both the CLI script
(``scripts/tools/verify_chat_template.py``) and the test suite
(``tests/fast/utils/chat_template_utils/test_pretokenized_chat.py``).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from miles.utils.chat_template_utils.template import apply_chat_template_from_str

if TYPE_CHECKING:
    from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer, TITOTokenizerType


def simulate_pretokenized_path(
    chat_template: str,
    messages: list[dict],
    pretokenized_num_message: int,
    tools: list[dict] | None = None,
    **template_kwargs,
) -> str:
    """Simulate the pretokenized incremental path at text level.

    1. Render first N messages (no generation prompt) -> prefix_text
    2. Render ALL messages (with generation prompt) -> full_text
    3. Verify prefix_text is a prefix of full_text

    Raises ``ValueError`` on prefix mismatch.
    """
    prefix_text = apply_chat_template_from_str(
        chat_template,
        messages[:pretokenized_num_message],
        add_generation_prompt=False,
        tools=tools,
        **template_kwargs,
    )

    full_text = apply_chat_template_from_str(
        chat_template,
        messages,
        add_generation_prompt=True,
        tools=tools,
        **template_kwargs,
    )

    if not full_text.startswith(prefix_text):
        raise ValueError(
            f"Prefix mismatch!\n"
            f"prefix_text ({len(prefix_text)} chars):\n{repr(prefix_text[-200:])}\n\n"
            f"full_text at same position:\n{repr(full_text[:len(prefix_text)][-200:])}"
        )

    return full_text


def get_standard_result(
    chat_template: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    **template_kwargs,
) -> str:
    """Standard path: render all messages with generation prompt."""
    return apply_chat_template_from_str(
        chat_template,
        messages,
        add_generation_prompt=True,
        tools=tools,
        **template_kwargs,
    )


def assert_pretokenized_equals_standard(chat_template, messages, pretokenized_num_message, tools=None, **kwargs):
    """Assert pretokenized incremental path produces same text as standard full render."""
    standard = get_standard_result(chat_template, messages, tools=tools, **kwargs)
    pretokenized = simulate_pretokenized_path(chat_template, messages, pretokenized_num_message, tools=tools, **kwargs)
    assert pretokenized == standard, f"Pretokenized (N={pretokenized_num_message}) != standard"


# ---------------------------------------------------------------------------
# Non-raising verification API for CLI / programmatic use
# ---------------------------------------------------------------------------


@dataclass
class VerifyResult:
    """Result of one accounted append-only verification case.

    ``expected_rejection`` distinguishes a contract-preserving rejection from
    a successful merge. Both have ``passed=True``; unexpected acceptance or
    rejection has ``passed=False``.
    """

    case_name: str
    passed: bool
    error: str | None = None
    expected_rejection: bool = False


def verify_append_only(
    chat_template: str,
    messages: list[dict],
    pretokenized_num_message: int,
    tools: list[dict] | None = None,
    case_name: str = "",
    **template_kwargs,
) -> VerifyResult:
    """Check that the template satisfies the append-only invariant.

    Returns a ``VerifyResult`` instead of raising, making it suitable for
    batch verification in CLI scripts.
    """
    try:
        standard = get_standard_result(chat_template, deepcopy(messages), tools=tools, **template_kwargs)
        pretokenized = simulate_pretokenized_path(
            chat_template, deepcopy(messages), pretokenized_num_message, tools=tools, **template_kwargs
        )
        if pretokenized != standard:
            return VerifyResult(
                case_name=case_name, passed=False, error=f"Pretokenized (N={pretokenized_num_message}) != standard"
            )
        return VerifyResult(case_name=case_name, passed=True)
    except ValueError as e:
        return VerifyResult(case_name=case_name, passed=False, error=str(e))
    except Exception as e:
        return VerifyResult(case_name=case_name, passed=False, error=f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Built-in test cases (shared between CLI and test suite)
# ---------------------------------------------------------------------------
#
# Trajectories expose ``IS_THINKING`` so callers can select render-mode
# variants.  Append roles are derived from each concrete pretokenize boundary:
# copying a trajectory-wide role union would misclassify tool-only cuts inside
# a trajectory that happens to use user or system elsewhere.
#
# Injected-assistant behavior is covered by the fixed-template capability
# matrix and the session lifecycle tests.

import re  # noqa: E402

from miles.utils.test_utils.mock_trajectories import (  # noqa: E402
    IntermediateSystemThinkingTrajectory,
    IntermediateSystemTrajectory,
    LongChainThinkingTrajectory,
    LongChainTrajectory,
    MultiRoleSequenceTrajectory,
    MultiToolSingleTurnTrajectory,
    MultiTurnNoToolThinkingTrajectory,
    MultiTurnNoToolTrajectory,
    MultiTurnThinkingTrajectory,
    MultiTurnTrajectory,
    MultiUserToolChainTrajectory,
    MultiUserTurnThinkingTrajectory,
    ParallelToolsTrajectory,
    RetrySystemTrajectory,
    SimpleNoToolTrajectory,
    SingleToolThinkingTrajectory,
    SingleToolTrajectory,
)


def _short_name(cls: type) -> str:
    name = cls.__name__.replace("Trajectory", "")
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


# Trajectories exercised by ``run_all_checks`` / the CLI.  Must be a subset of
# the classes defined in mock_trajectories.py.  Callers (CLI, tests) pick the
# applicable subset via :func:`select_cases` based on each template's supported
# append roles and thinking mode; there is no global "exclude" list here.
_TRAJECTORIES: list[type] = [
    SingleToolTrajectory,
    MultiTurnTrajectory,
    MultiToolSingleTurnTrajectory,
    ParallelToolsTrajectory,
    LongChainTrajectory,
    MultiUserToolChainTrajectory,
    RetrySystemTrajectory,
    IntermediateSystemTrajectory,
    SimpleNoToolTrajectory,
    MultiTurnNoToolTrajectory,
    SingleToolThinkingTrajectory,
    MultiTurnThinkingTrajectory,
    LongChainThinkingTrajectory,
    MultiUserTurnThinkingTrajectory,
    IntermediateSystemThinkingTrajectory,
    MultiTurnNoToolThinkingTrajectory,
    MultiRoleSequenceTrajectory,
]


@dataclass(frozen=True)
class CaseSpec:
    """One concrete request boundary expanded from a trajectory."""

    case_name: str
    traj_cls: type
    pretokenize_n: int
    append_end: int
    tools: list[dict] | None
    append_roles: frozenset[str]
    is_thinking: bool

    @property
    def has_appendix(self) -> bool:
        return self.append_end > self.pretokenize_n

    @property
    def request_messages(self) -> list[dict]:
        return self.traj_cls.MESSAGES[: self.append_end]

    @property
    def prior_append_roles(self) -> frozenset[str]:
        """Return client roles appended after the initial request and before this boundary."""
        roles: set[str] = set()
        saw_generated_assistant = False
        for message in self.traj_cls.MESSAGES[: self.pretokenize_n]:
            if message["role"] == "assistant":
                saw_generated_assistant = True
            elif saw_generated_assistant:
                roles.add(message["role"])
        return frozenset(roles)


def _expand(traj_cls: type) -> list[CaseSpec]:
    """Expand one trajectory into one CaseSpec per PRETOKENIZE_POSITIONS value."""
    short = _short_name(traj_cls)
    cases = []
    for n in traj_cls.PRETOKENIZE_POSITIONS:
        append_end = n
        while append_end < len(traj_cls.MESSAGES) and traj_cls.MESSAGES[append_end].get("role") != "assistant":
            append_end += 1
        cases.append(
            CaseSpec(
                case_name=f"{short}-N{n}",
                traj_cls=traj_cls,
                pretokenize_n=n,
                append_end=append_end,
                tools=traj_cls.TOOLS,
                append_roles=frozenset(message["role"] for message in traj_cls.MESSAGES[n:append_end]),
                is_thinking=traj_cls.IS_THINKING,
            )
        )
    return cases


ALL_CASES: list[CaseSpec] = [c for t in _TRAJECTORIES for c in _expand(t)]

THINKING_MODES: tuple[str, ...] = ("off", "on", "both")


def select_cases(
    *,
    is_thinking: bool | None = None,
) -> list[CaseSpec]:
    """Select only by render mode; append capabilities never hide a case."""
    if is_thinking is None:
        return list(ALL_CASES)
    return [case for case in ALL_CASES if case.is_thinking == is_thinking]


def enable_thinking_variants(thinking: str) -> list[dict]:
    """Return the list of ``enable_thinking`` kwarg variants to apply per case.

    * ``"off"`` → ``[{}]`` (no ``enable_thinking`` kwarg).
    * ``"on"``  → ``[{"enable_thinking": True}]``.
    * ``"both"`` → ``[{"enable_thinking": True}, {"enable_thinking": False}]``.

    Both CLI (:func:`run_all_checks`) and pytest parametrize callers use this
    to avoid drifting in how the ``enable_thinking`` knob is exercised.
    """
    if thinking == "off":
        return [{}]
    if thinking == "on":
        return [{"enable_thinking": True}]
    if thinking == "both":
        return [{"enable_thinking": True}, {"enable_thinking": False}]
    raise ValueError(f"thinking must be one of {THINKING_MODES}; got {thinking!r}")


def format_case_id(case: CaseSpec, kwargs: dict) -> str:
    """Human-readable label for a ``(case, template_kwargs)`` tuple.

    Used for both CLI ``VerifyResult.case_name`` and pytest test ids so the
    same tuple is identified the same way in both surfaces.  Format:

    * empty kwargs → ``case.case_name``.
    * otherwise → ``<case.case_name>-<k1>_on/off-<k2>=val`` (keys sorted;
      bool values emit ``key_on`` / ``key_off``; other values ``key=val``).
    """
    if not kwargs:
        return case.case_name
    parts: list[str] = []
    for k, v in sorted(kwargs.items()):
        if isinstance(v, bool):
            parts.append(f"{k}_{'on' if v else 'off'}")
        else:
            parts.append(f"{k}={v}")
    return f"{case.case_name}-{'-'.join(parts)}"


def run_all_checks(
    chat_template: str,
    *,
    thinking: str = "off",
    extra_template_kwargs: dict | None = None,
) -> list[VerifyResult]:
    """Run every trajectory case selected by *thinking*.

    ``thinking`` selects which ``enable_thinking`` variants are exercised —
    see :func:`enable_thinking_variants`.  When ``"both"``, **every** selected
    trajectory (thinking or not) is rerun with ``enable_thinking=True`` and
    ``enable_thinking=False``, so templates that branch on the flag are
    validated against non-reasoning input too.

    ``extra_template_kwargs`` is merged into every invocation — use it to
    thread template-specific kwargs (e.g. GLM's ``clear_thinking=False``)
    through the CLI.
    """
    if thinking not in THINKING_MODES:
        raise ValueError(f"thinking must be one of {THINKING_MODES}; got {thinking!r}")
    extra = extra_template_kwargs or {}

    is_thinking_filter = {"off": False, "on": True, "both": None}[thinking]
    selected = select_cases(is_thinking=is_thinking_filter)
    variants = enable_thinking_variants(thinking)

    results: list[VerifyResult] = []
    for case in selected:
        for variant in variants:
            kwargs = {**variant, **extra}
            case_name = format_case_id(case, kwargs)
            if not case.has_appendix:
                results.append(
                    VerifyResult(
                        case_name=case_name,
                        passed=True,
                        error=(
                            f"Invalid request boundary at N={case.pretokenize_n}: "
                            "there are no client-appended messages before the next generated assistant"
                        ),
                        expected_rejection=True,
                    )
                )
                continue
            results.append(
                verify_append_only(
                    chat_template,
                    deepcopy(case.request_messages),
                    case.pretokenize_n,
                    tools=case.tools,
                    case_name=case_name,
                    **kwargs,
                )
            )

    return results


# ---------------------------------------------------------------------------
# TITO-instance verification: decode-roundtrip equality
# ---------------------------------------------------------------------------
#
# The string-based primitive above asserts text-prefix at the chat-template
# layer.  This is necessary but not sufficient for production correctness —
# production runs ``get_tito_tokenizer(...)`` and exercises ``merge_tokens``
# (model-specific token-level boundary patches) plus
# ``tokenize_additional_messages`` (renders the complete appendix under a
# synthetic ``[_DUMMY_SYSTEM, dummy_assistant]`` context, not the real history).
#
# The primitive below mirrors the production path: it instantiates the actual
# TITO subclass + HF tokenizer, runs ``merge_tokens`` against the encoded
# prefix, decodes, and asserts text equality with the canonical full render.


def verify_append_only_via_tito_instance(
    tito: TITOTokenizer,
    tokenizer: Any,
    messages: list[dict],
    pretokenized_num_message: int,
    tools: list[dict] | None = None,
    case_name: str = "",
    **template_kwargs,
) -> VerifyResult:
    """Decode-roundtrip verify with a pre-built TITO instance.

    Asserts ``decode(tito.merge_tokens(prefix_msgs, full_msgs, encode(prefix_text)))
    == full_text`` where ``prefix_text`` and ``full_text`` come from running the
    chat template through ``tokenizer`` with the same kwargs ``tito`` was built
    with.  The test-only path (e.g. ``BuggyQwen3TITOTokenizer``) uses this
    instance form directly; production-shape callers go through
    :func:`verify_append_only_via_tito`.
    """
    try:
        # This shared verifier checks the maximal non-assistant run after boundary N.
        # Production also accepts injected assistant input; dedicated family CPU
        # tests and the session verifier cover that surface.
        n = pretokenized_num_message
        m = n
        while m < len(messages) and messages[m].get("role") != "assistant":
            m += 1
        if m == n:
            return VerifyResult(
                case_name=case_name,
                passed=False,
                error=(
                    f"Empty appendix at N={n}: messages[{n}] is assistant. "
                    "PRETOKENIZE_POSITIONS must land at a post-assistant boundary "
                    "where messages[N:] starts with a non-assistant turn."
                ),
            )

        prefix_msgs = deepcopy(messages[:n])
        full_msgs = deepcopy(messages[:m])

        prefix_text = tito.apply_chat_template(
            prefix_msgs,
            tools=tools,
            add_generation_prompt=False,
        )
        full_text = tito.apply_chat_template(
            full_msgs,
            tools=tools,
            add_generation_prompt=True,
        )

        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        # Simulate production's model-stop: in production, ``pretokenized_token_ids``
        # ends where the model actually stopped — typically before the trailing
        # tokens the chat template would otherwise emit (Qwen's ``\n`` after
        # ``<|im_end|>``, GLM's ambiguous ``<|user|>``/``<|observation|>`` boundary).
        # The TITO subclass declares those as ``trailing_token_ids``.  Trim them
        # here so ``merge_tokens``'s boundary patches see the prefix in its
        # production shape so the verifier sees the same prefix the
        # subclass merge_tokens / trailing trim path operates on.
        trailing = tito.trailing_token_ids
        while prefix_ids and prefix_ids[-1] in trailing:
            prefix_ids = prefix_ids[:-1]
        merged_ids = tito.merge_tokens(prefix_msgs, full_msgs, prefix_ids, tools=tools)
        merged_text = tokenizer.decode(merged_ids)

        if merged_text == full_text:
            return VerifyResult(case_name=case_name, passed=True)

        # Find first divergence and quote ~60 chars of context on each side.
        common_len = min(len(merged_text), len(full_text))
        diff_idx = next(
            (i for i in range(common_len) if merged_text[i] != full_text[i]),
            common_len,
        )
        ctx_start = max(0, diff_idx - 60)
        ctx_end = diff_idx + 60
        return VerifyResult(
            case_name=case_name,
            passed=False,
            error=(
                f"Decode-roundtrip mismatch (N={pretokenized_num_message}) at char {diff_idx}\n"
                f"  expected: ...{full_text[ctx_start:ctx_end]!r}...\n"
                f"  actual:   ...{merged_text[ctx_start:ctx_end]!r}..."
            ),
        )
    except Exception as e:
        return VerifyResult(case_name=case_name, passed=False, error=f"{type(e).__name__}: {e}")


def verify_append_only_via_tito(
    tokenizer: Any,
    tito_model: TITOTokenizerType | str,
    messages: list[dict],
    pretokenized_num_message: int,
    tools: list[dict] | None = None,
    case_name: str = "",
    **template_kwargs,
) -> VerifyResult:
    """Decode-roundtrip verify, building TITO from the registered family.

    Matches the production wiring at ``miles/rollout/session/sessions.py:35`` —
    the same ``get_tito_tokenizer`` factory call, with ``chat_template_kwargs``
    threaded through so ``merge_tokens`` and the dummy-context appendix render
    use the same kwargs as the reference full render.
    """
    from miles.utils.chat_template_utils import get_tito_tokenizer

    tito = get_tito_tokenizer(
        tokenizer,
        tokenizer_type=tito_model,
        chat_template_kwargs=dict(template_kwargs),
    )
    return verify_append_only_via_tito_instance(
        tito,
        tokenizer,
        messages,
        pretokenized_num_message,
        tools=tools,
        case_name=case_name,
        **template_kwargs,
    )


def _verify_expected_role_rejection_via_tito(
    tokenizer: Any,
    tito_model: TITOTokenizerType | str,
    messages: list[dict],
    pretokenized_num_message: int,
    tools: list[dict] | None,
    case_name: str,
    **template_kwargs,
) -> VerifyResult:
    """Exercise the production role gate without rendering an unsupported full prompt."""
    from miles.utils.chat_template_utils import get_tito_tokenizer

    tito = get_tito_tokenizer(
        tokenizer,
        tokenizer_type=tito_model,
        chat_template_kwargs=dict(template_kwargs),
    )
    try:
        tito.tokenize_additional_messages(
            messages[:pretokenized_num_message],
            messages,
            tools,
        )
    except ValueError as error:
        detail = f"{type(error).__name__}: {error}"
        if "appended message" in detail and "allowed=" in detail:
            return VerifyResult(
                case_name=case_name,
                passed=True,
                error=detail,
                expected_rejection=True,
            )
        return VerifyResult(case_name=case_name, passed=False, error=detail)

    return VerifyResult(
        case_name=case_name,
        passed=False,
        error="Expected the FixedTemplate role gate to reject the unsupported appendix, but it passed",
    )


def run_all_checks_via_tito(
    tokenizer: Any,
    tito_model: TITOTokenizerType | str,
    *,
    thinking: str = "off",
    extra_template_kwargs: dict | None = None,
    expected_append_roles: frozenset[str] | None = None,
) -> list[VerifyResult]:
    """Account for every selected case through TITO + tokenizer.

    Per-case TITO rebuild: each (case, ``enable_thinking`` variant) gets a fresh
    TITO instance constructed with the merged kwargs, so the dummy-context
    appendix render inside ``tokenize_additional_messages`` sees the same
    ``enable_thinking`` value as the reference render.  Construction is
    millisecond-level and runs ~50 times per CLI invocation; cheap.

    ``expected_append_roles`` is an oracle input. Tests pass an independent
    manifest so a mistaken production capability cannot validate itself; the
    CLI passes the selected family's live contract for an operational report.
    Unsupported roles must be rejected by the production gate, while malformed
    empty boundaries are classified as explicit expected rejections. No case
    is silently skipped.

    The caller is responsible for setting ``tokenizer.chat_template`` (e.g. via
    ``resolve_fixed_chat_template`` lookup or ``--template`` override) before
    calling this — this function does not consult ``FIXED_TEMPLATE``.
    """
    if thinking not in THINKING_MODES:
        raise ValueError(f"thinking must be one of {THINKING_MODES}; got {thinking!r}")
    extra = extra_template_kwargs or {}

    from miles.utils.chat_template_utils import TITOTokenizerType

    if isinstance(tito_model, str):
        tito_model = TITOTokenizerType(tito_model)
    fixed_template = TITOTokenizerType.get_tokenizer_class(tito_model).FIXED_TEMPLATE
    expected_roles = fixed_template.allowed_append_roles if expected_append_roles is None else expected_append_roles
    is_thinking_filter = {"off": False, "on": True, "both": None}[thinking]
    selected = select_cases(is_thinking=is_thinking_filter)
    variants = enable_thinking_variants(thinking)

    results: list[VerifyResult] = []
    for case in selected:
        for variant in variants:
            kwargs = {**variant, **extra}
            case_name = format_case_id(case, kwargs)
            if not case.has_appendix:
                results.append(
                    VerifyResult(
                        case_name=case_name,
                        passed=True,
                        error=(
                            f"Invalid request boundary at N={case.pretokenize_n}: "
                            "there are no client-appended messages before the next generated assistant"
                        ),
                        expected_rejection=True,
                    )
                )
                continue

            expected_role_rejection = not case.append_roles.issubset(expected_roles)
            if expected_role_rejection:
                results.append(
                    _verify_expected_role_rejection_via_tito(
                        tokenizer,
                        tito_model,
                        deepcopy(case.request_messages),
                        case.pretokenize_n,
                        case.tools,
                        case_name=case_name,
                        **kwargs,
                    )
                )
                continue

            unsupported_prior_roles = case.prior_append_roles - expected_roles
            if unsupported_prior_roles:
                results.append(
                    VerifyResult(
                        case_name=case_name,
                        passed=True,
                        error=(
                            f"Unreachable request boundary at N={case.pretokenize_n}: "
                            f"an earlier appendix used unsupported roles {sorted(unsupported_prior_roles)}"
                        ),
                        expected_rejection=True,
                    )
                )
                continue

            results.append(
                verify_append_only_via_tito(
                    tokenizer,
                    tito_model,
                    deepcopy(case.request_messages),
                    case.pretokenize_n,
                    tools=case.tools,
                    case_name=case_name,
                    **kwargs,
                )
            )

    return results
