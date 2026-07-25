"""Production-shape TITO decode-roundtrip integration tests.

Representative tokenizer families must pass every accounted trajectory case;
the original Qwen3 template and a test-local boundary-bug subclass must fail.
The renderer-only capability matrix covers all registered families.
"""

from copy import deepcopy
from pathlib import Path

import pytest
from tests.ci.ci_register import register_cpu_ci
from transformers import AutoTokenizer

register_cpu_ci(est_time=120, suite="stage-b-cpu", labels=[])


from miles.utils.chat_template_utils import TITOTokenizerType, resolve_fixed_chat_template
from miles.utils.chat_template_utils.tito_tokenizer import FixedTemplate, Qwen3TITOTokenizer
from miles.utils.test_utils.chat_template_verify import (
    ALL_CASES,
    run_all_checks_via_tito,
    verify_append_only_via_tito_instance,
)
from miles.utils.test_utils.mock_trajectories import SingleToolTrajectory

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _setup_tokenizer_with_registered_template(
    model_id: str,
    family: TITOTokenizerType,
):
    """Mirror what production wiring does at startup.

    Loads tokenizer, resolves the family's ``FIXED_TEMPLATE`` (resolution is
    role-independent), and applies the fixed template (if any) onto
    ``tokenizer.chat_template``. Returns ``(tokenizer, extra_kwargs)``.

    A fresh tokenizer instance per call avoids state-mutation hazards from
    overwriting ``chat_template``.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    fixed_path, extra_kwargs = resolve_fixed_chat_template(family)
    if fixed_path is not None:
        with open(fixed_path) as f:
            tokenizer.chat_template = f.read()
    return tokenizer, dict(extra_kwargs)


# ---------------------------------------------------------------------------
# (1) PASS on registered families
# ---------------------------------------------------------------------------


_PASS_PARAMS = [
    pytest.param(TITOTokenizerType.QWEN3, "Qwen/Qwen3-0.6B", 12, id="qwen3"),
    pytest.param(TITOTokenizerType.QWEN35, "Qwen/Qwen3.5-0.8B", 34, id="qwen35"),
    pytest.param(TITOTokenizerType.QWENNEXT, "Qwen/Qwen3-4B-Thinking-2507", 12, id="qwennext"),
    pytest.param(TITOTokenizerType.GLM47, "zai-org/GLM-4.7-Flash", 12, id="glm47"),
]


@pytest.mark.parametrize("family,model_id,expected_rejections", _PASS_PARAMS)
def test_via_tito_pass_on_registered_families(family, model_id, expected_rejections):
    """Representative TITO families round-trip every case without silent skips."""
    tokenizer, extra_kwargs = _setup_tokenizer_with_registered_template(model_id, family)
    results = run_all_checks_via_tito(
        tokenizer,
        family,
        thinking="both",
        extra_template_kwargs=extra_kwargs,
    )
    assert len(results) == len(ALL_CASES) * 2
    assert sum(result.expected_rejection for result in results) == expected_rejections
    failures = [r for r in results if not r.passed]
    assert not failures, (
        f"Expected all accounted outcomes to pass for {family.value} via TITO primitive; "
        f"got {len(failures)} FAIL(s) out of {len(results)}:\n"
        + "\n".join(f"  [{r.case_name}] {r.error}" for r in failures[:5])
    )


# ---------------------------------------------------------------------------
# (1b) PASS on DeepSeek V4 (official-encoder family, local checkpoint only)
# ---------------------------------------------------------------------------


_DEEPSEEK_V4_MODEL = "/cluster-storage/models/deepseek-ai/DeepSeek-V4-Flash"


def test_via_tito_pass_on_deepseek_v4():
    """DSv4's encoder folds contiguous tool/user turns into one ``<｜User｜>``
    block and auto-appends the assistant opener after a user tail; the
    subclass's real-history diff + opener strip must round-trip on both
    registered surfaces (the ``{tool, user}`` cuts land mid-user-block).
    """
    if not Path(_DEEPSEEK_V4_MODEL).exists():
        pytest.skip(f"DeepSeek V4 tokenizer not found: {_DEEPSEEK_V4_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(_DEEPSEEK_V4_MODEL, trust_remote_code=True)
    _fixed_path, extra_kwargs = resolve_fixed_chat_template(TITOTokenizerType.DEEPSEEKV4)
    results = run_all_checks_via_tito(
        tokenizer,
        TITOTokenizerType.DEEPSEEKV4,
        thinking="both",
        extra_template_kwargs=dict(extra_kwargs),
    )
    failures = [r for r in results if not r.passed]
    assert not failures, (
        f"Expected all PASS for deepseekv4 via TITO primitive; "
        f"got {len(failures)} FAIL(s) out of {len(results)}:\n"
        + "\n".join(f"  [{r.case_name}] {r.error}" for r in failures[:5])
    )


# ---------------------------------------------------------------------------
# (2) FAIL on the original unfixed Qwen3 chat template
# ---------------------------------------------------------------------------


def test_via_tito_fail_on_original_qwen3_template():
    """The original Qwen3 chat template uses ``loop.last`` and breaks append-only.

    Bypass ``resolve_fixed_chat_template`` entirely — keep the HF default
    ``tokenizer.chat_template`` and assert the primitive surfaces a FAIL.
    """
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    # Do NOT overwrite tokenizer.chat_template — keep the broken HF default.

    # Cast wide: thinking=both + multi-user-turn surface so trajectories that
    # actually advance ``last_query_index`` between prefix and full are exercised.
    # Those are the ones where ``loop.index0 > ns.last_query_index`` truncation
    # in the original Qwen3 template renders the same assistant turn differently
    # depending on the boundary position.
    results = run_all_checks_via_tito(
        tokenizer,
        TITOTokenizerType.QWEN3,
        thinking="both",
    )
    failures = [r for r in results if not r.passed]
    assert failures, "Expected ≥1 FAIL on the original (unfixed) Qwen3 chat template; got all PASS."


# ---------------------------------------------------------------------------
# (2b) EXPECTED_REJECT on a synthetic restricted production contract
# ---------------------------------------------------------------------------


def test_via_tito_accounts_for_restricted_role_rejections(monkeypatch):
    original = Qwen3TITOTokenizer.FIXED_TEMPLATE
    restricted_roles = frozenset({"tool", "user", "assistant"})
    monkeypatch.setattr(
        Qwen3TITOTokenizer,
        "FIXED_TEMPLATE",
        FixedTemplate(
            template=original.template,
            extra_kwargs=dict(original.extra_kwargs),
            allowed_append_roles=restricted_roles,
        ),
    )
    tokenizer, extra_kwargs = _setup_tokenizer_with_registered_template(
        "Qwen/Qwen3-0.6B",
        TITOTokenizerType.QWEN3,
    )

    results = run_all_checks_via_tito(
        tokenizer,
        TITOTokenizerType.QWEN3,
        thinking="off",
        extra_template_kwargs=extra_kwargs,
        expected_append_roles=restricted_roles,
    )

    role_rejections = [
        result for result in results if result.expected_rejection and result.error and "allowed=" in result.error
    ]
    assert len(results) == 26
    assert role_rejections
    assert all(result.passed for result in results)


# ---------------------------------------------------------------------------
# (3) FAIL on a test-local buggy subclass
# ---------------------------------------------------------------------------


class _BuggyQwen3TITOTokenizer(Qwen3TITOTokenizer):
    """Test-only Qwen3 variant with the ``\\n`` boundary insertion deleted.

    Real ``Qwen3TITOTokenizer.merge_tokens`` appends ``self._newline_id`` after
    a trailing ``<|im_end|>`` because the model stops without emitting the
    newline the chat template would otherwise produce.  This variant skips
    that fixup; the decode-roundtrip primitive is expected to surface it as a
    single-character diff at the prefix-suffix junction.
    """

    def merge_tokens(self, old_messages, new_messages, pretokenized_token_ids, tools=None):
        incremental = self.tokenize_additional_messages(old_messages, new_messages, tools)
        # Intentionally omit the `+\n` insertion — that's the bug we're catching.
        return list(pretokenized_token_ids) + incremental


def test_via_tito_fail_on_buggy_qwen3_subclass():
    """A buggy ``merge_tokens`` produces a junction-level diff that the verifier surfaces."""
    tokenizer, _ = _setup_tokenizer_with_registered_template("Qwen/Qwen3-0.6B", TITOTokenizerType.QWEN3)
    buggy = _BuggyQwen3TITOTokenizer(tokenizer)

    result = verify_append_only_via_tito_instance(
        buggy,
        tokenizer,
        deepcopy(SingleToolTrajectory.MESSAGES),
        pretokenized_num_message=3,
        tools=SingleToolTrajectory.TOOLS,
        case_name="buggy_qwen3-single_tool-N3",
    )
    assert not result.passed, "Expected FAIL on _BuggyQwen3TITOTokenizer (omits the `+\\n` boundary patch); got PASS."
    assert "Decode-roundtrip mismatch" in (
        result.error or ""
    ), f"Expected decode-roundtrip diff in error message; got: {result.error}"
