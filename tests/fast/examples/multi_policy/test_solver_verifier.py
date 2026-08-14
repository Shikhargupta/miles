from argparse import Namespace
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
from examples.multi_policy import solver_verifier
from examples.multi_policy.solver_verifier import _Verdict
from tests.fast.fixtures.megatron_config_fixtures import encode_megatron_config

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.types import Sample

SOLVER_URL = "http://solver-host:1111/generate"
VERIFIER_URL = "http://verifier-host:2222/generate"


@dataclass
class _FakeGenerate:
    responses: dict[str, str]
    calls: list[tuple[str, Sample]] = field(default_factory=list)

    async def __call__(self, input: GenerateFnInput, url: str | None = None) -> GenerateFnOutput:
        sample = input.sample
        self.calls.append((url, sample))
        sample.response = self.responses[url]
        sample.status = Sample.Status.COMPLETED
        return GenerateFnOutput(samples=sample)


def _make_input(*, prompt: str | list[dict[str, str]], label: str) -> GenerateFnInput:
    args = Namespace(
        megatron_config=encode_megatron_config("solver", "verifier"),
        use_critic=False,
        sglang_model_routers={"solver": ("solver-host", 1111), "verifier": ("verifier-host", 2222)},
    )
    sample = Sample(group_index=3, index=7, prompt=prompt, label=label)
    return GenerateFnInput(state=SimpleNamespace(args=args), sample=sample, sampling_params={}, evaluation=False)


@dataclass(frozen=True)
class _RunResult:
    fake: _FakeGenerate
    samples: list[Sample]


async def _run(monkeypatch, *, solver_response: str, verifier_response: str) -> _RunResult:
    fake = _FakeGenerate(responses={SOLVER_URL: solver_response, VERIFIER_URL: verifier_response})
    monkeypatch.setattr(solver_verifier, "single_turn_generate", fake)
    output = await solver_verifier.generate(
        _make_input(prompt=[dict(role="user", content="What is 9 + 9?")], label="#### 18")
    )
    return _RunResult(fake=fake, samples=output.samples)


class TestComputeVerifierReward:
    @pytest.mark.parametrize("verifier_correct", [False, True])
    def test_agreeing_with_a_right_solver_is_the_only_full_credit_case(self, verifier_correct):
        """The solver was right and the verifier said so, so its own answer never enters the score."""
        assert (
            solver_verifier._compute_verifier_reward(
                solver_correct=True, verdict=_Verdict.AGREE, verifier_correct=verifier_correct
            )
            == 1.0
        )

    @pytest.mark.parametrize("verifier_correct", [False, True])
    def test_calling_a_right_solver_wrong_scores_zero(self, verifier_correct):
        """A false accusation is worthless however good the verifier's replacement answer is."""
        assert (
            solver_verifier._compute_verifier_reward(
                solver_correct=True, verdict=_Verdict.WRONG, verifier_correct=verifier_correct
            )
            == 0.0
        )

    @pytest.mark.parametrize("verifier_correct", [False, True])
    def test_agreeing_with_a_wrong_solver_scores_zero(self, verifier_correct):
        """Endorsing a wrong solution is the failure the verifier exists to avoid."""
        assert (
            solver_verifier._compute_verifier_reward(
                solver_correct=False, verdict=_Verdict.AGREE, verifier_correct=verifier_correct
            )
            == 0.0
        )

    def test_catching_a_wrong_solver_without_fixing_it_scores_half(self):
        """Spotting the error is worth partial credit even when the replacement answer is wrong."""
        assert (
            solver_verifier._compute_verifier_reward(
                solver_correct=False, verdict=_Verdict.WRONG, verifier_correct=False
            )
            == 0.5
        )

    def test_catching_a_wrong_solver_and_fixing_it_scores_full(self):
        """Both halves of the verifier's job were done."""
        assert (
            solver_verifier._compute_verifier_reward(
                solver_correct=False, verdict=_Verdict.WRONG, verifier_correct=True
            )
            == 1.0
        )

    @pytest.mark.parametrize("solver_correct", [False, True])
    @pytest.mark.parametrize("verifier_correct", [False, True])
    def test_an_unparseable_verdict_scores_zero(self, solver_correct, verifier_correct):
        """A verdict nobody can read teaches the solver nothing, whatever the verifier meant."""
        assert (
            solver_verifier._compute_verifier_reward(
                solver_correct=solver_correct, verdict=None, verifier_correct=verifier_correct
            )
            == 0.0
        )


class TestParseVerdict:
    def test_a_plain_verdict_word_is_read(self):
        """The happy path the verifier prompt asks for."""
        assert solver_verifier._parse_verdict("The arithmetic checks out. AGREE") is _Verdict.AGREE

    def test_the_last_verdict_word_wins(self):
        """The prompt asks for the verdict at the very end, so earlier mentions are reasoning."""
        assert solver_verifier._parse_verdict("I would AGREE, but the sum is off.\nWRONG\n#### 18") is _Verdict.WRONG

    def test_a_lowercase_verdict_is_not_a_verdict(self):
        """Strict markers keep prose such as 'I agree with the setup' from scoring."""
        assert solver_verifier._parse_verdict("i agree with the solution") is None

    def test_a_longer_word_containing_the_marker_is_not_a_verdict(self):
        """Word boundaries stop 'AGREEMENT' and 'WRONGLY' from being read as verdicts."""
        assert solver_verifier._parse_verdict("There is AGREEMENT that it was WRONGLY set up") is None

    def test_a_response_without_any_marker_is_unparseable(self):
        """An empty or rambling reply has no verdict to score."""
        assert solver_verifier._parse_verdict("") is None


class TestExtractAnswer:
    def test_the_gsm8k_marker_is_read(self):
        """Both the dataset label and the prompted reply end with '#### <answer>'."""
        assert solver_verifier._extract_answer("Half of 36 is 18.\n#### 18") == "18"

    def test_the_last_marker_wins(self):
        """A reply that reconsiders itself is scored on its final answer."""
        assert solver_verifier._extract_answer("#### 17\nOn reflection:\n#### 18") == "18"

    def test_a_marked_answer_is_normalized(self):
        """Currency, thousand separators and a trailing period are formatting, not the answer."""
        assert solver_verifier._extract_answer("#### $1,234.") == "1234"

    def test_a_reply_without_the_marker_falls_back_to_its_last_number(self):
        """The solver prompt comes from the dataset, so it need not ask for the marker."""
        assert solver_verifier._extract_answer("First 9, then 9, so the total is 18") == "18"

    def test_a_reply_without_any_number_has_no_answer(self):
        """Nothing to compare against the ground truth."""
        assert solver_verifier._extract_answer("I cannot tell") is None


class TestGenerate:
    async def test_the_verifier_prompt_quotes_the_question_and_the_solver_answer(self, monkeypatch):
        """The verifier only sees the solver's work through the prompt this function assembles."""
        result = await _run(monkeypatch, solver_response="It is 18.\n#### 18", verifier_response="AGREE")

        verifier_prompt = result.fake.calls[1][1].prompt
        assert isinstance(verifier_prompt, list)
        assert verifier_prompt[0]["role"] == "user"
        assert "What is 9 + 9?" in verifier_prompt[0]["content"]
        assert "It is 18.\n#### 18" in verifier_prompt[0]["content"]

    async def test_a_raw_string_prompt_is_refused(self, monkeypatch):
        """A string prompt may already be chat templated, so quoting it as the question would leak tokens."""
        fake = _FakeGenerate(responses={SOLVER_URL: "#### 18", VERIFIER_URL: "AGREE"})
        monkeypatch.setattr(solver_verifier, "single_turn_generate", fake)

        with pytest.raises(AssertionError, match="chat templated"):
            await solver_verifier.generate(_make_input(prompt="What is 9 + 9?", label="#### 18"))

    async def test_each_policy_is_generated_against_its_own_router(self, monkeypatch):
        """Nothing routes by trainer_model_id, so the generate function picks the url itself."""
        result = await _run(monkeypatch, solver_response="#### 18", verifier_response="AGREE")

        assert [url for url, _ in result.fake.calls] == [SOLVER_URL, VERIFIER_URL]

    async def test_both_samples_are_returned_bound_to_their_own_policy(self, monkeypatch):
        """trainer_model_id is filled on return, and it is what sends each sample to its trainer."""
        result = await _run(monkeypatch, solver_response="#### 18", verifier_response="AGREE")

        solver_sample, verifier_sample = result.samples
        assert solver_sample.trainer_model_id == "solver"
        assert verifier_sample.trainer_model_id == "verifier"

    async def test_a_right_solver_endorsed_by_the_verifier_rewards_both(self, monkeypatch):
        """The end to end path of the full credit row of the reward matrix."""
        result = await _run(monkeypatch, solver_response="#### 18", verifier_response="Checks out. AGREE")

        solver_sample, verifier_sample = result.samples
        assert solver_sample.reward == 1.0
        assert verifier_sample.reward == 1.0

    async def test_a_wrong_solver_corrected_by_the_verifier_rewards_only_the_verifier(self, monkeypatch):
        """The solver is scored against the label, the verifier against what it did about the solver."""
        result = await _run(monkeypatch, solver_response="#### 17", verifier_response="WRONG\n#### 18")

        solver_sample, verifier_sample = result.samples
        assert solver_sample.reward == 0.0
        assert verifier_sample.reward == 1.0

    async def test_a_wrong_solver_caught_but_not_fixed_rewards_half(self, monkeypatch):
        """The verifier's own answer is graded against the same ground truth."""
        result = await _run(monkeypatch, solver_response="#### 17", verifier_response="WRONG\n#### 16")

        assert result.samples[1].reward == 0.5

    async def test_a_run_naming_one_policy_is_refused(self, monkeypatch):
        """This example needs a solver and a verifier, and it must not silently train one of them twice."""
        fake = _FakeGenerate(responses={})
        monkeypatch.setattr(solver_verifier, "single_turn_generate", fake)
        input = _make_input(prompt=[dict(role="user", content="What is 9 + 9?")], label="#### 18")
        input.args.megatron_config = encode_megatron_config("solver")

        with pytest.raises(AssertionError, match="pairs one solver policy with one verifier policy"):
            await solver_verifier.generate(input)
