import asyncio
from types import SimpleNamespace

import pytest

import miles.rollout.generate_hub.agentic_tool_call as agentic_tool_call
from miles.rollout.base_types import GenerateFnInput
from miles.rollout.session.samples.codec import SamplesReply
from miles.rollout.session.v2.metrics import SESSION_ROLLOUT_METRICS_KEY
from miles.utils.types import Sample


class _Tracer:
    session_id = "sid-1"
    session_server_id = "127.0.0.1:12345"
    session_server_instance_id = None
    base_url = "http://127.0.0.1:12345/sessions/sid-1"

    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error
        self.agent_metadata = None

    async def collect_samples(self, input_sample, *, max_seq_len, agent_metadata=None):
        self.agent_metadata = agent_metadata
        if self.error is not None:
            raise self.error
        return self.reply


def _generate_input(**args_kwargs) -> GenerateFnInput:
    args = SimpleNamespace(
        session_server_ip="127.0.0.1",
        session_server_ports=[12345],
        custom_agent_function_path="test.fake_agent",
        max_seq_len=None,
        use_session_server="v2",
        **args_kwargs,
    )
    state = SimpleNamespace(args=args)
    sample = Sample(
        group_index=3,
        index=7,
        prompt=[{"role": "user", "content": "hello"}],
        label="label",
        metadata={"source": "test"},
    )
    return GenerateFnInput(state=state, sample=sample, sampling_params={}, evaluation=False)


async def _fake_agent(**kwargs):
    return {"agent_result": "done"}


def _session_metadata(prefix_cache_info=None):
    return {
        SESSION_ROLLOUT_METRICS_KEY: {
            "session_id": "sid-1",
            "available": True,
            "metrics": {"prefix_cache_info": prefix_cache_info or Sample.PrefixCacheInfo().to_dict()},
        }
    }


def _patch_agent(monkeypatch, tracer):
    async def fake_create(args):
        return tracer

    monkeypatch.setattr(agentic_tool_call.OpenAIEndpointTracer, "create", fake_create)
    monkeypatch.setattr(agentic_tool_call, "load_function", lambda path: _fake_agent)


@pytest.mark.asyncio
async def test_success_returns_list_and_forwards_agent_metadata(monkeypatch):
    sample = Sample(status=Sample.Status.COMPLETED, response="done", response_length=1, tokens=[1])
    tracer = _Tracer(SamplesReply(samples=[sample], session_metadata=_session_metadata(), empty_reason=None))
    _patch_agent(monkeypatch, tracer)

    output = await agentic_tool_call.generate(_generate_input())

    assert output.samples == [sample]
    assert tracer.agent_metadata == {"agent_result": "done"}
    assert output.samples[0].metadata[SESSION_ROLLOUT_METRICS_KEY]["metrics"] == {
        "prefix_cache_info": Sample.PrefixCacheInfo().to_dict()
    }


@pytest.mark.asyncio
async def test_multi_leaf_success_assigns_one_session_metrics_carrier(monkeypatch):
    stale = {"session_id": "stale", "available": True, "metrics": {"agent": "plant"}}
    leaves = [
        Sample(metadata={SESSION_ROLLOUT_METRICS_KEY: stale}),
        Sample(metadata={SESSION_ROLLOUT_METRICS_KEY: stale}),
    ]
    prefix_cache_info = {"cached_tokens": 7, "total_prompt_tokens": 11}
    tracer = _Tracer(
        SamplesReply(
            samples=leaves,
            session_metadata=_session_metadata(prefix_cache_info),
            empty_reason=None,
        )
    )
    _patch_agent(monkeypatch, tracer)

    output = await agentic_tool_call.generate(_generate_input())

    assert [sample.metadata[SESSION_ROLLOUT_METRICS_KEY] for sample in output.samples] == [
        {
            "session_id": "sid-1",
            "available": True,
            "metrics": {"prefix_cache_info": prefix_cache_info},
        },
        {"session_id": "sid-1", "available": True, "metrics": None},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_reason", ["no_records", "all_truncated"])
async def test_empty_reply_returns_aborted_list(monkeypatch, empty_reason):
    prefix_cache_info = {"cached_tokens": 7, "total_prompt_tokens": 11}
    tracer = _Tracer(
        SamplesReply(
            samples=[],
            session_metadata=_session_metadata(prefix_cache_info),
            empty_reason=empty_reason,
        )
    )
    _patch_agent(monkeypatch, tracer)
    generate_input = _generate_input()

    output = await agentic_tool_call.generate(generate_input)

    assert isinstance(output.samples, list)
    assert len(output.samples) == 1
    assert output.samples[0] is not generate_input.sample
    assert output.samples[0].status == Sample.Status.ABORTED
    assert output.samples[0].metadata[SESSION_ROLLOUT_METRICS_KEY]["metrics"] == {
        "prefix_cache_info": prefix_cache_info
    }


@pytest.mark.asyncio
async def test_collection_timeout_marks_session_metrics_unavailable(monkeypatch):
    tracer = _Tracer(error=asyncio.TimeoutError("samples unavailable"))
    _patch_agent(monkeypatch, tracer)

    output = await agentic_tool_call.generate(_generate_input())

    (sample,) = output.samples
    assert sample.status == Sample.Status.ABORTED
    assert sample.metadata[SESSION_ROLLOUT_METRICS_KEY] == {
        "session_id": "sid-1",
        "available": False,
        "metrics": None,
    }


@pytest.mark.asyncio
async def test_v2_rejects_missing_server_session_metrics(monkeypatch):
    tracer = _Tracer(SamplesReply(samples=[Sample()], session_metadata={}, empty_reason=None))
    _patch_agent(monkeypatch, tracer)

    with pytest.raises(ValueError, match="missing session_rollout_metrics"):
        await agentic_tool_call.generate(_generate_input())


@pytest.mark.asyncio
async def test_collection_error_propagates(monkeypatch):
    tracer = _Tracer(error=RuntimeError("samples unavailable"))
    _patch_agent(monkeypatch, tracer)

    with pytest.raises(RuntimeError, match="samples unavailable"):
        await agentic_tool_call.generate(_generate_input())
