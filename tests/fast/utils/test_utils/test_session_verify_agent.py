import asyncio
from copy import deepcopy

from tests.e2e.sglang.test_session_server_multi_role._common import ModelConfig

from miles.utils.chat_template_utils.tito_tokenizer import VALID_APPEND_ROLES
from miles.utils.test_utils import session_verify_agent
from miles.utils.test_utils.session_verify_agent import (
    ASSISTANT_INPUT_FOLLOWUP_TEXT,
    ASSISTANT_INPUT_TEXTS,
    DriverAction,
    _fixed_template_append_roles,
    run_agent,
    select_schedule,
)


def test_all_role_schedule_places_assistant_input_last():
    schedule = select_schedule(VALID_APPEND_ROLES, cycles=2)

    assert schedule[-1] is DriverAction.ASSISTANT_INPUT
    assert schedule.count(DriverAction.ASSISTANT_INPUT) == 1
    assert DriverAction.ROLLBACK in schedule[:-1]


def test_model_config_uses_family_fixed_template_capability():
    cfg = ModelConfig(
        model_name="model",
        reasoning_parser="reasoning",
        tool_call_parser="tool",
        tito_model="qwen3",
    )

    assert _fixed_template_append_roles(cfg.tito_model) == VALID_APPEND_ROLES


def test_assistant_input_appends_two_text_messages_then_user(monkeypatch):
    calls = []

    class FakeAsyncClient:
        def __init__(self, *, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    async def fake_chat(client, base_url, messages, request_kwargs, *, label):
        calls.append(deepcopy(messages))
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": f"generated response {len(calls)}",
                        "tool_calls": [],
                    }
                }
            ]
        }

    monkeypatch.setattr(session_verify_agent.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(session_verify_agent, "_chat", fake_chat)
    monkeypatch.setattr(
        session_verify_agent,
        "select_schedule",
        lambda allowed_roles, *, cycles: [DriverAction.ASSISTANT_INPUT],
    )

    result = asyncio.run(
        run_agent(
            "http://session",
            prompt=None,
            request_kwargs={},
            metadata={"tito_model": "qwen3"},
        )
    )

    assert [message["role"] for message in calls[1]] == [
        "system",
        "user",
        "assistant",
        "assistant",
        "assistant",
        "user",
    ]
    assert [message["content"] for message in calls[1][-3:-1]] == list(ASSISTANT_INPUT_TEXTS)
    assert calls[1][-1] == {"role": "user", "content": ASSISTANT_INPUT_FOLLOWUP_TEXT}
    assert result["driver_events"] == ["initial", "append_assistant"]
    assert result["assistant_input_count"] == 2
    assert result["user_count"] == 1
