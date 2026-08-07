import json
from unittest.mock import patch

# ruff: noqa: F811 -- imported pytest fixture names are injected as test arguments.

import requests
from fastapi.responses import JSONResponse
from tests.fast.router.test_sessions import router_env  # noqa: F401

from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, ProcessResult


def _create_session(url: str) -> str:
    return requests.post(f"{url}/sessions", timeout=5.0).json()["session_id"]


def _post_messages(url: str, session_id: str, payload: dict) -> requests.Response:
    return requests.post(f"{url}/sessions/{session_id}/v1/messages", json=payload, timeout=10.0)


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for frame in body.strip().split("\n\n"):
        event_line, data_line = frame.splitlines()
        events.append((event_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: "))))
    return events


def _request(**overrides: object) -> dict:
    request = {
        "model": "mock-model",
        "max_tokens": 512,
        "messages": [{"role": "user", "content": "hello"}],
    }
    request.update(overrides)
    return request


class TestAnthropicSessionRoute:
    def test_non_streaming_request_is_recorded_in_openai_form(self, router_env) -> None:
        session_id = _create_session(router_env.url)
        response = _post_messages(
            router_env.url,
            session_id,
            _request(
                system="Be concise.",
                temperature=0.7,
                top_p=0.9,
                top_k=20,
            ),
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        body = response.json()
        assert body["type"] == "message"
        assert body["role"] == "assistant"
        assert body["content"][0]["type"] == "text"

        records = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0).json()["records"]
        assert len(records) == 1
        record = records[0]
        assert record["path"] == "/v1/chat/completions"
        assert record["request"]["messages"][:2] == [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "hello"},
        ]
        assert record["request"]["temperature"] == 0.7
        assert record["request"]["top_p"] == 0.9
        assert record["request"]["top_k"] == 20
        assert record["request"]["logprobs"] is True
        assert record["request"]["return_meta_info"] is True
        assert "input_ids" in record["request"]
        assert "stream" not in record["request"]

    def test_streaming_response_has_anthropic_events_and_one_record(self, router_env) -> None:
        session_id = _create_session(router_env.url)
        response = _post_messages(router_env.url, session_id, _request(stream=True))

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _parse_sse(response.text)
        assert [event_type for event_type, _ in events] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        assert events[2][1]["delta"]["type"] == "text_delta"
        assert events[-2][1]["delta"]["stop_reason"] == "end_turn"

        records = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0).json()["records"]
        assert len(records) == 1
        assert "stream" not in router_env.backend.request_log[-1]

    def test_multi_turn_response_replay_passes_tito_prefix_check(self, router_env) -> None:
        session_id = _create_session(router_env.url)
        first_request = _request()
        first = _post_messages(router_env.url, session_id, first_request)
        assert first.status_code == 200

        second_messages = [
            *first_request["messages"],
            {"role": "assistant", "content": first.json()["content"]},
            {"role": "user", "content": "continue"},
        ]
        second = _post_messages(router_env.url, session_id, _request(messages=second_messages))

        assert second.status_code == 200
        records = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0).json()["records"]
        assert len(records) == 2

    def test_tool_use_and_result_round_trip_passes_tito_prefix_check(self, router_env) -> None:
        session_id = _create_session(router_env.url)
        tools = [
            {
                "name": "bash",
                "description": "Run a command",
                "input_schema": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            }
        ]

        def tool_call_process_fn(prompt: str) -> ProcessResult:
            return ProcessResult(text='<tool_call>\n{"name":"bash","arguments":{"command":"pwd"}}\n</tool_call>')

        first_request = _request(messages=[{"role": "user", "content": "show the directory"}], tools=tools)
        with patch.object(router_env.backend, "process_fn", new=tool_call_process_fn):
            first = _post_messages(router_env.url, session_id, first_request)
        assert first.status_code == 200
        [tool_use] = [block for block in first.json()["content"] if block["type"] == "tool_use"]

        second_messages = [
            *first_request["messages"],
            {"role": "assistant", "content": first.json()["content"]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_use["id"], "content": "/workspace"}],
            },
        ]
        with patch.object(router_env.backend, "process_fn", new=lambda prompt: ProcessResult(text="done")):
            second = _post_messages(
                router_env.url,
                session_id,
                _request(messages=second_messages, tools=tools),
            )

        assert second.status_code == 200
        records = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0).json()["records"]
        assert len(records) == 2
        assert records[1]["request"]["messages"][-1] == {
            "role": "tool",
            "tool_call_id": tool_use["id"],
            "content": "/workspace",
        }

    def test_invalid_request_uses_anthropic_error_envelope(self, router_env) -> None:
        session_id = _create_session(router_env.url)
        response = _post_messages(router_env.url, session_id, _request(max_tokens=0))

        assert response.status_code == 400
        assert response.json() == {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "max_tokens must be a positive integer"},
        }

    def test_upstream_error_uses_anthropic_envelope_and_is_not_recorded(self, router_env) -> None:
        session_id = _create_session(router_env.url)

        async def reject(self: MockSGLangServer, request: object, compute_fn: object) -> JSONResponse:
            return JSONResponse(content={"error": {"message": "context too long"}}, status_code=400)

        with patch.object(MockSGLangServer, "_handle_generate_like_request", new=reject):
            response = _post_messages(router_env.url, session_id, _request())

        assert response.status_code == 400
        assert response.json() == {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "context too long"},
        }
        records = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0).json()["records"]
        assert records == []
