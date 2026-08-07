import pytest

from miles.rollout.session.anthropic import (
    AnthropicProtocolError,
    anthropic_to_openai_request,
)


def _request(**overrides: object) -> dict:
    request = {
        "model": "glm-4.7-flash",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "hello"}],
    }
    request.update(overrides)
    return request


def test_request_converts_system_sampling_and_tools() -> None:
    converted = anthropic_to_openai_request(
        _request(
            system=[{"type": "text", "text": "You are concise.", "cache_control": {"type": "ephemeral"}}],
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            stop_sequences=["STOP"],
            stream=True,
            tools=[
                {
                    "name": "bash",
                    "description": "Run a command",
                    "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}},
                }
            ],
            tool_choice={"type": "tool", "name": "bash"},
        )
    )

    assert converted["messages"] == [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "hello"},
    ]
    assert converted["temperature"] == 0.7
    assert converted["top_p"] == 0.9
    assert converted["top_k"] == 40
    assert converted["stop"] == ["STOP"]
    assert converted["stream"] is True
    assert converted["tools"][0]["function"]["name"] == "bash"
    assert converted["tool_choice"] == {"type": "function", "function": {"name": "bash"}}


def test_request_preserves_assistant_thinking_and_tool_use() -> None:
    converted = anthropic_to_openai_request(
        _request(
            messages=[
                {"role": "user", "content": "inspect the repo"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "I should list files."},
                        {"type": "text", "text": "I will inspect it."},
                        {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"command": "ls"}},
                    ],
                },
            ]
        )
    )

    assistant = converted["messages"][1]
    assert assistant["reasoning_content"] == "I should list files."
    assert assistant["content"] == "I will inspect it."
    assert assistant["tool_calls"] == [
        {
            "id": "toolu_1",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command":"ls"}'},
        }
    ]


def test_request_preserves_text_tool_result_text_order() -> None:
    converted = anthropic_to_openai_request(
        _request(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "before"},
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "output"},
                        {"type": "text", "text": "after"},
                    ],
                }
            ]
        )
    )

    assert converted["messages"] == [
        {"role": "user", "content": "before"},
        {"role": "tool", "tool_call_id": "toolu_1", "content": "output"},
        {"role": "user", "content": "after"},
    ]


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"max_tokens": 0}, "max_tokens must be a positive integer"),
        ({"messages": "hello"}, "messages must be an array"),
        ({"messages": [{"role": "user", "content": [{"type": "redacted_thinking", "data": "x"}]}]}, "redacted_thinking history is not supported"),
        ({"tools": [{"name": "web_search", "type": "web_search_20250305"}]}, "Anthropic server-side tools are not supported"),
    ],
)
def test_request_rejects_unrepresentable_inputs(overrides: dict, error: str) -> None:
    with pytest.raises(AnthropicProtocolError, match=error):
        anthropic_to_openai_request(_request(**overrides))
