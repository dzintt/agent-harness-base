from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest

from agent_harness import Agent, AgentConfig, tool
from agent_harness.errors import MaxTurnsExceededError, ProviderError, ToolExecutionError
from agent_harness.providers.base import ConversationItem, ProviderCompletedEvent, ProviderResponse, ProviderTextDeltaEvent


class FakeProvider:
    def __init__(self, responses: list[ProviderResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def create_response(
        self,
        *,
        input_items: Sequence[ConversationItem],
        tools: Sequence[dict[str, Any]],
    ) -> ProviderResponse:
        self.calls.append({"input_items": list(input_items), "tools": list(tools)})
        if not self.responses:
            raise ProviderError("No more fake responses configured.")
        return self.responses.pop(0)

    async def stream_response(
        self,
        *,
        input_items: Sequence[ConversationItem],
        tools: Sequence[dict[str, Any]],
    ) -> AsyncIterator[ProviderTextDeltaEvent | ProviderCompletedEvent]:
        raise NotImplementedError

    async def close(self) -> None:
        return None


@tool
async def ping(message: str) -> str:
    """Echo a message back."""
    return f"pong: {message}"


@tool
async def uppercase(value: str) -> str:
    """Uppercase a value."""
    return value.upper()


@tool
async def explode(message: str) -> str:
    """Always fail."""
    raise ValueError(message)


@pytest.mark.asyncio
async def test_run_without_tools_returns_plain_text() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                output_text="hello world",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "hello world"}],
                    }
                ],
                raw_response={"id": "resp_1"},
            )
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)

    result = await agent.run("Say hello.")

    assert result.output_text == "hello world"
    assert result.tool_results == []


@pytest.mark.asyncio
async def test_run_executes_one_tool_then_returns_final_response() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                output_text="",
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "ping",
                        "arguments": {"message": "hello"},
                        "raw_arguments": '{"message":"hello"}',
                    }
                ],
                output_items=[
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "ping",
                        "arguments": '{"message":"hello"}',
                    }
                ],
                raw_response={"id": "resp_1"},
            ),
            ProviderResponse(
                response_id="resp_2",
                output_text="The tool said pong: hello",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "The tool said pong: hello"}],
                    }
                ],
                raw_response={"id": "resp_2"},
            ),
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), tools=[ping], provider=provider)

    result = await agent.run("Use the tool.")

    assert result.output_text == "The tool said pong: hello"
    assert len(result.tool_results) == 1
    assert result.tool_results[0].output == "pong: hello"
    assert provider.calls[1]["input_items"][-1] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "pong: hello",
    }


@pytest.mark.asyncio
async def test_run_executes_multiple_sequential_tool_calls() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "ping",
                        "arguments": {"message": "hello"},
                        "raw_arguments": '{"message":"hello"}',
                    },
                    {
                        "call_id": "call_2",
                        "name": "uppercase",
                        "arguments": {"value": "world"},
                        "raw_arguments": '{"value":"world"}',
                    },
                ],
                output_items=[
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "ping",
                        "arguments": '{"message":"hello"}',
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_2",
                        "name": "uppercase",
                        "arguments": '{"value":"world"}',
                    },
                ],
                raw_response={"id": "resp_1"},
            ),
            ProviderResponse(
                response_id="resp_2",
                output_text="pong: hello / WORLD",
                output_items=[],
                raw_response={"id": "resp_2"},
            ),
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), tools=[ping, uppercase], provider=provider)

    result = await agent.run("Use both tools.")

    assert [tool_result.output for tool_result in result.tool_results] == ["pong: hello", "WORLD"]
    second_turn_items = provider.calls[1]["input_items"]
    assert second_turn_items[-2:] == [
        {"type": "function_call_output", "call_id": "call_1", "output": "pong: hello"},
        {"type": "function_call_output", "call_id": "call_2", "output": "WORLD"},
    ]


@pytest.mark.asyncio
async def test_max_turns_exceeded_raises_error() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "ping",
                        "arguments": {"message": "again"},
                        "raw_arguments": '{"message":"again"}',
                    }
                ],
                output_items=[],
                raw_response={"id": "resp_1"},
            )
        ]
    )
    agent = Agent(
        config=AgentConfig(model="gpt-5", max_turns=1),
        tools=[ping],
        provider=provider,
    )

    with pytest.raises(MaxTurnsExceededError):
        await agent.run("Loop forever.")


@pytest.mark.asyncio
async def test_tool_errors_surface_as_tool_execution_errors() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "explode",
                        "arguments": {"message": "boom"},
                        "raw_arguments": '{"message":"boom"}',
                    }
                ],
                output_items=[],
                raw_response={"id": "resp_1"},
            )
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), tools=[explode], provider=provider)

    with pytest.raises(ToolExecutionError):
        await agent.run("Fail.")
