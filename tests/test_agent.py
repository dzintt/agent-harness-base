from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
from pydantic import BaseModel

from agent_harness import Agent, AgentConfig, ChatMessage, tool
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
        response_model: type[BaseModel] | None = None,
    ) -> ProviderResponse:
        self.calls.append(
            {
                "input_items": list(input_items),
                "tools": list(tools),
                "response_model": response_model,
            }
        )
        if not self.responses:
            raise ProviderError("No more fake responses configured.")
        return self.responses.pop(0)

    async def stream_response(
        self,
        *,
        input_items: Sequence[ConversationItem],
        tools: Sequence[dict[str, Any]],
        response_model: type[BaseModel] | None = None,
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


class Person(BaseModel):
    name: str
    age: int


class WeatherAnswer(BaseModel):
    city: str
    temperature_f: int
    summary: str


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
    assert result.output_data is None
    assert result.tool_results == []


@pytest.mark.asyncio
async def test_run_accepts_multiple_messages() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                output_text="I remember the earlier messages.",
                output_items=[],
                raw_response={"id": "resp_1"},
            )
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)

    result = await agent.run(
        [
            ChatMessage(role="system", content="You are concise."),
            ChatMessage(role="user", content="My name is Anson."),
            ChatMessage(role="assistant", content="Noted."),
            ChatMessage(role="user", content="What's my name?"),
        ]
    )

    assert result.output_text == "I remember the earlier messages."
    assert provider.calls[0]["input_items"] == [
        {
            "type": "message",
            "role": "system",
            "content": "You are concise.",
        },
        {
            "type": "message",
            "role": "user",
            "content": "My name is Anson.",
        },
        {
            "type": "message",
            "role": "assistant",
            "content": "Noted.",
        },
        {
            "type": "message",
            "role": "user",
            "content": "What's my name?",
        },
    ]


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


@pytest.mark.asyncio
async def test_run_returns_structured_output_without_tools() -> None:
    person = Person(name="Sarah", age=29)
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                output_text='{"name":"Sarah","age":29}',
                output_data=person,
                output_items=[],
                raw_response={"id": "resp_1"},
            )
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)

    result = await agent.run(
        "Extract the person from: Sarah is 29 years old.",
        response_model=Person,
    )

    assert result.output_data == person
    assert result.output_text == '{"name":"Sarah","age":29}'
    assert provider.calls[0]["response_model"] is Person


@pytest.mark.asyncio
async def test_run_returns_structured_output_after_tool_call() -> None:
    weather = WeatherAnswer(city="San Francisco", temperature_f=65, summary="Foggy")
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                tool_calls=[
                    {
                        "call_id": "call_1",
                        "name": "ping",
                        "arguments": {"message": "weather"},
                        "raw_arguments": '{"message":"weather"}',
                    }
                ],
                output_items=[
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "ping",
                        "arguments": '{"message":"weather"}',
                    }
                ],
                raw_response={"id": "resp_1"},
            ),
            ProviderResponse(
                response_id="resp_2",
                output_text='{"city":"San Francisco","temperature_f":65,"summary":"Foggy"}',
                output_data=weather,
                output_items=[],
                raw_response={"id": "resp_2"},
            ),
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), tools=[ping], provider=provider)

    result = await agent.run(
        "Use the tool and return a structured weather answer.",
        response_model=WeatherAnswer,
    )

    assert result.output_data == weather
    assert [tool_result.output for tool_result in result.tool_results] == ["pong: weather"]
    assert provider.calls[0]["response_model"] is WeatherAnswer
    assert provider.calls[1]["response_model"] is WeatherAnswer


@pytest.mark.asyncio
async def test_run_surfaces_structured_provider_failures() -> None:
    class FailingStructuredProvider(FakeProvider):
        async def create_response(
            self,
            *,
            input_items: Sequence[ConversationItem],
            tools: Sequence[dict[str, Any]],
            response_model: type[BaseModel] | None = None,
        ) -> ProviderResponse:
            raise ProviderError("structured parse failed")

    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        provider=FailingStructuredProvider([]),
    )

    with pytest.raises(ProviderError, match="structured parse failed"):
        await agent.run("Return structured data.", response_model=Person)


@pytest.mark.asyncio
async def test_chat_session_preserves_conversation_history() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                response_id="resp_1",
                output_text="Your name is Anson.",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Your name is Anson."}],
                    }
                ],
                raw_response={"id": "resp_1"},
            ),
            ProviderResponse(
                response_id="resp_2",
                output_text="You told me your name is Anson.",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "You told me your name is Anson."}],
                    }
                ],
                raw_response={"id": "resp_2"},
            ),
        ]
    )
    agent = Agent(config=AgentConfig(model="gpt-5"), provider=provider)
    chat = agent.chat()

    first = await chat.run("My name is Anson.")
    second = await chat.run("What name did I give you?")

    assert first.output_text == "Your name is Anson."
    assert second.output_text == "You told me your name is Anson."
    assert provider.calls[1]["input_items"] == [
        {
            "type": "message",
            "role": "user",
            "content": "My name is Anson.",
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Your name is Anson."}],
        },
        {
            "type": "message",
            "role": "user",
            "content": "What name did I give you?",
        },
    ]
    assert chat.history == [
        ChatMessage(role="user", content="My name is Anson."),
        ChatMessage(role="assistant", content="Your name is Anson."),
        ChatMessage(role="user", content="What name did I give you?"),
        ChatMessage(role="assistant", content="You told me your name is Anson."),
    ]
