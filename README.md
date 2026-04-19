## Agent Harness Base

Small async-first base for building agents with the OpenAI Python SDK.

The package exposes a minimal public API:

- `Agent`
- `AgentConfig`
- `ToolRegistry`
- `tool`

It is designed for:

- async execution with `AsyncOpenAI`
- simple decorator-based tool definitions
- sequential local tool execution
- streaming text deltas through an async iterator
- trivial structured outputs with Pydantic models

## What This Gives You

This project is a small reusable base for future agent projects.

You can:

- create an `Agent`
- register async tools with `@tool`
- call `await agent.run(...)` for a normal request
- call `await agent.run(..., response_model=MySchema)` for structured output
- call `agent.stream(...)` for incremental text and a final structured result

The goal is to keep the public API small while still covering the common paths you will use repeatedly in future projects.

## Install

```bash
uv sync
```

Set your API key:

```bash
$env:OPENAI_API_KEY="your_key_here"
```

Run any example with:

```bash
uv run python examples/basic_agent.py
```

## Quickstart

```python
import asyncio

from agent_harness import Agent, AgentConfig, tool


@tool
async def ping(message: str) -> str:
    """Echo a message back."""
    return f"pong: {message}"


async def main() -> None:
    agent = Agent(
        config=AgentConfig(model="gpt-5"),
        tools=[ping],
    )

    result = await agent.run("Call ping with hello and tell me the result.")
    print(result.output_text)


asyncio.run(main())
```

The result object includes:

- `result.output_text`: the final assistant text
- `result.output_data`: validated structured output when you pass `response_model=...`
- `result.tool_results`: tool execution history
- `result.raw_responses`: raw-ish provider payloads for debugging

## Examples

Available examples:

- [basic_agent.py](/C:/Users/Anson/Desktop/agent-harness-base/examples/basic_agent.py): smallest possible agent with one tool
- [structured_output.py](/C:/Users/Anson/Desktop/agent-harness-base/examples/structured_output.py): extract typed data with a Pydantic schema
- [structured_with_tools.py](/C:/Users/Anson/Desktop/agent-harness-base/examples/structured_with_tools.py): combine tools and structured outputs
- [streaming.py](/C:/Users/Anson/Desktop/agent-harness-base/examples/streaming.py): stream text deltas and read the final result
- [examples/README.md](/C:/Users/Anson/Desktop/agent-harness-base/examples/README.md): example index and when to use each one

## Structured Outputs

Pass a Pydantic model directly to `run()` and read the validated result from `output_data`:

```python
from pydantic import BaseModel
from agent_harness import Agent, AgentConfig


class Person(BaseModel):
    name: str
    age: int


agent = Agent(config=AgentConfig(model="gpt-5"))

result = await agent.run(
    "Extract the person from: Sarah is 29 years old.",
    response_model=Person,
)

print(result.output_data)
```

This is the trivial path the harness is optimized for:

```python
result = await agent.run("...", response_model=MySchema)
data = result.output_data
```

Structured outputs also work alongside tools:

```python
from pydantic import BaseModel
from agent_harness import Agent, AgentConfig, tool


class WeatherAnswer(BaseModel):
    city: str
    temperature_f: int
    summary: str


@tool
async def get_weather(city: str) -> str:
    """Return fake weather data."""
    return '{"city":"San Francisco","temperature_f":65,"summary":"Foggy"}'


agent = Agent(config=AgentConfig(model="gpt-5"), tools=[get_weather])

result = await agent.run(
    "Use the weather tool and return a structured answer.",
    response_model=WeatherAnswer,
)

print(result.output_data)
```

When you pass `response_model=...`:

- the harness uses the OpenAI SDK structured output path
- the final parsed value is returned as `result.output_data`
- the normal text result still remains available as `result.output_text`

## Streaming

Use `run()` when you want a single final result object.

Use `stream()` when you want incremental text and tool lifecycle events:

```python
async for event in agent.stream("Say hi"):
    if event.type == "text_delta":
        print(event.delta, end="")
```

Structured streaming exposes the parsed schema on the final completed event:

```python
async for event in agent.stream("Summarize this text.", response_model=Summary):
    if event.type == "text_delta":
        print(event.delta, end="")
    elif event.type == "completed":
        print(event.result.output_data)
```

Streaming event types:

- `text_delta`: incremental assistant text
- `tool_call_started`: a tool call is about to run
- `tool_call_completed`: a tool finished running
- `completed`: final `AgentRunResult`
- `error`: the stream failed

Structured output in streaming mode is exposed only on the final `completed` event:

- you still receive text deltas as normal
- `event.result.output_data` is populated only after the response is complete and validated

## Tools

Tools are async functions decorated with `@tool`:

```python
from agent_harness import tool


@tool
async def lookup_user(user_id: int) -> str:
    """Return a serialized user record."""
    return '{"id": 1, "name": "Ada"}'
```

Tool rules:

- tools must use `async def`
- every parameter must have a type annotation
- the tool description comes from the first line of the docstring
- tools are executed sequentially

You can pass tools as a list:

```python
agent = Agent(
    config=AgentConfig(model="gpt-5"),
    tools=[lookup_user],
)
```

## Environment

The harness reads `OPENAI_API_KEY` automatically if `api_key` is not passed directly.

An `OPENAI_MODEL` value can also be used as a convenience when creating `AgentConfig()` from environment-backed values.

## API Summary

- `Agent(config, tools=None, provider=None)`: main entrypoint
- `await agent.run(prompt, response_model=None)`: final result API
- `agent.stream(prompt, response_model=None)`: streaming API
- `AgentConfig(...)`: runtime configuration
- `@tool`: tool decorator
- `ToolRegistry`: explicit tool registration if you need it

## Development

Run tests:

```bash
uv run pytest
```

This repo keeps tests fully local and does not require a real OpenAI API key for the test suite.
