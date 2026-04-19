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

## Install

```bash
uv sync
```

Set your API key:

```bash
$env:OPENAI_API_KEY="your_key_here"
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

## Streaming

Use `run()` when you want the final structured result.

Use `stream()` when you want incremental text and tool lifecycle events:

```python
async for event in agent.stream("Say hi"):
    if event.type == "text_delta":
        print(event.delta, end="")
```

## Environment

The harness reads `OPENAI_API_KEY` automatically if `api_key` is not passed directly.

An `OPENAI_MODEL` value can also be used as a convenience when creating `AgentConfig()` from environment-backed values.

## Example

See [examples/basic_agent.py](/C:/Users/Anson/Desktop/agent-harness-base/examples/basic_agent.py).
