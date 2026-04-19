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
    await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())
