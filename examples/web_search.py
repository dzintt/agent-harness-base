import asyncio

from simple_agent_base import Agent, AgentConfig


async def main() -> None:
    agent = Agent(
        config=AgentConfig(model="gpt-5.4"),
        hosted_tools=[{"type": "web_search"}],
    )

    try:
        result = await agent.run("What's the latest news about Python 3.13?")
        print("Final text:")
        print(result.output_text)
    finally:
        await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())
