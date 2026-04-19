from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

from agent_harness.config import AgentConfig
from agent_harness.errors import MaxTurnsExceededError
from agent_harness.providers.base import Provider
from agent_harness.providers.openai import OpenAIResponsesProvider
from agent_harness.tools import ToolRegistry
from agent_harness.types import AgentEvent, AgentRunResult, ToolCallRequest, ToolExecutionResult


class Agent:
    def __init__(
        self,
        config: AgentConfig,
        tools: list[Callable[..., Any]] | ToolRegistry | None = None,
        provider: Provider | None = None,
    ) -> None:
        self.config = config
        self.registry = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
        self.provider = provider or OpenAIResponsesProvider(config)

    async def run(self, prompt: str) -> AgentRunResult:
        transcript = [self._user_message(prompt)]
        tool_results: list[ToolExecutionResult] = []
        raw_responses: list[dict[str, Any]] = []

        for _ in range(self.config.max_turns):
            response = await self.provider.create_response(
                input_items=transcript,
                tools=self.registry.to_openai_tools(),
            )
            raw_responses.append(response.raw_response or {})
            transcript.extend(response.output_items)

            if not response.tool_calls:
                return AgentRunResult(
                    output_text=response.output_text,
                    response_id=response.response_id,
                    tool_results=tool_results,
                    raw_responses=raw_responses,
                )

            for call in response.tool_calls:
                result = await self._execute_tool(call)
                tool_results.append(result)
                transcript.append(self._tool_output_item(result))

        raise MaxTurnsExceededError(
            f"Agent exceeded max_turns={self.config.max_turns} before reaching a final response."
        )

    async def stream(self, prompt: str) -> AsyncIterator[AgentEvent]:
        transcript = [self._user_message(prompt)]
        tool_results: list[ToolExecutionResult] = []
        raw_responses: list[dict[str, Any]] = []

        try:
            for _ in range(self.config.max_turns):
                final_response = None

                async for event in self.provider.stream_response(
                    input_items=transcript,
                    tools=self.registry.to_openai_tools(),
                ):
                    if event.type == "text_delta":
                        yield AgentEvent(type="text_delta", delta=event.delta)
                    elif event.type == "completed":
                        final_response = event.response

                if final_response is None:
                    raise MaxTurnsExceededError("Provider stream completed without a final response.")

                raw_responses.append(final_response.raw_response or {})
                transcript.extend(final_response.output_items)

                if not final_response.tool_calls:
                    result = AgentRunResult(
                        output_text=final_response.output_text,
                        response_id=final_response.response_id,
                        tool_results=tool_results,
                        raw_responses=raw_responses,
                    )
                    yield AgentEvent(type="completed", result=result)
                    return

                for call in final_response.tool_calls:
                    yield AgentEvent(type="tool_call_started", tool_call=call)
                    tool_result = await self._execute_tool(call)
                    tool_results.append(tool_result)
                    transcript.append(self._tool_output_item(tool_result))
                    yield AgentEvent(type="tool_call_completed", tool_result=tool_result)

            raise MaxTurnsExceededError(
                f"Agent exceeded max_turns={self.config.max_turns} before reaching a final response."
            )
        except Exception as exc:
            yield AgentEvent(type="error", error=str(exc))

    async def aclose(self) -> None:
        await self.provider.close()

    async def _execute_tool(self, call: ToolCallRequest) -> ToolExecutionResult:
        return await self.registry.execute(call)

    @staticmethod
    def _user_message(prompt: str) -> dict[str, Any]:
        return {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": prompt}],
        }

    @staticmethod
    def _tool_output_item(result: ToolExecutionResult) -> dict[str, Any]:
        return {
            "type": "function_call_output",
            "call_id": result.call_id,
            "output": result.output,
        }
