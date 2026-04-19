from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel

from agent_harness.providers.base import ConversationItem
from agent_harness.types import AgentEvent, AgentRunResult, ChatMessage, MessageInput

if TYPE_CHECKING:
    from agent_harness.agent import Agent


class ChatSession:
    def __init__(
        self,
        agent: Agent,
        items: Sequence[ConversationItem] | None = None,
    ) -> None:
        self._agent = agent
        self._items = list(items or [])

    @property
    def history(self) -> list[ChatMessage]:
        return self._agent._messages_from_items(self._items)

    @property
    def items(self) -> list[ConversationItem]:
        return list(self._items)

    async def run(
        self,
        input_data: str | Sequence[MessageInput],
        *,
        response_model: type[BaseModel] | None = None,
    ) -> AgentRunResult:
        transcript = [*self._items, *self._agent._normalize_input(input_data)]
        result = await self._agent._run_transcript(transcript, response_model=response_model)
        self._items = transcript
        return result

    async def stream(
        self,
        input_data: str | Sequence[MessageInput],
        *,
        response_model: type[BaseModel] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        transcript = [*self._items, *self._agent._normalize_input(input_data)]

        async for event in self._agent._stream_transcript(transcript, response_model=response_model):
            if event.type == "completed":
                self._items = transcript
            yield event

    def reset(self) -> None:
        self._items.clear()
