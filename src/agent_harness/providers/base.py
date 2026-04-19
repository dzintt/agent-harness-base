from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from agent_harness.types import ToolCallRequest

ConversationItem = dict[str, Any]


class ProviderResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response_id: str | None = None
    output_text: str = ""
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    output_items: list[ConversationItem] = Field(default_factory=list)
    raw_response: dict[str, Any] | None = None


class ProviderTextDeltaEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text_delta"] = "text_delta"
    delta: str


class ProviderCompletedEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["completed"] = "completed"
    response: ProviderResponse


ProviderEvent = ProviderTextDeltaEvent | ProviderCompletedEvent


class Provider(Protocol):
    async def create_response(
        self,
        *,
        input_items: Sequence[ConversationItem],
        tools: Sequence[dict[str, Any]],
    ) -> ProviderResponse: ...

    async def stream_response(
        self,
        *,
        input_items: Sequence[ConversationItem],
        tools: Sequence[dict[str, Any]],
    ) -> AsyncIterator[ProviderEvent]: ...

    async def close(self) -> None: ...
