from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from simple_agent_base.types import ConversationItem, JSONObject, ToolCallRequest, UsageMetadata

class ProviderResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response_id: str | None = None
    output_text: str = ""
    reasoning_summary: str | None = None
    output_data: BaseModel | None = None
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    output_items: list[ConversationItem] = Field(default_factory=list)
    usage: UsageMetadata | None = None
    raw_response: JSONObject | None = None


class ProviderTextDeltaEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text_delta"] = "text_delta"
    delta: str


class ProviderReasoningDeltaEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["reasoning_delta"] = "reasoning_delta"
    delta: str


class ProviderToolArgumentsDeltaEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_arguments_delta"] = "tool_arguments_delta"
    item_id: str
    call_id: str | None = None
    name: str | None = None
    delta: str


class ProviderHostedToolCallEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["hosted_tool_call_started", "hosted_tool_call_updated", "hosted_tool_call_completed"]
    item_id: str
    tool_type: str
    status: str
    output_index: int | None = None
    sequence_number: int | None = None
    item: JSONObject | None = None


class ProviderCompletedEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["completed"] = "completed"
    response: ProviderResponse

ProviderEvent = (
    ProviderTextDeltaEvent
    | ProviderReasoningDeltaEvent
    | ProviderToolArgumentsDeltaEvent
    | ProviderHostedToolCallEvent
    | ProviderCompletedEvent
)


class Provider(Protocol):
    async def create_response(
        self,
        *,
        input_items: Sequence[ConversationItem],
        tools: Sequence[JSONObject],
        response_model: type[BaseModel] | None = None,
    ) -> ProviderResponse: ...

    async def stream_response(
        self,
        *,
        input_items: Sequence[ConversationItem],
        tools: Sequence[JSONObject],
        response_model: type[BaseModel] | None = None,
    ) -> AsyncIterator[ProviderEvent]: ...

    async def close(self) -> None: ...
