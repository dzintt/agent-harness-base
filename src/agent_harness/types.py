from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Awaitable[Any]] = field(repr=False)
    arguments_model: type[BaseModel] = field(repr=False)
    strict: bool = True


class ToolCallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw_arguments: str = "{}"


class ToolExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    output: str
    raw_output: Any | None = None


class AgentRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_text: str
    response_id: str | None = None
    tool_results: list[ToolExecutionResult] = Field(default_factory=list)
    raw_responses: list[dict[str, Any]] = Field(default_factory=list)


class AgentEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[
        "text_delta",
        "tool_call_started",
        "tool_call_completed",
        "completed",
        "error",
    ]
    delta: str | None = None
    tool_call: ToolCallRequest | None = None
    tool_result: ToolExecutionResult | None = None
    result: AgentRunResult | None = None
    error: str | None = None
