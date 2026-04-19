import pytest

from agent_harness.errors import ToolDefinitionError, ToolRegistrationError
from agent_harness.tools import ToolRegistry, tool


@tool
async def sample_tool(name: str, count: int = 1) -> str:
    """Return a simple greeting."""
    return f"{name}:{count}"


def test_decorator_accepts_valid_async_function() -> None:
    registry = ToolRegistry([sample_tool])
    definition = registry.get("sample_tool")

    assert definition.name == "sample_tool"
    assert definition.description == "Return a simple greeting."


def test_non_async_function_is_rejected() -> None:
    with pytest.raises(ToolDefinitionError):

        @tool
        def invalid_tool(name: str) -> str:
            return name


def test_missing_annotations_are_rejected() -> None:
    with pytest.raises(ToolDefinitionError):

        @tool
        async def invalid_tool(name) -> str:
            return str(name)


def test_duplicate_tool_names_are_rejected() -> None:
    registry = ToolRegistry([sample_tool])

    with pytest.raises(ToolRegistrationError):
        registry.register(sample_tool)


def test_json_schema_matches_parameters() -> None:
    registry = ToolRegistry([sample_tool])
    tool_schema = registry.to_openai_tools()[0]
    parameters = tool_schema["parameters"]

    assert set(parameters["properties"]) == {"name", "count"}
    assert parameters["required"] == ["name"]
