import json

from ..approval import WriteCancelled

from .registry import ToolDefinition
from .results import ToolResult, tool_failure


def dispatch_tool_call(
    name: str,
    arguments_json: str,
    registry: dict[str, ToolDefinition],
) -> ToolResult:
    if not isinstance(name, str) or name not in registry:
        return tool_failure(
            "UNKNOWN_TOOL",
            f"Unknown tool: {name!r}",
        )

    if not isinstance(arguments_json, str):
        return tool_failure(
            "INVALID_ARGUMENTS",
            "Tool arguments must be a JSON string.",
        )

    try:
        arguments = json.loads(arguments_json)
    except json.JSONDecodeError as error:
        return tool_failure(
            "INVALID_JSON",
            f"Could not parse tool arguments: {error.msg}",
        )

    if not isinstance(arguments, dict):
        return tool_failure(
            "INVALID_ARGUMENTS",
            "Tool arguments must decode to a JSON object.",
        )

    definition = registry[name]
    expected = set(definition.parameters)
    received = set(arguments)

    missing = expected - received
    unexpected = received - expected

    if missing:
        return tool_failure(
            "MISSING_ARGUMENTS",
            f"Missing required arguments: {', '.join(sorted(missing))}",
        )

    if unexpected:
        return tool_failure(
            "UNEXPECTED_ARGUMENTS",
            f"Unexpected arguments: {', '.join(sorted(unexpected))}",
        )

    # All tools in our current registry require string arguments.
    invalid_types = [
        key
        for key, value in arguments.items()
        if not isinstance(value, str)
    ]

    if invalid_types:
        return tool_failure(
            "INVALID_ARGUMENT_TYPE",
            "These arguments must be strings: "
            + ", ".join(sorted(invalid_types)),
        )

    try:
        return definition.handler(**arguments)
    except WriteCancelled:
        raise
    except Exception:
        # Expected failures are already handled by individual tools.
        # Keep an unexpected failure from terminating the agent loop.
        return tool_failure(
            "TOOL_EXECUTION_ERROR",
            f"Tool '{name}' encountered an unexpected error.",
        )
