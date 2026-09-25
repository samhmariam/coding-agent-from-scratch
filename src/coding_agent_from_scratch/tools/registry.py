from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

from ..config import Config
from ..approval import WriteApproval, require_approval
from .files import (
    create_file_tool,
    edit_file_tool,
    list_files_tool,
    read_file_tool,
)
from .results import ToolResult
from .search import search_files_tool


@dataclass(frozen=True)
class ToolDefinition:
    handler: Callable[..., ToolResult]
    description: str
    # Maps argument names to their descriptions.
    # All arguments for our current tools are strings.
    parameters: dict[str, str]


def build_tool_registry(config: Config, approval: WriteApproval = require_approval) -> dict[str, ToolDefinition]:
    return {
        "search_files": ToolDefinition(
            handler=partial(search_files_tool, workspace_root=config.workspace_root),
            description=(
                "Recursively search workspace UTF-8 files for case-sensitive literal text. "
                "Returns workspace-relative paths, 1-based line numbers, and matching lines. "
                "Skips generated directories, environment files, links, binary and oversized files. "
                "Results are bounded; narrow the path or query when truncated."
            ),
            parameters={
                "query": "Nonempty single-line literal text to find (not a regular expression).",
                "path": "Workspace directory to search recursively. Use '.' for the root.",
            },
        ),
        "list_files": ToolDefinition(
            handler=partial(
                list_files_tool,
                workspace_root=config.workspace_root,
            ),
            description=(
                "List the immediate children of a workspace directory. "
                "Returns sorted names and entry types. Does not recurse."
            ),
            parameters={
                "path": "Directory path within the workspace. Use '.' for its root.",
            },
        ),
        "read_file": ToolDefinition(
            handler=partial(
                read_file_tool,
                workspace_root=config.workspace_root,
            ),
            description=(
                "Read a UTF-8 text file within the workspace. "
                "Files exceeding the application's size limit are rejected."
            ),
            parameters={
                "path": "Path of the file to read within the workspace.",
            },
        ),
        "create_file": ToolDefinition(
            handler=partial(
                create_file_tool,
                workspace_root=config.workspace_root,
                approval=approval,
            ),
            description=(
                "Create a UTF-8 text file within the workspace. "
                "Fails if the target already exists. "
                "The parent directory must already exist."
            ),
            parameters={
                "path": "Path of the new file within the workspace.",
                "content": "Complete text to write into the new file.",
            },
        ),
        "edit_file": ToolDefinition(
            handler=partial(
                edit_file_tool,
                workspace_root=config.workspace_root,
                approval=approval,
            ),
            description=(
                "Replace exactly one occurrence of text in an existing "
                "UTF-8 workspace file. Read the file before editing. "
                "Fails if the match is empty, missing, or ambiguous."
            ),
            parameters={
                "path": "Path of the existing file within the workspace.",
                "old_str": (
                    "Exact nonempty text to replace. Include enough surrounding "
                    "text to identify exactly one occurrence."
                ),
                "new_str": "Replacement text. Use an empty string to delete the match.",
            },
        ),
    }


def get_tool_schemas(
    registry: dict[str, ToolDefinition],
) -> list[dict[str, Any]]:
    schemas = []

    for name, definition in registry.items():
        schemas.append({
            "type": "function",
            "name": name,
            "description": definition.description,
            "parameters": {
                "type": "object",
                "properties": {
                    argument_name: {
                        "type": "string",
                        "description": argument_description,
                    }
                    for argument_name, argument_description
                    in definition.parameters.items()
                },
                "required": list(definition.parameters),
                "additionalProperties": False,
            },
            "strict": True,
        })

    return schemas
