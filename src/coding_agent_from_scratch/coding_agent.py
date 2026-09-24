import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import stat
import tempfile

from dotenv import load_dotenv
from openai import OpenAI


@dataclass(frozen=True)
class Config:
    api_key: str
    model: str
    workspace_root: Path
    max_output_tokens: int
    max_agent_iterations: int


class ToolError(TypedDict):
    code: str
    message: str


class ToolResult(TypedDict):
    ok: bool
    data: dict[str, Any] | None
    error: ToolError | None


def tool_success(data: dict[str, Any]) -> ToolResult:
    return {
        "ok": True,
        "data": data,
        "error": None,
    }


def tool_failure(code: str, message: str) -> ToolResult:
    return {
        "ok": False,
        "data": None,
        "error": {
            "code": code,
            "message": message,
        },
    }


class WorkspacePathError(ValueError):
    """A requested path is invalid or outside the workspace."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def resolve_workspace_path(
    path_str: str,
    workspace_root: Path,
) -> Path:
    if not isinstance(path_str, str) or not path_str.strip():
        raise WorkspacePathError(
            "INVALID_PATH",
            "Path must be a nonempty string.",
        )

    if "\x00" in path_str:
        raise WorkspacePathError(
            "INVALID_PATH",
            "Path must not contain null characters.",
        )

    try:
        root = workspace_root.resolve(strict=True)

        if not root.is_dir():
            raise WorkspacePathError(
                "INVALID_WORKSPACE",
                "Workspace root must be an existing directory.",
            )

        requested_path = Path(path_str).expanduser()

        if not requested_path.is_absolute():
            requested_path = root / requested_path

        # Resolve ".." and existing symbolic links.
        # Allow nonexistent targets so file creation can use this function.
        resolved_path = requested_path.resolve(strict=False)

    except WorkspacePathError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise WorkspacePathError(
            "INVALID_PATH",
            f"Could not resolve path: {error}",
        ) from error

    if not resolved_path.is_relative_to(root):
        raise WorkspacePathError(
            "OUTSIDE_WORKSPACE",
            f"Path is outside the workspace: {path_str}",
        )

    return resolved_path


def list_files_tool(
    path: str,
    workspace_root: Path,
) -> ToolResult:
    try:
        directory = resolve_workspace_path(path, workspace_root)

        entries = []

        for item in sorted(
            directory.iterdir(),
            key=lambda entry: entry.name,
        ):
            # Check links first so we don't classify their targets.
            if item.is_symlink():
                entry_type = "symlink"
            elif item.is_file():
                entry_type = "file"
            elif item.is_dir():
                entry_type = "directory"
            else:
                entry_type = "other"

            entries.append({
                "name": item.name,
                "type": entry_type,
            })

        return tool_success({
            "path": str(directory),
            "entries": entries,
        })

    except WorkspacePathError as error:
        return tool_failure(error.code, str(error))

    except FileNotFoundError:
        return tool_failure(
            "PATH_NOT_FOUND",
            f"Directory does not exist: {path}",
        )

    except NotADirectoryError:
        return tool_failure(
            "NOT_A_DIRECTORY",
            f"Path is not a directory: {path}",
        )

    except PermissionError:
        return tool_failure(
            "PERMISSION_DENIED",
            f"Permission denied while listing: {path}",
        )

    except OSError as error:
        return tool_failure(
            "DIRECTORY_READ_ERROR",
            f"Could not list directory: {error}",
        )


def read_file_tool(
    path: str,
    workspace_root: Path,
    max_bytes: int = 100_000,
) -> ToolResult:
    if max_bytes <= 0:
        return tool_failure(
            "INVALID_LIMIT",
            "File size limit must be a positive integer.",
        )

    try:
        full_path = resolve_workspace_path(path, workspace_root)

        if full_path.is_dir():
            return tool_failure(
                "NOT_A_FILE",
                f"Path is a directory: {path}",
            )

        if not full_path.exists():
            return tool_failure(
                "FILE_NOT_FOUND",
                f"File does not exist: {path}",
            )

        # Exclude special filesystem entries, such as named pipes.
        if not full_path.is_file():
            return tool_failure(
                "NOT_A_FILE",
                f"Path is not a regular file: {path}",
            )

        with full_path.open("rb") as file:
            raw_content = file.read(max_bytes + 1)

        if len(raw_content) > max_bytes:
            return tool_failure(
                "FILE_TOO_LARGE",
                f"File exceeds the {max_bytes}-byte limit: {path}",
            )

        # A useful binary-content heuristic, not a complete detector.
        if b"\x00" in raw_content:
            return tool_failure(
                "UNSUPPORTED_CONTENT",
                f"File contains null bytes; expected UTF-8 text: {path}",
            )

        content = raw_content.decode("utf-8")

        return tool_success({
            "path": str(full_path),
            "content": content,
            "size_bytes": len(raw_content),
        })

    except WorkspacePathError as error:
        return tool_failure(error.code, str(error))

    except UnicodeDecodeError:
        return tool_failure(
            "UNSUPPORTED_ENCODING",
            f"File is not valid UTF-8 text: {path}",
        )

    except FileNotFoundError:
        return tool_failure(
            "FILE_NOT_FOUND",
            f"File does not exist: {path}",
        )

    except (IsADirectoryError, NotADirectoryError):
        return tool_failure(
            "INVALID_FILE_PATH",
            f"Path does not identify a readable file: {path}",
        )

    except PermissionError:
        return tool_failure(
            "PERMISSION_DENIED",
            f"Permission denied while reading: {path}",
        )

    except OSError as error:
        return tool_failure(
            "FILE_READ_ERROR",
            f"Could not read file: {error}",
        )


def replace_file_content(path: Path, content: str) -> None:
    temporary_path = None

    try:
        permissions = stat.S_IMODE(path.stat().st_mode)

        # Use the same directory so replacement stays on one filesystem.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(content)

        temporary_path.chmod(permissions)
        temporary_path.replace(path)

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def create_file_tool(
    path: str,
    content: str,
    workspace_root: Path,
) -> ToolResult:
    if not isinstance(content, str):
        return tool_failure(
            "INVALID_ARGUMENT",
            "Content must be a string.",
        )

    try:
        full_path = resolve_workspace_path(path, workspace_root)

        # Validate encoding before creating the file.
        encoded_content = content.encode("utf-8")

        # Exclusive creation: fail if the target already exists.
        with full_path.open("xb") as file:
            file.write(encoded_content)

        return tool_success({
            "path": str(full_path),
            "action": "created",
            "size_bytes": len(encoded_content),
        })

    except WorkspacePathError as error:
        return tool_failure(error.code, str(error))

    except FileExistsError:
        return tool_failure(
            "FILE_ALREADY_EXISTS",
            f"Path already exists: {path}",
        )

    except (FileNotFoundError, NotADirectoryError):
        return tool_failure(
            "INVALID_PARENT_DIRECTORY",
            f"Parent directory is missing or invalid: {path}",
        )

    except PermissionError:
        return tool_failure(
            "PERMISSION_DENIED",
            f"Permission denied while creating: {path}",
        )

    except UnicodeEncodeError:
        return tool_failure(
            "INVALID_TEXT",
            "Content cannot be encoded as UTF-8.",
        )

    except OSError as error:
        return tool_failure(
            "FILE_CREATE_ERROR",
            f"Could not create file: {error}",
        )


def edit_file_tool(
    path: str,
    old_str: str,
    new_str: str,
    workspace_root: Path,
) -> ToolResult:
    if not isinstance(old_str, str) or not isinstance(new_str, str):
        return tool_failure(
            "INVALID_ARGUMENT",
            "Both old_str and new_str must be strings.",
        )

    if old_str == "":
        return tool_failure(
            "EMPTY_MATCH",
            "old_str must not be empty. Use create_file for new files.",
        )

    try:
        full_path = resolve_workspace_path(path, workspace_root)

        read_result = read_file_tool(str(full_path), workspace_root)
        if not read_result["ok"]:
            return read_result

        data = read_result["data"]
        assert data is not None
        original = data["content"]

        first_match = original.find(old_str)

        if first_match == -1:
            return tool_failure(
                "MATCH_NOT_FOUND",
                "The requested text was not found. Read the file again.",
            )

        # Starting one character later detects overlapping matches too.
        if original.find(old_str, first_match + 1) != -1:
            return tool_failure(
                "AMBIGUOUS_MATCH",
                "The text appears more than once. Include more surrounding text.",
            )

        if old_str == new_str:
            return tool_success({
                "path": str(full_path),
                "action": "unchanged",
            })

        edited = original.replace(old_str, new_str, 1)
        replace_file_content(full_path, edited)

        return tool_success({
            "path": str(full_path),
            "action": "edited",
            "replacements": 1,
        })

    except WorkspacePathError as error:
        return tool_failure(error.code, str(error))

    except FileNotFoundError:
        return tool_failure(
            "FILE_NOT_FOUND",
            f"File no longer exists: {path}",
        )

    except PermissionError:
        return tool_failure(
            "PERMISSION_DENIED",
            f"Permission denied while editing: {path}",
        )

    except UnicodeEncodeError:
        return tool_failure(
            "INVALID_TEXT",
            "Replacement content cannot be encoded as UTF-8.",
        )

    except OSError as error:
        return tool_failure(
            "FILE_EDIT_ERROR",
            f"Could not edit file: {error}",
        )


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required configuration: {name}")
    return value


def positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default))

    try:
        value = int(raw_value)
    except ValueError:
        raise ValueError(f"{name} must be a positive integer.") from None

    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")

    return value

def load_config() -> Config:
    # Load .env beside this script; existing environment variables take priority.
    load_dotenv(Path(__file__).resolve().parent / ".env")

    api_key = require_env("OPENAI_API_KEY")
    model = require_env("OPENAI_MODEL")

    # Relative workspace paths are resolved against the launch directory.
    workspace_root = Path(
        os.getenv("WORKSPACE_ROOT", ".")
    ).expanduser().resolve()

    if not workspace_root.is_dir():
        raise ValueError(
            f"Workspace must be an existing directory: {workspace_root}"
        )

    return Config(
        api_key=api_key,
        model=model,
        workspace_root=workspace_root,
        max_output_tokens=positive_int_env("MAX_OUTPUT_TOKENS", 5000),
        max_agent_iterations=positive_int_env("MAX_AGENT_ITERATIONS", 10),
    )

def create_client(config: Config) -> OpenAI:
    return OpenAI(api_key=config.api_key)


def main() -> None:
    try:
        config = load_config()
    except (ValueError, OSError) as error:
        raise SystemExit(f"Configuration error: {error}") from None

    with create_client(config) as client:
        print(f"Workspace: {config.workspace_root}")
        print(f"Model: {config.model}")
        print("Configuration loaded; client initialized.")

        # Later: pass config and client to your agent loop.


if __name__ == "__main__":
    main()