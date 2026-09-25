import stat
import tempfile
from pathlib import Path
from typing import Any

from ..approval import WriteApproval, WriteApprovalError, approve_change

from .paths import WorkspacePathError, resolve_workspace_path
from .results import ToolResult, tool_failure, tool_success


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
    approval: WriteApproval | None = None,
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

        if full_path.exists() or full_path.is_symlink():
            return tool_failure("FILE_ALREADY_EXISTS", f"Path already exists: {path}")
        if not full_path.parent.is_dir():
            return tool_failure("INVALID_PARENT_DIRECTORY", "Parent directory must exist.")
        approve_change(path, full_path, workspace_root, None, content, approval)

        # Exclusive creation: fail if the target already exists.
        with full_path.open("xb") as file:
            file.write(encoded_content)

        return tool_success({
            "path": str(full_path),
            "action": "created",
            "size_bytes": len(encoded_content),
        })

    except WriteApprovalError as error:
        return tool_failure(error.code, str(error))

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
    approval: WriteApproval | None = None,
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
        edited.encode("utf-8")
        approve_change(path, full_path, workspace_root, original, edited, approval)
        replace_file_content(full_path, edited)

        return tool_success({
            "path": str(full_path),
            "action": "edited",
            "replacements": 1,
        })

    except WriteApprovalError as error:
        return tool_failure(error.code, str(error))

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
