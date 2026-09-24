from pathlib import Path


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
