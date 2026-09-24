"""Bounded, literal text search over workspace files."""

import os
from pathlib import Path

from .files import read_file_tool
from .paths import WorkspacePathError, resolve_workspace_path
from .results import ToolResult, tool_failure, tool_success

EXCLUDED_DIRECTORIES = frozenset({
    ".git", ".venv", "venv", "__pycache__", "node_modules",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "build", "dist",
})


def search_files_tool(
    query: str,
    workspace_root: Path,
    path: str = ".",
    *,
    max_results: int = 100,
    max_file_bytes: int = 100_000,
    max_files: int = 1_000,
    max_directories: int = 1_000,
) -> ToolResult:
    """Return one match per line; limits remain application-controlled."""
    if not isinstance(query, str) or not query or "\n" in query or "\r" in query:
        return tool_failure("INVALID_QUERY", "Query must be nonempty, single-line text.")
    if any(value <= 0 for value in (max_results, max_file_bytes, max_files, max_directories)):
        return tool_failure("INVALID_LIMIT", "Search limits must be positive.")

    try:
        root = workspace_root.resolve(strict=True)
        start = resolve_workspace_path(path, root)
        if not start.exists():
            return tool_failure("PATH_NOT_FOUND", f"Directory does not exist: {path}")
        if not start.is_dir():
            return tool_failure("NOT_A_DIRECTORY", f"Search path must be a directory: {path}")
        if any(part in EXCLUDED_DIRECTORIES for part in start.relative_to(root).parts):
            return tool_failure("EXCLUDED_PATH", "The search path is an excluded directory.")
    except WorkspacePathError as error:
        return tool_failure(error.code, str(error))
    except OSError as error:
        return tool_failure("SEARCH_ERROR", str(error))

    matches = []
    files_examined = 0
    files_searched = 0
    directories_examined = 0
    skipped = 0
    directory_errors = 0

    def record_directory_error(error: OSError) -> None:
        nonlocal directory_errors
        directory_errors += 1

    def finish(reason: str | None = None) -> ToolResult:
        return tool_success({
            "query": query,
            "path": start.relative_to(root).as_posix(),
            "matches": matches,
            "files_searched": files_searched,
            "files_skipped": skipped,
            "directory_errors": directory_errors,
            "truncated": reason is not None,
            "limit_reached": reason,
        })

    for directory, directories, filenames in os.walk(
        start, topdown=True, followlinks=False, onerror=record_directory_error,
    ):
        if directories_examined >= max_directories:
            return finish("max_directories")
        directories_examined += 1
        # Resolve each traversed directory as well as every file before reading.
        try:
            current = resolve_workspace_path(directory, root)
        except WorkspacePathError:
            directories[:] = []
            skipped += 1
            continue
        directories[:] = sorted(
            name for name in directories
            if name not in EXCLUDED_DIRECTORIES
            and not (current / name).is_symlink()
            and not (current / name).is_junction()
        )
        for name in sorted(filenames):
            if files_examined >= max_files:
                return finish("max_files")
            files_examined += 1
            candidate = current / name
            # Avoid accidentally returning local environment credentials.
            if name == ".env" or name.startswith(".env.") or candidate.is_symlink():
                skipped += 1
                continue
            result = read_file_tool(str(candidate), root, max_bytes=max_file_bytes)
            if not result["ok"]:
                skipped += 1
                continue
            files_searched += 1
            data = result["data"]
            assert data is not None
            for line_number, line in enumerate(data["content"].splitlines(), start=1):
                position = line.find(query)
                if position < 0:
                    continue
                if len(matches) >= max_results:
                    return finish("max_results")
                # Bound output even when a minified file has a very long line.
                offset = max(0, position - 100) if len(line) > 500 else 0
                matches.append({
                    "path": candidate.relative_to(root).as_posix(),
                    "line_number": line_number,
                    "text": line[offset:offset + 500],
                    "text_truncated": len(line) > 500,
                    "column_offset": offset,
                })
    return finish()
