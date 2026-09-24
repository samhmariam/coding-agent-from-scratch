from typing import Any, TypedDict


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
