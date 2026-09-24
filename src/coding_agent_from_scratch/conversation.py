import json
from typing import Any

from openai.types.responses import Response


def add_user_message(
    conversation: list[Any],
    message: str,
) -> None:
    """Append an actual user message to the conversation."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("User message must be a non-empty string.")

    conversation.append({
        "role": "user",
        "content": message,
    })


def add_model_response(
    conversation: list[Any],
    response: Response,
) -> None:
    """Preserve every output item, including tool calls and reasoning."""
    conversation.extend(response.output)


def add_tool_result(
    conversation: list[Any],
    call_id: str,
    result: dict[str, Any],
) -> None:
    """Attach a tool result to the precise call that requested it."""
    if not isinstance(call_id, str) or not call_id.strip():
        raise ValueError("Tool call ID must be a non-empty string.")

    conversation.append({
        "type": "function_call_output",
        "call_id": call_id,
        "output": json.dumps(result, ensure_ascii=False),
    })
