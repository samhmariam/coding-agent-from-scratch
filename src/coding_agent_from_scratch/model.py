from typing import Any

from openai import OpenAI
from openai.types.responses import Response

from .config import Config
from .instructions import AGENT_INSTRUCTIONS


def execute_llm_call(
    client: OpenAI,
    config: Config,
    conversation: list[Any],
    tool_schemas: list[Any],
) -> Response:
    """Make one model request without modifying conversation history."""
    response = client.responses.create(
        model=config.model,
        instructions=AGENT_INSTRUCTIONS,
        input=conversation,
        tools=tool_schemas,
        max_output_tokens=config.max_output_tokens,
        parallel_tool_calls=False,
    )

    # Do not execute tool calls from an incomplete response.
    if response.status != "completed":
        raise RuntimeError(
            f"Model response did not complete: {response.status}. "
            f"Details: {response.incomplete_details or response.error}"
        )

    return response


def create_client(config: Config) -> OpenAI:
    return OpenAI(api_key=config.api_key)
