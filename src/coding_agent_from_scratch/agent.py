from typing import Any

from openai import OpenAI

from .config import Config
from .conversation import add_model_response, add_tool_result, add_user_message
from .model import execute_llm_call
from .tools.dispatcher import dispatch_tool_call
from .tools.registry import ToolDefinition, get_tool_schemas
from .tools.results import ToolResult, tool_failure


class AgentIterationLimitError(RuntimeError):
    """Raised when a turn reaches its model-request limit."""


def run_agent_turn(
    client: OpenAI,
    config: Config,
    conversation: list[Any],
    registry: dict[str, ToolDefinition],
    user_message: str,
) -> str:
    """Run one user turn and return the assistant's final text."""
    tool_schemas = get_tool_schemas(registry)
    add_user_message(conversation, user_message)

    for _ in range(config.max_agent_iterations):
        response = execute_llm_call(
            client=client,
            config=config,
            conversation=conversation,
            tool_schemas=tool_schemas,
        )

        add_model_response(conversation, response)

        tool_calls = [
            item
            for item in response.output
            if item.type == "function_call"
        ]

        # A completed response without tool calls ends this turn.
        if not tool_calls:
            answer = response.output_text.strip()

            if not answer:
                raise RuntimeError(
                    "The model returned no text and requested no tools."
                )

            return answer

        for index, tool_call in enumerate(tool_calls):
            try:
                result = dispatch_tool_call(
                    name=tool_call.name,
                    arguments_json=tool_call.arguments,
                    registry=registry,
                )
            except KeyboardInterrupt:
                # An interrupted write may already have changed a file.
                # Record the uncertainty rather than claiming failure
                # or automatically retrying the operation.
                add_tool_result(
                    conversation=conversation,
                    call_id=tool_call.call_id,
                    result=tool_failure(
                        "TOOL_INTERRUPTED",
                        "Execution was interrupted. The operation may "
                        "have partially or fully completed. Inspect "
                        "the affected files before retrying.",
                    ),
                )

                # Every outstanding call needs a result so that the
                # conversation remains usable on the next user turn.
                for pending_call in tool_calls[index + 1:]:
                    add_tool_result(
                        conversation=conversation,
                        call_id=pending_call.call_id,
                        result=tool_failure(
                            "TOOL_SKIPPED",
                            "This tool was not executed because the "
                            "user interrupted the turn.",
                        ),
                    )

                raise

            add_tool_result(
                conversation=conversation,
                call_id=tool_call.call_id,
                result=result,
            )

        # All results are now in history. The next iteration lets
        # the model interpret them and answer or request more tools.

    raise AgentIterationLimitError(
        f"Stopped after {config.max_agent_iterations} model requests "
        "without a final answer. Completed tool operations remain "
        "in effect; inspect the workspace before continuing."
    )
