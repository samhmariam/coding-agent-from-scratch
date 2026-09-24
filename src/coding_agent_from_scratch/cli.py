from typing import Any

from openai import OpenAIError

from .agent import AgentIterationLimitError, run_agent_turn
from .config import load_config
from .model import create_client
from .tools.registry import build_tool_registry


def main() -> None:
    try:
        config = load_config()
    except (ValueError, OSError) as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc

    registry = build_tool_registry(config)
    conversation: list[Any] = []

    print("Coding agent")
    print(f"Workspace: {config.workspace_root}")
    print(f"Model: {config.model}")
    print("Commands: /help, /clear, /exit")
    print("Enter one message per line.")

    with create_client(config) as client:
        while True:
            try:
                user_message = input("\nYou: ").strip()
            except EOFError:
                print("\nGoodbye.")
                break
            except KeyboardInterrupt:
                print("\nGoodbye.")
                break

            if not user_message:
                continue

            command = user_message.lower()

            if command == "/exit":
                print("Goodbye.")
                break

            if command == "/help":
                print(
                    "\n/help  — Show available commands\n"
                    "/clear — Clear conversation history; keep files\n"
                    "/exit  — Exit the agent\n\n"
                    "Ctrl+C during an agent turn interrupts that turn.\n"
                    "Ctrl+C at the input prompt exits the program."
                )
                continue

            if command == "/clear":
                conversation.clear()
                print("Conversation cleared. Workspace files are unchanged.")
                continue

            print("\nAgent: Working...", flush=True)

            try:
                answer = run_agent_turn(
                    client=client,
                    config=config,
                    conversation=conversation,
                    registry=registry,
                    user_message=user_message,
                )
            except KeyboardInterrupt:
                # Reset history because interruption can happen while
                # a response or tool result is being appended, leaving
                # an unfinished function-call exchange.
                conversation.clear()
                print(
                    "\nTurn interrupted. Conversation history cleared.\n"
                    "Completed file changes remain. Inspect affected "
                    "files before retrying."
                )
            except AgentIterationLimitError as exc:
                print(f"\nAgent: {exc}")
            except OpenAIError as exc:
                print(f"\nOpenAI request failed: {exc}")
                print(
                    "Earlier tool operations may have completed. "
                    "Inspect affected files before retrying."
                )
            except RuntimeError as exc:
                print(f"\nAgent stopped: {exc}")
            else:
                print(f"\nAgent: {answer}")
