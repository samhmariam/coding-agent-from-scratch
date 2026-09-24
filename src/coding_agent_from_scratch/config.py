import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    api_key: str
    model: str
    workspace_root: Path
    max_output_tokens: int
    max_agent_iterations: int


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
    # Load configuration from the launch directory; exported variables take priority.
    load_dotenv(Path.cwd() / ".env")

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
