# Coding agent from scratch

A Python terminal assistant that can list, read, create, and edit files inside a configured workspace.

## Setup (PowerShell)

From the project root, install the project and its dependencies:

```powershell
uv sync
```

For a new checkout, copy `.env.example` to `.env` and fill in your API key and a model available to your account. Keep `.env` private; it is ignored by Git. Existing users can keep their current `.env`.

```powershell
Copy-Item .env.example .env
```

Configuration is loaded from `.env` in the directory where you launch the application. Exported environment variables take precedence. `WORKSPACE_ROOT` defaults to that launch directory; relative workspace paths are also resolved there.

## Run

From the project root:

```powershell
.\.venv\Scripts\python.exe -m coding_agent_from_scratch
```

Alternatively, run the installed console command:

```powershell
uv run coding-agent-from-scratch
```

Use `/help`, `/clear`, and `/exit` in the terminal. Clearing history does not undo file changes. The assistant has no code execution or test-running tool.

## Tests

After `uv sync`, run from the project root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests use temporary workspaces and mocked model responses. No live API requests are made.

## Layout

```text
src/coding_agent_from_scratch/
    __init__.py       Package metadata
    __main__.py       Module entry point
    cli.py            Terminal commands and output
    agent.py          Bounded model/tool loop
    model.py          OpenAI client and requests
    conversation.py   Conversation history helpers
    config.py         Environment configuration
    instructions.py   Agent instructions
    tools/            File operations, paths, results, registry, dispatcher
tests/
    test_cli.py       Installed launch paths
    test_agent.py     Tool exchanges and iteration limits
    test_config.py    Environment loading
    test_tools.py     File operations and input validation
```

Add new tools under `tools/` and register them in `tools/registry.py`. Keep terminal interactions in `cli.py` and model request handling in `model.py`.

The former nested `coding_agent` package and direct-file launcher have been removed. Use the module or console command above.
