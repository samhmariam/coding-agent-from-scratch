# Coding agent from scratch

A Python terminal assistant that can search, list, read, create, and edit files inside a configured workspace.

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
    approval.py       Write previews and conflict checks
    model.py          OpenAI client and requests
    conversation.py   Conversation history helpers
    config.py         Environment configuration
    instructions.py   Agent instructions
    tools/            File operations, paths, results, registry, dispatcher
tests/
    test_cli.py       Installed launch paths
    test_agent.py     Tool exchanges and iteration limits
    test_approval.py  Approval decisions and concurrent file changes
    test_config.py    Environment loading
    test_tools.py     File operations and input validation
```

Add new tools under `tools/` and register them in `tools/registry.py`. Keep terminal interactions in `cli.py` and model request handling in `model.py`.

The former nested `coding_agent` package and direct-file launcher have been removed. Use the module or console command above.

## Workspace search

Try: `Find every reference to load_config in the workspace.`

`search_files` recursively matches case-sensitive literal text, returning workspace-relative paths, 1-based line numbers, and matching text. The model supplies `query` and `path` (use `.` for the root).

Search skips generated directories (including `.git`, `.venv`, `node_modules`, and caches), `.env` files, symbolic links, directory junctions, binary files, and files larger than 100 KB. It does not interpret `.gitignore`. Results are capped at 100 matching lines, 1,000 examined files, and 1,000 traversed directories. Long lines return a 500-character excerpt with its column offset. A limit flag and skipped-file/error counts indicate incomplete coverage; narrow the search when needed.

## Approving file changes

Every agent create/edit operation displays a unified diff before writing. Choose `a` to approve, `r` (or Enter) to reject, or `c` to cancel the turn. EOF and Ctrl+C at the approval prompt also cancel. Read, list, and search operations do not prompt.

Rejection is returned to the model as a tool result. Cancellation stops further tool execution for the turn; earlier approved changes remain. Existing files are checked again after approval, and an intervening content/path change rejects the write with `WRITE_CONFLICT`. Creation never overwrites an existing file. This is an optimistic conflict check, not a filesystem lock against simultaneous writers.

Applications embedding the agent must pass an approval callback to `build_tool_registry(config, approval=callback)`. Without one, registered write tools return `APPROVAL_REQUIRED`. The callback receives an immutable `WriteProposal` and returns `approve`, `reject`, or `cancel`. Low-level file helpers remain usable directly for programmatic file operations.
