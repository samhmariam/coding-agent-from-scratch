"""Write previews and approval checks, independent of the terminal UI."""

from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path
from typing import Callable, Literal

from .tools.paths import resolve_workspace_path


class WriteCancelled(Exception):
    """The user cancelled the current agent turn before a write."""


class WriteApprovalError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class WriteProposal:
    path: str
    before: str | None
    after: str

    def diff(self) -> str:
        def lines(text: str) -> list[str]:
            # Preserve missing-final-newline information in the preview.
            result = text.splitlines(keepends=True)
            if result and not result[-1].endswith("\n"):
                result[-1] += "\n\\ No newline at end of file\n"
            return result

        return "".join(unified_diff(
            lines(self.before or ""), lines(self.after),
            fromfile="/dev/null" if self.before is None else f"a/{self.path}",
            tofile=f"b/{self.path}",
        )) or "(Empty file creation; no text diff.)\n"


ApprovalDecision = Literal["approve", "reject", "cancel"]
WriteApproval = Callable[[WriteProposal], ApprovalDecision]


def require_approval(proposal: WriteProposal) -> Literal["reject"]:
    """Fail closed when the application has not supplied an approval UI."""
    raise WriteApprovalError("APPROVAL_REQUIRED", "A user approval callback is required to write files.")


def approve_change(
    requested_path: str,
    target: Path,
    workspace_root: Path,
    before: str | None,
    after: str,
    approval: WriteApproval | None,
) -> None:
    # Low-level file helpers can be used directly; registered agent tools
    # always supply an approval callback (or the fail-closed default).
    if approval is None:
        return
    snapshot = target.stat() if before is not None else None
    proposal = WriteProposal(target.relative_to(workspace_root.resolve()).as_posix(), before, after)
    decision = approval(proposal)
    if decision == "cancel":
        raise WriteCancelled()
    if decision != "approve":
        raise WriteApprovalError("WRITE_REJECTED", "The user rejected this change. Do not retry it without a new user request.")

    conflict = False
    try:
        conflict = resolve_workspace_path(requested_path, workspace_root) != target
        if before is None:
            conflict = conflict or target.exists() or target.is_symlink()
        else:
            current = target.stat()
            fingerprint = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
            conflict = conflict or fingerprint(snapshot) != fingerprint(current) or not target.is_file()
            if not conflict:
                with target.open("rb") as file:
                    conflict = file.read(len(before.encode("utf-8")) + 1) != before.encode("utf-8")
    except (OSError, ValueError):
        conflict = True
    if conflict:
        raise WriteApprovalError("WRITE_CONFLICT", "The file or its path changed during approval. Read it again and request approval for a fresh preview.")
