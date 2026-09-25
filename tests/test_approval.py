import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from coding_agent_from_scratch.agent import run_agent_turn
from coding_agent_from_scratch.approval import WriteCancelled, WriteProposal
from coding_agent_from_scratch.cli import prompt_write_approval
from coding_agent_from_scratch.config import Config
from coding_agent_from_scratch.tools.dispatcher import dispatch_tool_call
from coding_agent_from_scratch.tools.registry import build_tool_registry, get_tool_schemas


class ApprovalTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.config = Config("key", "model", self.root, 100, 3)
        self.path = self.root / "example.txt"
        self.approve = Mock(return_value="approve")
        self.registry = build_tool_registry(self.config, approval=self.approve)

    def dispatch(self, name="create_file", **arguments):
        return dispatch_tool_call(name, json.dumps(arguments), self.registry)

    def test_create_previews_exact_content_before_writing(self):
        def approve(proposal):
            self.assertFalse(self.path.exists())
            self.assertEqual(proposal.path, "example.txt")
            self.assertIsNone(proposal.before)
            self.assertEqual(proposal.after, "hello\n")
            self.assertIn("+hello\n", proposal.diff())
            return "approve"
        self.approve.side_effect = approve
        self.assertTrue(self.dispatch(path="example.txt", content="hello\n")["ok"])
        self.assertEqual(self.path.read_bytes(), b"hello\n")

    def test_edit_previews_and_applies_exact_replacement(self):
        self.path.write_bytes(b"hello\r\nworld\r\n")
        result = self.dispatch("edit_file", path="example.txt", old_str="world", new_str="Python")
        self.assertTrue(result["ok"])
        proposal = self.approve.call_args.args[0]
        self.assertEqual(proposal.before, "hello\r\nworld\r\n")
        self.assertIn("-world", proposal.diff())
        self.assertIn("+Python", proposal.diff())
        self.assertEqual(self.path.read_bytes(), b"hello\r\nPython\r\n")

    def test_reject_does_not_create_or_edit(self):
        self.approve.return_value = "reject"
        self.assertEqual(self.dispatch(path="example.txt", content="hello")["error"]["code"], "WRITE_REJECTED")
        self.assertFalse(self.path.exists())
        self.path.write_text("original", encoding="utf-8")
        result = self.dispatch("edit_file", path="example.txt", old_str="original", new_str="changed")
        self.assertEqual(result["error"]["code"], "WRITE_REJECTED")
        self.assertEqual(self.path.read_text(), "original")

    def test_missing_approval_ui_fails_closed(self):
        self.registry = build_tool_registry(self.config)
        result = self.dispatch(path="example.txt", content="hello")
        self.assertEqual(result["error"]["code"], "APPROVAL_REQUIRED")
        self.assertFalse(self.path.exists())

    def test_concurrent_edit_is_not_overwritten(self):
        self.path.write_text("hello", encoding="utf-8")
        def approve(proposal):
            self.path.write_text("external change", encoding="utf-8")
            return "approve"
        self.approve.side_effect = approve
        result = self.dispatch("edit_file", path="example.txt", old_str="hello", new_str="goodbye")
        self.assertEqual(result["error"]["code"], "WRITE_CONFLICT")
        self.assertEqual(self.path.read_text(), "external change")

    def test_concurrent_create_is_not_overwritten(self):
        def approve(proposal):
            self.path.write_text("external creation", encoding="utf-8")
            return "approve"
        self.approve.side_effect = approve
        result = self.dispatch(path="example.txt", content="hello")
        self.assertEqual(result["error"]["code"], "WRITE_CONFLICT")
        self.assertEqual(self.path.read_text(), "external creation")

    def test_deleted_edit_target_is_not_recreated(self):
        self.path.write_text("hello", encoding="utf-8")
        def approve(proposal):
            self.path.unlink()
            return "approve"
        self.approve.side_effect = approve
        result = self.dispatch("edit_file", path="example.txt", old_str="hello", new_str="goodbye")
        self.assertEqual(result["error"]["code"], "WRITE_CONFLICT")
        self.assertFalse(self.path.exists())

    def test_read_tools_and_invalid_writes_do_not_prompt(self):
        self.path.write_text("hello hello", encoding="utf-8")
        for name, args in [
            ("list_files", {"path": "."}),
            ("read_file", {"path": "example.txt"}),
            ("search_files", {"path": ".", "query": "hello"}),
        ]:
            self.assertTrue(self.dispatch(name, **args)["ok"])
        self.assertFalse(self.dispatch("edit_file", path="example.txt", old_str="hello", new_str="x")["ok"])
        self.assertFalse(self.dispatch(path="../escape.txt", content="x")["ok"])
        self.assertFalse(self.dispatch(path="example.txt", content="x")["ok"])
        self.approve.assert_not_called()

    def test_approval_cannot_be_supplied_by_model(self):
        result = self.dispatch(path="example.txt", content="hello", approval="approve")
        self.assertEqual(result["error"]["code"], "UNEXPECTED_ARGUMENTS")
        for schema in get_tool_schemas(self.registry):
            self.assertNotIn("approval", schema["parameters"]["properties"])

    def test_rejection_is_returned_to_model(self):
        self.approve.return_value = "reject"
        call = SimpleNamespace(type="function_call", name="create_file", call_id="write-1", arguments='{"path":"example.txt","content":"hello"}')
        client = Mock()
        client.responses.create.side_effect = [
            SimpleNamespace(status="completed", output=[call]),
            SimpleNamespace(status="completed", output=[], output_text="Change rejected."),
        ]
        history = []
        answer = run_agent_turn(client, self.config, history, self.registry, "Create a file")
        self.assertEqual(answer, "Change rejected.")
        self.assertEqual(json.loads(history[2]["output"])["error"]["code"], "WRITE_REJECTED")
        self.assertFalse(self.path.exists())

    def test_cancel_stops_turn_and_completes_pending_call_history(self):
        self.approve.return_value = "cancel"
        calls = [SimpleNamespace(type="function_call", name="create_file", call_id=f"write-{i}", arguments=json.dumps({"path": f"{i}.txt", "content": "hello"})) for i in range(2)]
        client = Mock()
        client.responses.create.return_value = SimpleNamespace(status="completed", output=calls)
        history = []
        with self.assertRaises(WriteCancelled):
            run_agent_turn(client, self.config, history, self.registry, "Create files")
        self.assertEqual(client.responses.create.call_count, 1)
        self.approve.assert_called_once()
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertEqual([item["call_id"] for item in history[-2:]], ["write-0", "write-1"])
        self.assertTrue(all(json.loads(item["output"])["error"]["code"] == "TURN_CANCELLED" for item in history[-2:]))

    def test_terminal_prompt_defaults_to_reject_and_handles_cancel(self):
        proposal = WriteProposal("test.txt", None, "\x1b[2Jhello")
        for response, expected in [("", "reject"), ("a", "approve"), ("r", "reject"), ("c", "cancel")]:
            with self.subTest(response=response), patch("builtins.input", return_value=response), patch("sys.stdout", new_callable=io.StringIO) as output:
                self.assertEqual(prompt_write_approval(proposal), expected)
                self.assertNotIn("\x1b", output.getvalue())
        for error in (EOFError, KeyboardInterrupt):
            with patch("builtins.input", side_effect=error), patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(prompt_write_approval(proposal), "cancel")

    def test_preview_marks_missing_newline_and_empty_creation(self):
        self.assertIn("No newline at end of file", WriteProposal("x", "old", "new").diff())
        self.assertIn("Empty file creation", WriteProposal("x", None, "").diff())
