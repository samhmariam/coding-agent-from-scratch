from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from coding_agent_from_scratch.config import Config
from coding_agent_from_scratch.tools.dispatcher import dispatch_tool_call
from coding_agent_from_scratch.tools.files import create_file_tool, edit_file_tool, read_file_tool
from coding_agent_from_scratch.tools.paths import WorkspacePathError, resolve_workspace_path
from coding_agent_from_scratch.tools.registry import build_tool_registry


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.config = Config("offline-key", "offline-model", self.root, 100, 2)
        self.registry = build_tool_registry(self.config)

    def test_parent_escape_is_rejected(self):
        with self.assertRaises(WorkspacePathError) as caught:
            resolve_workspace_path("../outside.txt", self.root)
        self.assertEqual(caught.exception.code, "OUTSIDE_WORKSPACE")

    def test_create_does_not_overwrite_existing_file(self):
        self.assertTrue(create_file_tool("example.txt", "original", self.root)["ok"])
        result = create_file_tool("example.txt", "replacement", self.root)
        self.assertEqual(result["error"]["code"], "FILE_ALREADY_EXISTS")
        self.assertEqual((self.root / "example.txt").read_text(), "original")

    def test_edit_rejects_ambiguous_match_without_changing_file(self):
        path = self.root / "example.txt"
        path.write_text("hello hello", encoding="utf-8")
        result = edit_file_tool("example.txt", "hello", "goodbye", self.root)
        self.assertEqual(result["error"]["code"], "AMBIGUOUS_MATCH")
        self.assertEqual(path.read_text(), "hello hello")

    def test_unique_edit_can_be_read_back(self):
        create_file_tool("example.txt", "hello world", self.root)
        result = edit_file_tool("example.txt", "world", "Python", self.root)
        self.assertTrue(result["ok"])
        self.assertEqual(
            read_file_tool("example.txt", self.root)["data"]["content"],
            "hello Python",
        )

    def test_invalid_arguments_never_reach_handler(self):
        handler = Mock()
        definition = self.registry["read_file"]
        self.registry["read_file"] = type(definition)(
            handler, definition.description, definition.parameters
        )
        cases = [
            ("{", "INVALID_JSON"),
            ('{}', "MISSING_ARGUMENTS"),
            ('{"path": 42}', "INVALID_ARGUMENT_TYPE"),
            ('{"path": "x", "workspace_root": "/"}', "UNEXPECTED_ARGUMENTS"),
        ]
        for arguments, code in cases:
            with self.subTest(arguments=arguments):
                result = dispatch_tool_call("read_file", arguments, self.registry)
                self.assertEqual(result["error"]["code"], code)
        handler.assert_not_called()
