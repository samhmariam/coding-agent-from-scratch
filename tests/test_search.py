import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from coding_agent_from_scratch.config import Config
from coding_agent_from_scratch.tools.dispatcher import dispatch_tool_call
from coding_agent_from_scratch.tools.registry import build_tool_registry, get_tool_schemas
from coding_agent_from_scratch.tools.search import search_files_tool


class SearchTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()

    def write(self, name, text):
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return target

    def search(self, query="needle", **kwargs):
        result = search_files_tool(query, self.root, **kwargs)
        self.assertTrue(result["ok"], result)
        return result["data"]

    def test_recursive_literal_matches_include_paths_and_line_numbers(self):
        self.write("a.py", "intro\nneedle needle\nNeedle\n")
        self.write("nested/b.py", "needle\n")
        data = self.search()
        self.assertEqual(
            [(m["path"], m["line_number"], m["text"]) for m in data["matches"]],
            [("a.py", 2, "needle needle"), ("nested/b.py", 1, "needle")],
        )
        self.assertFalse(data["truncated"])
        self.write("literal.txt", "a.b\naxb\n")
        self.assertEqual(len(self.search("a.b")["matches"]), 1)

    def test_subdirectory_scope(self):
        self.write("outside.txt", "needle")
        self.write("src/inside.txt", "needle")
        self.assertEqual([m["path"] for m in self.search(path="src")["matches"]], ["src/inside.txt"])

    def test_skips_generated_directories_env_binary_and_large_files(self):
        for name in (".git", ".venv", "__pycache__", "node_modules", "build", "dist"):
            self.write(f"{name}/hidden.txt", "needle")
        self.write(".env", "needle")
        self.write(".env.local", "needle")
        self.write("large.txt", "needle" * 10)
        (self.root / "binary.bin").write_bytes(b"needle\x00")
        (self.root / "invalid.txt").write_bytes(b"needle\xff")
        self.write("visible.txt", "needle")
        data = self.search(max_file_bytes=20)
        self.assertEqual([m["path"] for m in data["matches"]], ["visible.txt"])
        self.assertEqual(data["files_skipped"], 5)

    def test_invalid_and_outside_paths(self):
        self.write("file.txt", "needle")
        for path, code in [("../escape", "OUTSIDE_WORKSPACE"), ("missing", "PATH_NOT_FOUND"), ("file.txt", "NOT_A_DIRECTORY")]:
            with self.subTest(path=path):
                result = search_files_tool("needle", self.root, path)
                self.assertEqual(result["error"]["code"], code)
        self.write(".git/file.txt", "needle")
        self.assertEqual(search_files_tool("needle", self.root, ".git")["error"]["code"], "EXCLUDED_PATH")

    def test_invalid_queries(self):
        for query in ("", "a\nb", "a\rb", None):
            with self.subTest(query=query):
                self.assertEqual(search_files_tool(query, self.root)["error"]["code"], "INVALID_QUERY")

    def test_result_limit_is_reported_only_when_a_match_is_omitted(self):
        self.write("a.txt", "needle\nneedle\nneedle")
        self.assertFalse(self.search(max_results=3)["truncated"])
        data = self.search(max_results=2)
        self.assertEqual(len(data["matches"]), 2)
        self.assertEqual(data["limit_reached"], "max_results")

    def test_scan_limits_are_reported_even_without_matches(self):
        self.write("a.txt", "other")
        self.write("b.txt", "other")
        self.write("nested/c.txt", "other")
        self.assertEqual(self.search(max_files=1)["limit_reached"], "max_files")
        self.assertEqual(self.search(max_directories=1)["limit_reached"], "max_directories")

    def test_long_line_excerpt_contains_the_match(self):
        self.write("long.txt", "x" * 2_000 + "needle" + "y" * 2_000)
        match = self.search()["matches"][0]
        self.assertEqual(len(match["text"]), 500)
        self.assertIn("needle", match["text"])
        self.assertTrue(match["text_truncated"])
        self.assertEqual(match["column_offset"], 1900)

    def test_registry_dispatch_and_schema(self):
        self.write("a.txt", "needle")
        registry = build_tool_registry(Config("key", "model", self.root, 100, 2))
        result = dispatch_tool_call("search_files", json.dumps({"query": "needle", "path": "."}), registry)
        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["matches"][0]["path"], "a.txt")
        schema = next(s for s in get_tool_schemas(registry) if s["name"] == "search_files")
        self.assertEqual(set(schema["parameters"]["required"]), {"query", "path"})
        result = dispatch_tool_call("search_files", '{"query":"needle","path":".","max_results":"100000"}', registry)
        self.assertEqual(result["error"]["code"], "UNEXPECTED_ARGUMENTS")

    def test_directory_errors_are_reported(self):
        def failed_walk(*args, **kwargs):
            kwargs["onerror"](PermissionError("denied"))
            return iter(())
        with patch("coding_agent_from_scratch.tools.search.os.walk", side_effect=failed_walk):
            self.assertEqual(self.search()["directory_errors"], 1)

    def test_symlinks_are_skipped_and_outside_target_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            outside = Path(outside).resolve()
            (outside / "secret.txt").write_text("needle", encoding="utf-8")
            try:
                (self.root / "link").symlink_to(outside, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"Creating symlinks is unavailable: {error}")
            self.assertEqual(self.search()["matches"], [])
            self.assertEqual(search_files_tool("needle", self.root, "link")["error"]["code"], "OUTSIDE_WORKSPACE")
