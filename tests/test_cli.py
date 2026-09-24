import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from coding_agent_from_scratch import cli

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LaunchTests(unittest.TestCase):
    def test_module_entry_point_calls_canonical_main(self):
        with patch.object(cli, "main") as main:
            runpy.run_module("coding_agent_from_scratch", run_name="__main__")
        main.assert_called_once_with()

    def test_module_and_console_script_start_and_exit(self):
        executable = Path(sys.executable)
        console = executable.parent / (
            "coding-agent-from-scratch.exe" if os.name == "nt" else "coding-agent-from-scratch"
        )
        self.assertTrue(console.is_file(), "Install the project before running the tests.")
        with tempfile.TemporaryDirectory() as directory:
            env = os.environ.copy()
            env.update({
                "OPENAI_API_KEY": "offline-test-key",
                "OPENAI_MODEL": "offline-test-model",
                "WORKSPACE_ROOT": directory,
                "MAX_OUTPUT_TOKENS": "100",
                "MAX_AGENT_ITERATIONS": "2",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONIOENCODING": "utf-8",
                "PYTHON_DOTENV_DISABLED": "1",
            })
            env.pop("PYTHONPATH", None)
            for command in ([sys.executable, "-B", "-m", "coding_agent_from_scratch"], [str(console)]):
                with self.subTest(command=command):
                    result = subprocess.run(
                        command, input="/help\n/clear\n/exit\n", text=True,
                        encoding="utf-8", capture_output=True, cwd=directory,
                        env=env, timeout=20,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("Coding agent", result.stdout)
                    self.assertIn("Conversation cleared", result.stdout)
                    self.assertIn("Goodbye.", result.stdout)
