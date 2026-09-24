import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from coding_agent_from_scratch.config import load_config


class ConfigTests(unittest.TestCase):
    def test_loads_env_from_launch_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / ".env").write_text(
                "OPENAI_API_KEY=offline-key\nOPENAI_MODEL=file-model\n"
                f"WORKSPACE_ROOT={root.as_posix()}\n", encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True), patch(
                "coding_agent_from_scratch.config.Path.cwd", return_value=root
            ):
                config = load_config()
            self.assertEqual(config.api_key, "offline-key")
            self.assertEqual(config.model, "file-model")
            self.assertEqual(config.workspace_root, root)

    def test_exported_variables_override_env_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / ".env").write_text("OPENAI_MODEL=file-model\n", encoding="utf-8")
            env = {"OPENAI_API_KEY": "offline-key", "OPENAI_MODEL": "exported-model", "WORKSPACE_ROOT": str(root)}
            with patch.dict(os.environ, env, clear=True), patch(
                "coding_agent_from_scratch.config.Path.cwd", return_value=root
            ):
                self.assertEqual(load_config().model, "exported-model")
