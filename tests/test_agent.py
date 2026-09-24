import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from coding_agent_from_scratch import agent
from coding_agent_from_scratch.config import Config
from coding_agent_from_scratch.tools.registry import build_tool_registry


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.config = Config("offline-key", "offline-model", self.root, 100, 2)
        self.registry = build_tool_registry(self.config)

    def test_tool_result_is_linked_and_reasoning_preserved(self):
        reasoning = SimpleNamespace(type="reasoning")
        call = SimpleNamespace(
            type="function_call", name="list_files", arguments='{"path": "."}',
            call_id="call-1",
        )
        first = SimpleNamespace(status="completed", output=[reasoning, call])
        final = SimpleNamespace(status="completed", output=[], output_text="Done")
        client = Mock()

        def respond(**kwargs):
            history = kwargs["input"]
            if client.responses.create.call_count == 1:
                return first
            self.assertIs(history[1], reasoning)
            self.assertIs(history[2], call)
            self.assertEqual(history[3]["type"], "function_call_output")
            self.assertEqual(history[3]["call_id"], "call-1")
            self.assertTrue(json.loads(history[3]["output"])["ok"])
            return final

        client.responses.create.side_effect = respond
        answer = agent.run_agent_turn(client, self.config, [], self.registry, "List files")
        self.assertEqual(answer, "Done")
        self.assertEqual(client.responses.create.call_count, 2)

    def test_iteration_limit_stops_repeated_tool_requests(self):
        client = Mock()
        client.responses.create.side_effect = [
            SimpleNamespace(status="completed", output=[SimpleNamespace(
                type="function_call", name="list_files", arguments='{"path": "."}',
                call_id=f"call-{index}",
            )]) for index in range(2)
        ]
        history = []
        with self.assertRaises(agent.AgentIterationLimitError):
            agent.run_agent_turn(client, self.config, history, self.registry, "List files")
        self.assertEqual(client.responses.create.call_count, 2)
        self.assertEqual(history[-1]["type"], "function_call_output")
        self.assertEqual(history[-1]["call_id"], "call-1")

    def test_incomplete_response_does_not_execute_tools(self):
        client = Mock()
        client.responses.create.return_value = SimpleNamespace(
            status="incomplete", incomplete_details="max_output_tokens", error=None,
            output=[SimpleNamespace(type="function_call")],
        )
        with patch.object(agent, "dispatch_tool_call") as dispatch:
            with self.assertRaises(RuntimeError):
                agent.run_agent_turn(client, self.config, [], self.registry, "Create file")
        dispatch.assert_not_called()
