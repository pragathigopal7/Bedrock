"""Runs the real Strands agent loop against a fake Ollama server, so no model or AWS account is needed.

The fake server speaks Ollama's streaming /api/chat protocol: on the first call it asks for the
search_code tool, and once a tool result comes back it answers with a citation copied from that
result. This checks the provider wiring, tool execution, the tool trace, fresh state per question,
and the demo's citation check.

Skipped when strands-agents (with the ollama extra) or bedrock-agentcore is not installed.
Run from the repository root:
    python -m unittest tests.test_agent_loop -v
"""

import importlib.util
import json
import os
import re
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HAVE_AGENT_DEPS = all(importlib.util.find_spec(name) for name in ("strands", "ollama", "bedrock_agentcore"))
REF_RE = re.compile(r"\[([^\]\s]+?:\d+-\d+)\]")


class FakeOllama(BaseHTTPRequestHandler):
    requests: list = []

    def log_message(self, format, *args):
        pass

    def _send(self, payload: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send(json.dumps({"models": [{"name": "fake-model:latest"}]}).encode(), "application/json")
        else:
            self.send_error(404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append(body)
        messages = body["messages"]
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        if tool_messages:
            match = REF_RE.search(tool_messages[-1].get("content", ""))
            cite = match.group(1) if match else "no citation"
            message = {"role": "assistant", "content": f"According to the retrieved code ({cite}), here is the answer."}
        else:
            question = [m for m in messages if m.get("role") == "user"][-1]["content"]
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "search_code", "arguments": {"query": question}}}],
            }
        stamp = {"model": body["model"], "created_at": "2026-01-01T00:00:00Z"}
        chunks = [
            {**stamp, "message": message, "done": False},
            {
                **stamp,
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
                "total_duration": 1_000_000,
                "load_duration": 0,
                "prompt_eval_count": 10,
                "prompt_eval_duration": 0,
                "eval_count": 5,
                "eval_duration": 0,
            },
        ]
        self._send("".join(json.dumps(c) + "\n" for c in chunks).encode(), "application/x-ndjson")


@unittest.skipUnless(HAVE_AGENT_DEPS, "needs strands-agents[ollama] and bedrock-agentcore")
class AgentLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.saved_env = {k: os.environ.get(k) for k in ("MODEL_PROVIDER", "MODEL_ID", "OLLAMA_HOST")}
        os.environ.update(
            MODEL_PROVIDER="ollama",
            MODEL_ID="fake-model",
            OLLAMA_HOST=f"http://127.0.0.1:{cls.server.server_port}",
        )
        import agent

        cls.agent = agent

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        for key, value in cls.saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def setUp(self):
        FakeOllama.requests.clear()

    def test_agent_searches_then_answers_with_a_citation(self):
        run = self.agent.ask_with_trace("Can a shipped order be cancelled?")
        self.assertEqual(len(FakeOllama.requests), 2)  # one call asks for the tool, one writes the answer
        self.assertEqual([call["name"] for call in run["tool_calls"]], ["search_code"])
        self.assertEqual(run["tool_calls"][0]["input"], {"query": "Can a shipped order be cancelled?"})
        self.assertEqual(run["tool_calls"][0]["status"], "success")
        self.assertIn("OrderService.java", run["tool_calls"][0]["output"])
        self.assertIn("OrderService.java:", run["answer"])

    def test_tools_are_offered_to_the_model(self):
        self.agent.ask_with_trace("How long are idempotency keys kept?")
        offered = {tool["function"]["name"] for tool in FakeOllama.requests[0]["tools"]}
        self.assertEqual(offered, {"search_code", "read_file"})
        self.assertEqual(FakeOllama.requests[0]["messages"][0]["role"], "system")

    def test_each_question_starts_without_earlier_conversation(self):
        self.agent.ask_with_trace("How long are idempotency keys kept?")
        self.agent.ask_with_trace("Can a shipped order be cancelled?")
        first_call_of_second_question = FakeOllama.requests[2]["messages"]
        self.assertEqual([m["role"] for m in first_call_of_second_question], ["system", "user"])

    def test_demo_verifies_the_agent_citations(self):
        import demo

        result = demo.Demo(ROOT / "sample_repo", llm="ollama").ask("Can a shipped order be cancelled?")
        self.assertEqual(result["mode"], "agent")
        self.assertFalse(result["refused"])
        self.assertTrue(result["citations"])
        self.assertTrue(all(c["valid"] for c in result["citations"]))
        self.assertEqual(result["tool_budget"], 8)


if __name__ == "__main__":
    unittest.main()
