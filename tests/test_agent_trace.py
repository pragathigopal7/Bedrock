"""Tests for the agent trace helpers. Standard library only."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_trace import MAX_OUTPUT_CHARS, extract_tool_trace, find_citations, is_refusal  # noqa: E402


class FindCitationsTests(unittest.TestCase):
    def test_finds_ranges_and_single_lines_once_each(self):
        text = (
            "Retries are set in src/main/resources/application.yml:21-38 and used by "
            "PaymentClient.java:18. See application.yml:21-38 again (src/main/resources/application.yml:21-38)."
        )
        refs = [c["ref"] for c in find_citations(text)]
        self.assertEqual(
            refs,
            ["src/main/resources/application.yml:21-38", "PaymentClient.java:18-18", "application.yml:21-38"],
        )

    def test_normalizes_windows_separators(self):
        self.assertEqual(find_citations(r"src\main\App.java:3-4")[0]["path"], "src/main/App.java")

    def test_ignores_text_without_citations(self):
        self.assertEqual(find_citations("Not found in the code."), [])
        self.assertEqual(find_citations(""), [])


class RefusalTests(unittest.TestCase):
    def test_detects_the_refusal_phrase(self):
        self.assertTrue(is_refusal("Not found in the code."))
        self.assertTrue(is_refusal("That is NOT FOUND IN THE CODE, sorry."))
        self.assertFalse(is_refusal("Retries use exponential backoff."))


class ExtractToolTraceTests(unittest.TestCase):
    def test_pairs_calls_with_results_in_order(self):
        messages = [
            {"role": "user", "content": [{"text": "How are retries configured?"}]},
            {
                "role": "assistant",
                "content": [
                    {"text": "Let me look."},
                    {"toolUse": {"toolUseId": "a", "name": "search_code", "input": {"query": "retry"}}},
                    {"toolUse": {"toolUseId": "b", "name": "read_file", "input": {"path": "x.yml"}}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"toolResult": {"toolUseId": "b", "status": "error", "content": [{"text": "Cannot read file"}]}},
                    {"toolResult": {"toolUseId": "a", "status": "success", "content": [{"text": "[x.yml:1-3] ..."}]}},
                ],
            },
            {"role": "assistant", "content": [{"text": "Answer."}]},
        ]
        trace = extract_tool_trace(messages)
        self.assertEqual([t["name"] for t in trace], ["search_code", "read_file"])
        self.assertEqual(trace[0]["status"], "success")
        self.assertEqual(trace[0]["output"], "[x.yml:1-3] ...")
        self.assertEqual(trace[1]["status"], "error")
        self.assertEqual(trace[1]["input"], {"path": "x.yml"})

    def test_long_outputs_are_truncated(self):
        messages = [
            {"role": "assistant", "content": [{"toolUse": {"toolUseId": "a", "name": "search_code", "input": {}}}]},
            {"role": "user", "content": [{"toolResult": {"toolUseId": "a", "content": [{"text": "x" * 10_000}]}}]},
        ]
        output = extract_tool_trace(messages)[0]["output"]
        self.assertTrue(output.endswith("(truncated)"))
        self.assertLess(len(output), MAX_OUTPUT_CHARS + 50)

    def test_call_without_result_stays_pending(self):
        messages = [{"role": "assistant", "content": [{"toolUse": {"toolUseId": "a", "name": "search_code", "input": {}}}]}]
        self.assertEqual(extract_tool_trace(messages)[0]["status"], "pending")


if __name__ == "__main__":
    unittest.main()
