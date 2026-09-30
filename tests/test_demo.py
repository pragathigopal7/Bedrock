"""Tests for the local demo server in offline mode. Standard library only.

Run from the repository root:
    python -m unittest discover tests -v
"""

import json
import sys
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import demo  # noqa: E402


class DemoApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = demo.Demo(ROOT / "sample_repo")
        cls.server = demo.DemoServer(("127.0.0.1", 0), cls.demo)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def request(self, method, path, body=None, content_type="application/json"):
        conn = HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
        headers = {"Content-Type": content_type} if body is not None else {}
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response.status, response.getheader("Content-Type"), data

    def get_json(self, path):
        status, _, data = self.request("GET", path)
        return status, json.loads(data)

    def ask(self, question):
        status, _, data = self.request("POST", "/api/ask", json.dumps({"question": question}))
        return status, json.loads(data)

    def test_page_is_served(self):
        status, content_type, data = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertTrue(content_type.startswith("text/html"))
        self.assertIn(b"Codebase Q&amp;A Agent", data)

    def test_status_reports_offline_mode_and_samples(self):
        status, body = self.get_json("/api/status")
        self.assertEqual(status, 200)
        self.assertEqual(body["mode"], "offline")
        self.assertIsNone(body["model"])
        self.assertEqual(len(body["samples"]), 11)
        self.assertEqual([s["id"] for s in body["samples"] if not s["answerable"]], ["q11"])

    def test_ask_returns_ranked_evidence_with_exact_source_lines(self):
        status, body = self.ask("Can a shipped order be cancelled?")
        self.assertEqual(status, 200)
        self.assertEqual(body["mode"], "offline")
        top = body["evidence"][0]
        self.assertTrue(top["path"].endswith("OrderService.java"))
        file_lines = self.demo.index.file_lines(top["path"])
        self.assertEqual(top["lines"], file_lines[top["start"] - 1 : top["end"]])
        scores = [e["score"] for e in body["evidence"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_ask_rejects_empty_and_overlong_questions(self):
        self.assertEqual(self.ask("   ")[0], 400)
        self.assertEqual(self.ask("x" * 501)[0], 400)

    def test_ask_requires_json(self):
        status, _, _ = self.request("POST", "/api/ask", "question=hi", content_type="application/x-www-form-urlencoded")
        self.assertEqual(status, 415)
        status, _, _ = self.request("POST", "/api/ask", "{not json")
        self.assertEqual(status, 400)

    def test_file_resolves_a_unique_file_name(self):
        status, body = self.get_json("/api/file?path=PaymentClient.java")
        self.assertEqual(status, 200)
        self.assertEqual(body["path"], "src/main/java/com/example/orders/PaymentClient.java")
        self.assertTrue(body["lines"][0].startswith("package "))

    def test_file_blocks_paths_outside_the_repo(self):
        self.assertEqual(self.get_json("/api/file?path=../README.md")[0], 400)
        self.assertEqual(self.get_json("/api/file?path=/etc/passwd")[0], 400)

    def test_missing_file_is_404(self):
        self.assertEqual(self.get_json("/api/file?path=src/Nope.java")[0], 404)

    def test_eval_scores_the_golden_set(self):
        status, body = self.get_json("/api/eval")
        self.assertEqual(status, 200)
        self.assertEqual(body["n"], 10)
        self.assertEqual(body["recall_at_k"], 1.0)
        self.assertGreaterEqual(body["mrr"], 0.9)
        self.assertEqual(len(body["rows"]), 10)

    def test_unknown_routes_are_404(self):
        self.assertEqual(self.get_json("/api/nope")[0], 404)
        status, _, _ = self.request("POST", "/api/nope", "{}")
        self.assertEqual(status, 404)


class CitationCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = demo.Demo(ROOT / "sample_repo")

    def check(self, path, start, end):
        return self.demo.check_citations([{"path": path, "start": start, "end": end}])[0]

    def test_real_lines_are_valid(self):
        self.assertTrue(self.check("src/main/resources/application.yml", 21, 38)["valid"])

    def test_file_name_alone_resolves(self):
        result = self.check("OrderService.java", 1, 5)
        self.assertTrue(result["valid"])
        self.assertEqual(result["path"], "src/main/java/com/example/orders/OrderService.java")

    def test_out_of_range_lines_are_invalid(self):
        self.assertFalse(self.check("OrderService.java", 50, 9999)["valid"])
        self.assertFalse(self.check("OrderService.java", 20, 10)["valid"])

    def test_missing_and_outside_files_are_invalid(self):
        self.assertFalse(self.check("src/GraphQlResolver.java", 1, 5)["valid"])
        self.assertFalse(self.check("../README.md", 1, 5)["valid"])


if __name__ == "__main__":
    unittest.main()
