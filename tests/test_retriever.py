"""Unit tests for the retriever. Standard library only.

Run from the repository root:
    python -m unittest discover tests -v
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from retriever import CodeIndex, tokenize  # noqa: E402
from run_eval import retrieval_eval  # noqa: E402
import json  # noqa: E402

SAMPLE_REPO = ROOT / "sample_repo"


class TokenizeTests(unittest.TestCase):
    def test_splits_camel_case_and_punctuation(self):
        tokens = tokenize("OrderService.createOrder")
        self.assertEqual(tokens, ["order", "service", "create", "order"])

    def test_splits_snake_and_kebab_case(self):
        self.assertEqual(tokenize("order-events_dlq"), ["order", "events", "dlq"])

    def test_drops_single_characters(self):
        self.assertEqual(tokenize("a b cd"), ["cd"])


class ChunkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = CodeIndex(SAMPLE_REPO)

    def test_every_java_and_yaml_file_is_indexed(self):
        paths = {chunk.path for chunk in self.index.chunks}
        self.assertTrue(any(p.endswith("OrderService.java") for p in paths))
        self.assertTrue(any(p.endswith("application.yml") for p in paths))

    def test_annotations_stay_with_their_method(self):
        # @Transactional sits directly above cancelOrder and must be in the same chunk.
        chunk = next(c for c in self.index.chunks if "public void cancelOrder" in c.text)
        self.assertIn("@Transactional", chunk.text)

    def test_chunks_have_valid_line_ranges(self):
        for chunk in self.index.chunks:
            self.assertGreaterEqual(chunk.start, 1)
            self.assertGreaterEqual(chunk.end, chunk.start)


class SearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = CodeIndex(SAMPLE_REPO)

    def test_finds_the_right_file(self):
        results = self.index.search("Can a shipped order be cancelled?", top_k=3)
        self.assertTrue(results[0][1].path.endswith("OrderService.java"))

    def test_results_are_sorted_by_score(self):
        scores = [score for score, _ in self.index.search("payment retry circuit breaker", top_k=5)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_respects_top_k(self):
        self.assertLessEqual(len(self.index.search("order", top_k=2)), 2)

    def test_unknown_terms_return_nothing(self):
        self.assertEqual(self.index.search("zzzzqqqq"), [])


class ReadLinesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = CodeIndex(SAMPLE_REPO)

    def test_returns_numbered_lines(self):
        text = self.index.read_lines("src/main/resources/application.yml", 1, 2)
        self.assertEqual(text.splitlines()[0], "1: server:")

    def test_blocks_path_traversal(self):
        with self.assertRaises(ValueError):
            self.index.read_lines("../README.md", 1, 5)

    def test_blocks_absolute_paths_outside_repo(self):
        with self.assertRaises(ValueError):
            self.index.read_lines("/etc/passwd", 1, 5)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.index.read_lines("src/does_not_exist.java", 1, 5)

    def test_file_lines_match_numbered_read(self):
        lines = self.index.file_lines("src/main/resources/application.yml")
        numbered = self.index.read_lines("src/main/resources/application.yml", 1, len(lines))
        self.assertEqual([row.split(": ", 1)[1] if ": " in row else "" for row in numbered.splitlines()][:3], lines[:3])

    def test_file_lines_share_the_path_guard(self):
        with self.assertRaises(ValueError):
            self.index.file_lines("../README.md")

    def test_files_lists_every_indexed_file(self):
        self.assertIn("src/main/resources/application.yml", self.index.files)
        self.assertEqual(len(self.index.files), len(set(self.index.files)))


class GoldenSetTests(unittest.TestCase):
    def test_recall_at_3_stays_high(self):
        cases = json.loads((ROOT / "eval" / "golden_set.json").read_text(encoding="utf-8"))
        result = retrieval_eval(CodeIndex(SAMPLE_REPO), cases, k=3)
        self.assertGreaterEqual(result["recall_at_k"], 0.9)


if __name__ == "__main__":
    unittest.main()
