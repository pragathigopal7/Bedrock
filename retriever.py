"""Dependency free code retriever: method aware chunking + BM25 ranking.

Kept deliberately simple so the retrieval layer can be evaluated and swapped later
(for example for Amazon Titan embeddings plus a vector store) without touching the agent.
"""

from __future__ import annotations

import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

CODE_SUFFIXES = {".java", ".py", ".go", ".yml", ".yaml", ".properties", ".xml", ".md"}
METHOD_RE = re.compile(
    r"^\s*(?:@\w+(?:\([^)]*\))?\s*)*(?:public|private|protected)\s+[\w<>\[\],.? ]+\s+\w+\s*\([^;]*$"
)
MAX_CHUNK_LINES = 40
YAML_SUFFIXES = {".yml", ".yaml"}
YAML_SECTION_RE = re.compile(r"^[A-Za-z_][\w.-]*:\s*(?:#.*)?$")


@dataclass
class Chunk:
    path: str
    start: int
    end: int
    text: str

    @property
    def ref(self) -> str:
        return f"{self.path}:{self.start}-{self.end}"


def tokenize(text: str) -> list[str]:
    """Split identifiers on camelCase, snake_case and punctuation, lowercase everything."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return [t for t in re.split(r"[^A-Za-z0-9]+", text.lower()) if len(t) > 1]


def _chunk_file(rel_path: str, lines: list[str]) -> list[Chunk]:
    """Start a new chunk at each method signature (or YAML top level section); cap chunk length."""
    starts = [0]
    is_yaml = Path(rel_path).suffix in YAML_SUFFIXES
    for i, line in enumerate(lines):
        if is_yaml:
            # one chunk per top level section, so each config block is scored on its own terms
            if i > 0 and YAML_SECTION_RE.match(line):
                starts.append(i)
            continue
        if i > 0 and METHOD_RE.match(line):
            # pull annotations sitting directly above the signature into this chunk
            j = i
            while j > 0 and lines[j - 1].strip().startswith("@"):
                j -= 1
            starts.append(j)
    starts = sorted(set(starts))
    bounds = starts + [len(lines)]
    chunks: list[Chunk] = []
    for a, b in zip(bounds, bounds[1:]):
        for s in range(a, b, MAX_CHUNK_LINES):
            e = min(s + MAX_CHUNK_LINES, b)
            body = "\n".join(lines[s:e]).strip()
            if body:
                chunks.append(Chunk(rel_path, s + 1, e, body))
    return chunks


class CodeIndex:
    def __init__(self, repo_root: str | Path, k1: float = 1.5, b: float = 0.75):
        self.root = Path(repo_root).resolve()
        self.k1, self.b = k1, b
        self.chunks: list[Chunk] = []
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.suffix in CODE_SUFFIXES:
                rel = path.relative_to(self.root).as_posix()
                lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
                self.chunks.extend(_chunk_file(rel, lines))
        # the file path is searchable too, so "OrderService" finds OrderService.java
        self.doc_tokens = [tokenize(c.path + " " + c.text) for c in self.chunks]
        self.doc_freq: Counter[str] = Counter()
        for toks in self.doc_tokens:
            self.doc_freq.update(set(toks))
        self.avg_len = sum(len(t) for t in self.doc_tokens) / max(len(self.doc_tokens), 1)

    def search(self, query: str, top_k: int = 5) -> list[tuple[float, Chunk]]:
        q_tokens = tokenize(query)
        n = len(self.chunks)
        scored: list[tuple[float, Chunk]] = []
        for toks, chunk in zip(self.doc_tokens, self.chunks):
            tf = Counter(toks)
            score = 0.0
            for term in q_tokens:
                if term not in tf:
                    continue
                idf = math.log(1 + (n - self.doc_freq[term] + 0.5) / (self.doc_freq[term] + 0.5))
                denom = tf[term] + self.k1 * (1 - self.b + self.b * len(toks) / self.avg_len)
                score += idf * tf[term] * (self.k1 + 1) / denom
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def read_lines(self, rel_path: str, start: int, end: int) -> str:
        """Read a line range, refusing anything outside the repo root."""
        target = (self.root / rel_path).resolve()
        if self.root not in target.parents and target != self.root:
            raise ValueError("path is outside the repository")
        if not target.is_file():
            raise FileNotFoundError(rel_path)
        lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
        start = max(1, start)
        end = min(len(lines), max(start, end))
        return "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))


if __name__ == "__main__":
    root = Path(__file__).parent / "sample_repo"
    index = CodeIndex(root)
    question = " ".join(sys.argv[1:]) or "how are payment retries configured"
    for score, chunk in index.search(question):
        print(f"{score:6.2f}  {chunk.ref}")
