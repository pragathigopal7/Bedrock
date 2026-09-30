"""Local demo for the codebase Q&A agent: a small web app on http://localhost:8765.

    python demo.py                                   # offline: retrieval and citations, no model, no AWS
    python demo.py --llm ollama --model qwen2.5:7b   # full agent with a free local model through Ollama
    python demo.py --llm bedrock --model <model id>  # full agent on Amazon Bedrock (needs AWS credentials)

Offline mode needs only the Python standard library. The server listens on 127.0.0.1 only,
so it is not reachable from other machines.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from agent_trace import find_citations, is_refusal  # noqa: E402
from retriever import CodeIndex  # noqa: E402
from run_eval import retrieval_eval  # noqa: E402

WEB_PAGE = ROOT / "web" / "index.html"
GOLDEN_SET = ROOT / "eval" / "golden_set.json"
MAX_BODY_BYTES = 16_000
MAX_QUESTION_CHARS = 500


class DemoError(Exception):
    def __init__(self, status: HTTPStatus, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class Demo:
    """Everything the web page can ask for. Agent mode loads agent.py lazily."""

    def __init__(self, repo_root: str | Path, llm: str | None = None):
        self.repo_root = Path(repo_root).resolve()
        self.index = CodeIndex(self.repo_root)
        self.llm = llm
        self.cases = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))
        self._agent_module: Any = None

    # --- agent -----------------------------------------------------------------------
    def agent_module(self) -> Any:
        if self._agent_module is None:
            import agent  # needs strands-agents and bedrock-agentcore, plus the ollama extra for Ollama

            self._agent_module = agent
        return self._agent_module

    def model_settings(self) -> dict[str, str] | None:
        return self.agent_module().model_settings() if self.llm else None

    # --- API ------------------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        return {
            "mode": "agent" if self.llm else "offline",
            "model": self.model_settings(),
            "repo": self.repo_root.name,
            "files": len(self.index.files),
            "chunks": len(self.index.chunks),
            "tool_budget": int(os.environ.get("MAX_TOOL_CALLS", "8")),
            "samples": [
                {"id": c["id"], "question": c["question"], "answerable": bool(c["expected_files"])}
                for c in self.cases
            ],
        }

    def evidence(self, question: str, top_k: int = 5) -> list[dict[str, Any]]:
        """What search_code would return, with the exact source lines so numbering and indentation are right."""
        return [
            {
                "ref": chunk.ref,
                "path": chunk.path,
                "start": chunk.start,
                "end": chunk.end,
                "score": round(score, 2),
                "lines": self.index.file_lines(chunk.path)[chunk.start - 1 : chunk.end],
            }
            for score, chunk in self.index.search(question, top_k=top_k)
        ]

    def resolve_path(self, path: str) -> str:
        """Accept a full repo relative path, or a bare file name when it is unique in the repo."""
        path = path.replace("\\", "/")
        while path.startswith("./"):
            path = path[2:]
        if path in self.index.files:
            return path
        matches = [f for f in self.index.files if f.endswith("/" + path)]
        if len(matches) == 1:
            return matches[0]
        return path

    def check_citations(self, citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Mark each citation valid only if the file exists in the repo and the lines are in range."""
        checked = []
        for c in citations:
            path = self.resolve_path(c["path"])
            try:
                n_lines = len(self.index.file_lines(path))
                valid = 1 <= c["start"] <= c["end"] <= n_lines
            except (ValueError, FileNotFoundError):
                valid = False
            checked.append({**c, "path": path, "ref": f"{path}:{c['start']}-{c['end']}", "valid": valid})
        return checked

    def ask(self, question: str) -> dict[str, Any]:
        question = (question or "").strip()
        if not question:
            raise DemoError(HTTPStatus.BAD_REQUEST, "Type a question first.")
        if len(question) > MAX_QUESTION_CHARS:
            raise DemoError(HTTPStatus.BAD_REQUEST, f"Keep questions under {MAX_QUESTION_CHARS} characters.")
        started = time.perf_counter()

        if not self.llm:
            return {
                "mode": "offline",
                "question": question,
                "evidence": self.evidence(question),
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
            }

        try:
            run = self.agent_module().ask_with_trace(question)
        except Exception as exc:  # surface model and credential problems to the page
            raise DemoError(HTTPStatus.BAD_GATEWAY, model_error_hint(self.llm, exc)) from exc
        answer = run["answer"]
        return {
            "mode": "agent",
            "question": question,
            "answer": answer,
            "refused": is_refusal(answer),
            "citations": self.check_citations(find_citations(answer)),
            "tool_calls": run["tool_calls"],
            "tool_budget": run["tool_budget"],
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        }

    def evaluate(self, k: int = 3) -> dict[str, Any]:
        result = retrieval_eval(self.index, self.cases, k)
        expected = {c["id"]: c["expected_files"] for c in self.cases}
        rows = [
            {"id": qid, "question": question, "expected": expected[qid], "rank": rank, "hit": hit}
            for qid, hit, rank, question in result["rows"]
        ]
        return {
            "k": result["k"],
            "n": result["n"],
            "recall_at_k": result["recall_at_k"],
            "mrr": result["mrr"],
            "ranked_first": sum(1 for r in rows if r["rank"] == 1),
            "rows": rows,
        }

    def file(self, path: str) -> dict[str, Any]:
        resolved = self.resolve_path(path or "")
        try:
            lines = self.index.file_lines(resolved)
        except ValueError as exc:
            raise DemoError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        except FileNotFoundError as exc:
            raise DemoError(HTTPStatus.NOT_FOUND, f"No such file in the repo: {path}") from exc
        return {"path": resolved, "lines": lines}


def model_error_hint(llm: str, exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if llm == "ollama":
        return (
            f"The local model call failed ({text}). Check that Ollama is running and the model is "
            "pulled, for example: ollama pull qwen2.5:7b"
        )
    return f"The Bedrock call failed ({text}). Check your AWS credentials, region, and model access."


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows, SO_REUSEADDR lets a second server share a busy port silently, so only use it elsewhere
    allow_reuse_address = sys.platform != "win32"

    def __init__(self, address: tuple[str, int], demo: Demo):
        super().__init__(address, DemoHandler)
        self.demo = demo


class DemoHandler(BaseHTTPRequestHandler):
    server: DemoServer

    def log_message(self, format: str, *args: Any) -> None:  # keep the console quiet
        pass

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _handle(self, action) -> None:
        try:
            self._json(HTTPStatus.OK, action())
        except DemoError as err:
            self._json(err.status, {"error": err.message})
        except Exception as exc:  # never leave the page hanging
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(exc).__name__}: {exc}"})

    def do_GET(self) -> None:
        url = urlparse(self.path)
        demo = self.server.demo
        if url.path in ("/", "/index.html"):
            self._send(HTTPStatus.OK, WEB_PAGE.read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/api/status":
            self._handle(demo.status)
        elif url.path == "/api/eval":
            self._handle(demo.evaluate)
        elif url.path == "/api/file":
            path = parse_qs(url.query).get("path", [""])[0]
            self._handle(lambda: demo.file(path))
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/ask":
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        # JSON only: browsers must preflight cross site JSON requests, which this server never approves
        if not self.headers.get("Content-Type", "").startswith("application/json"):
            self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "Send JSON."})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Request too large."})
            return
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON."})
            return
        question = payload.get("question", "") if isinstance(payload, dict) else ""

        def run() -> dict[str, Any]:
            result = self.server.demo.ask(str(question))
            print(f"[{result['mode']}] {result['elapsed_ms']:>6} ms  {result['question']}")
            return result

        self._handle(run)


def check_ollama(host: str, model_id: str) -> str | None:
    """A warning to print if Ollama is not reachable or the model is not pulled, else None."""
    try:
        with urlopen(host.rstrip("/") + "/api/tags", timeout=3) as response:
            names = {m.get("name", "") for m in json.load(response).get("models", [])}
    except (URLError, OSError, ValueError):
        return f"Ollama is not reachable at {host}. Start the Ollama app, then ask again."
    wanted = model_id if ":" in model_id else model_id + ":latest"
    if wanted not in names:
        return f"Model {model_id} is not pulled yet. Run: ollama pull {model_id}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the codebase Q&A agent demo in your browser.")
    parser.add_argument("--llm", choices=["ollama", "bedrock"], help="connect a model for full agent answers")
    parser.add_argument("--model", help="model id, for example qwen2.5:7b for Ollama")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--repo", help="repository to answer questions about (default: sample_repo)")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):  # never crash on a character the console cannot print
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    if args.repo:
        os.environ["REPO_ROOT"] = str(Path(args.repo).resolve())
    if args.llm:
        os.environ["MODEL_PROVIDER"] = args.llm
    if args.model:
        os.environ["MODEL_ID"] = args.model

    demo = Demo(os.environ.get("REPO_ROOT", ROOT / "sample_repo"), llm=args.llm)

    if args.llm:
        try:
            settings = demo.model_settings()
        except ImportError as exc:
            extra = ' "strands-agents[ollama]"' if args.llm == "ollama" else ""
            sys.exit(f"Agent mode needs the agent packages ({exc}). Run: pip install -r requirements.txt{extra}")
        if args.llm == "ollama":
            warning = check_ollama(settings["host"], settings["model_id"])
            if warning:
                print("WARNING: " + warning)

    try:
        server = DemoServer(("127.0.0.1", args.port), demo)
    except OSError:
        sys.exit(f"Port {args.port} is busy. Try: python demo.py --port {args.port + 1}")

    url = f"http://localhost:{args.port}"
    mode = f"agent ({settings['provider']}, {settings['model_id']})" if args.llm else "offline (retrieval only)"
    print(f"Codebase Q&A demo running at {url}")
    print(f"Mode: {mode}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
