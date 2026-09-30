"""Codebase Q&A agent, built for Amazon Bedrock AgentCore Runtime.

The agent is a Strands agent with two tools (search_code, read_file). AgentCore Runtime
hosts it behind an HTTP endpoint; BedrockAgentCoreApp wires the entrypoint to that contract.

Model choice (environment variables, read on every question):
    MODEL_PROVIDER=bedrock   (default) Amazon Bedrock; MODEL_ID picks the model, needs AWS credentials
    MODEL_PROVIDER=ollama    a local model served by Ollama; MODEL_ID picks it (default qwen2.5:7b)
    OLLAMA_HOST              Ollama address (default http://127.0.0.1:11434)

Run locally:   python agent.py            (serves the AgentCore contract on localhost:8080)
Demo UI:       python demo.py             (see README)
Deploy:        agentcore configure --entrypoint agent.py && agentcore launch
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent, tool

from agent_trace import extract_tool_trace
from retriever import CodeIndex

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).parent / "sample_repo"))
MAX_TOOL_CALLS = int(os.environ.get("MAX_TOOL_CALLS", "8"))
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"

index = CodeIndex(REPO_ROOT)
_calls = {"count": 0}
_run_lock = threading.Lock()


def _budget_exceeded() -> bool:
    """Guardrail: cap tool calls per question so a confused agent cannot loop or run up cost."""
    _calls["count"] += 1
    return _calls["count"] > MAX_TOOL_CALLS


@tool
def search_code(query: str, top_k: int = 5) -> str:
    """Search the repository and return the most relevant code snippets with file and line references.

    Args:
        query: Natural language or identifier keywords describing what to find.
        top_k: Number of snippets to return (1 to 8).
    """
    if _budget_exceeded():
        return "Tool call budget exhausted. Answer with what you have, or say the code does not contain it."
    top_k = max(1, min(int(top_k), 8))
    results = index.search(query, top_k=top_k)
    if not results:
        return "No matching code found."
    return "\n\n".join(f"[{chunk.ref}] (score {score:.2f})\n{chunk.text}" for score, chunk in results)


@tool
def read_file(path: str, start_line: int = 1, end_line: int = 80) -> str:
    """Read a line range from a file in the repository. Paths are relative to the repo root.

    Args:
        path: Relative file path, for example src/main/resources/application.yml.
        start_line: First line to read (1 based).
        end_line: Last line to read.
    """
    if _budget_exceeded():
        return "Tool call budget exhausted."
    try:
        return index.read_lines(path, int(start_line), min(int(end_line), int(start_line) + 120))
    except (ValueError, FileNotFoundError) as exc:
        return f"Cannot read file: {exc}"


SYSTEM_PROMPT = """You answer questions about a Java Spring Boot codebase.
Rules:
1. Always call search_code before answering. Use read_file when a snippet is cut off.
2. Answer only from code you retrieved. Cite every claim as path:start-end.
3. If the code does not contain the answer, say exactly: "Not found in the code." Never guess.
4. You are read only. Never claim to have changed code."""


def model_settings() -> dict[str, str]:
    """The provider and model the next question will use, from the environment."""
    provider = os.environ.get("MODEL_PROVIDER", "bedrock").strip().lower() or "bedrock"
    if provider not in {"bedrock", "ollama"}:
        raise ValueError(f"MODEL_PROVIDER must be bedrock or ollama, not {provider!r}")
    model_id = os.environ.get("MODEL_ID", "").strip()
    if provider == "ollama":
        return {
            "provider": provider,
            "model_id": model_id or DEFAULT_OLLAMA_MODEL,
            "host": os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
        }
    return {"provider": provider, "model_id": model_id or "Strands default Bedrock model"}


def build_model() -> Any:
    """A Strands model object for the configured provider, or None for the Strands Bedrock default."""
    settings = model_settings()
    if settings["provider"] == "ollama":
        from strands.models.ollama import OllamaModel

        return OllamaModel(host=settings["host"], model_id=settings["model_id"], temperature=0)
    if os.environ.get("MODEL_ID", "").strip():
        from strands.models import BedrockModel

        return BedrockModel(model_id=settings["model_id"])
    return None


def create_agent() -> Agent:
    """A fresh agent, so every question starts without earlier conversation."""
    kwargs: dict[str, Any] = {
        "system_prompt": SYSTEM_PROMPT,
        "tools": [search_code, read_file],
        "callback_handler": None,
    }
    model = build_model()
    if model is not None:
        kwargs["model"] = model
    return Agent(**kwargs)


def ask_with_trace(question: str) -> dict[str, Any]:
    """Answer one question and report which tools the agent called and what they returned."""
    with _run_lock:  # the tool call budget is shared state, so run one question at a time
        _calls["count"] = 0
        agent = create_agent()
        result = agent(question)
        return {
            "answer": str(result).strip(),
            "tool_calls": extract_tool_trace(agent.messages),
            "tool_budget": MAX_TOOL_CALLS,
        }


def ask(question: str) -> str:
    return ask_with_trace(question)["answer"]


app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload: dict) -> dict:
    question = (payload or {}).get("prompt", "").strip()
    if not question:
        return {"error": "Send a JSON body like {\"prompt\": \"How are payment retries configured?\"}"}
    return {"result": ask(question)}


if __name__ == "__main__":
    app.run()
