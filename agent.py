"""Codebase Q&A agent, hosted on Amazon Bedrock AgentCore Runtime.

The agent is a Strands agent with two tools (search_code, read_file). AgentCore Runtime
hosts it behind an HTTP endpoint; BedrockAgentCoreApp wires the entrypoint to that contract.

Run locally:   python agent.py            (serves on localhost:8080)
Deploy:        agentcore configure --entrypoint agent.py && agentcore launch
"""

from __future__ import annotations

import os
from pathlib import Path

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent, tool

from retriever import CodeIndex

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).parent / "sample_repo"))
MAX_TOOL_CALLS = int(os.environ.get("MAX_TOOL_CALLS", "8"))
MODEL_ID = os.environ.get("MODEL_ID")  # set to a Bedrock model your account has enabled

index = CodeIndex(REPO_ROOT)
_calls = {"count": 0}


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

agent_kwargs = {"system_prompt": SYSTEM_PROMPT, "tools": [search_code, read_file]}
if MODEL_ID:
    from strands.models import BedrockModel

    agent_kwargs["model"] = BedrockModel(model_id=MODEL_ID)

agent = Agent(**agent_kwargs)
app = BedrockAgentCoreApp()


def ask(question: str) -> str:
    _calls["count"] = 0
    return str(agent(question))


@app.entrypoint
def invoke(payload: dict) -> dict:
    question = (payload or {}).get("prompt", "").strip()
    if not question:
        return {"error": "Send a JSON body like {\"prompt\": \"How are payment retries configured?\"}"}
    return {"result": ask(question)}


if __name__ == "__main__":
    app.run()
