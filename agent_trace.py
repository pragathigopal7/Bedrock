"""Helpers for inspecting an agent run. Standard library only, so they are unit testable
without the Strands or AgentCore packages installed.
"""

from __future__ import annotations

import re
from typing import Any

# path:start or path:start-end, for the file types the retriever indexes
CITATION_RE = re.compile(
    r"(?P<path>[\w./\\-]+\.(?:java|py|go|ya?ml|properties|xml|md)):(?P<start>\d+)(?:-(?P<end>\d+))?"
)

REFUSAL_TEXT = "not found in the code"

MAX_OUTPUT_CHARS = 4000


def find_citations(text: str) -> list[dict[str, Any]]:
    """Every distinct path:start-end reference in the text, in order of appearance."""
    seen: set[tuple[str, int, int]] = set()
    citations: list[dict[str, Any]] = []
    for match in CITATION_RE.finditer(text or ""):
        path = match.group("path").replace("\\", "/")
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        key = (path, start, end)
        if key not in seen:
            seen.add(key)
            citations.append({"path": path, "start": start, "end": end, "ref": f"{path}:{start}-{end}"})
    return citations


def is_refusal(text: str) -> bool:
    return REFUSAL_TEXT in (text or "").lower()


def _result_text(tool_result: dict[str, Any]) -> str:
    parts = []
    for item in tool_result.get("content") or []:
        if "text" in item:
            parts.append(str(item["text"]))
        elif "json" in item:
            parts.append(str(item["json"]))
    text = "\n".join(parts)
    if len(text) > MAX_OUTPUT_CHARS:
        text = text[:MAX_OUTPUT_CHARS] + "\n... (truncated)"
    return text


def extract_tool_trace(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair each toolUse block with its toolResult, in the order the agent called them.

    Works on Strands conversation messages, which use Bedrock Converse style content blocks:
    {"toolUse": {"toolUseId", "name", "input"}} and {"toolResult": {"toolUseId", "status", "content"}}.
    """
    calls: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for message in messages or []:
        for block in message.get("content") or []:
            if "toolUse" in block:
                use = block["toolUse"]
                call = {"name": use.get("name"), "input": use.get("input") or {}, "status": "pending", "output": ""}
                calls.append(call)
                if use.get("toolUseId"):
                    by_id[use["toolUseId"]] = call
            elif "toolResult" in block:
                result = block["toolResult"]
                call = by_id.get(result.get("toolUseId", ""))
                if call is not None:
                    call["status"] = result.get("status", "success")
                    call["output"] = _result_text(result)
    return calls
