"""
ouro/engine/tool_parser.py — Parse tool/function calls from raw LLM output.

Supports three detection patterns:
  1. <tool_call>{...}</tool_call> XML tags
  2. Bare JSON blobs matching {"name": ..., "arguments": ...} or
     {"tool": ..., "parameters": ...}
  3. Markdown code fences: ```json {...} ```

Returns OpenAI-format tool_calls or None if no calls are detected.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

log = logging.getLogger("ouro.engine.tool_parser")

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# 1. <tool_call> … </tool_call>
_RE_XML_TOOL_CALL = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)

# 2. Bare JSON object (greedy, handles nested braces via post-processing)
#    We look for opening { at the start of a candidate and extract balanced JSON.
_RE_BARE_JSON = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", re.DOTALL)

# 3. Markdown fenced JSON block
_RE_FENCE_JSON = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_id() -> str:
    """Generate a unique call ID in the format ``call_<hex>``."""
    return f"call_{uuid.uuid4().hex[:24]}"


def _extract_balanced_json(text: str, start: int) -> Optional[str]:
    """
    Extract a balanced JSON object starting at *start* in *text*.

    Returns the raw JSON string or ``None`` if extraction fails.
    """
    depth = 0
    in_string = False
    escape_next = False
    i = start

    while i < len(text):
        ch = text[i]
        if escape_next:
            escape_next = False
            i += 1
            continue
        if ch == "\\" and in_string:
            escape_next = True
            i += 1
            continue
        if ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        i += 1
    return None


def _to_openai_tool_call(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Normalise a raw parsed dict to an OpenAI tool_call object.

    Handles two common schemas:
    - ``{"name": ..., "arguments": ...}``
    - ``{"tool": ..., "parameters": ...}``

    Returns ``None`` if the dict doesn't match either schema.
    """
    name: Optional[str] = None
    arguments: Any = None

    if "name" in raw:
        name = raw["name"]
        arguments = raw.get("arguments", {})
    elif "tool" in raw:
        name = raw["tool"]
        arguments = raw.get("parameters", raw.get("arguments", {}))

    if not name or not isinstance(name, str):
        return None

    # Serialise arguments to a JSON string (OpenAI format)
    if isinstance(arguments, dict):
        arguments_str = json.dumps(arguments)
    elif isinstance(arguments, str):
        arguments_str = arguments
    else:
        arguments_str = json.dumps(arguments)

    return {
        "id": _make_id(),
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments_str,
        },
    }


def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Attempt to parse *text* as JSON, returning the dict or ``None``."""
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# Main extraction strategies
# ---------------------------------------------------------------------------


def _extract_xml_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extract tool calls from <tool_call>…</tool_call> tags."""
    results: List[Dict[str, Any]] = []
    for match in _RE_XML_TOOL_CALL.finditer(text):
        raw = _try_parse_json(match.group(1))
        if raw is None:
            log.debug("XML tool_call block contained invalid JSON: %s", match.group(1))
            continue
        call = _to_openai_tool_call(raw)
        if call:
            results.append(call)
    return results


def _extract_fence_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extract tool calls from ```json … ``` fenced code blocks."""
    results: List[Dict[str, Any]] = []
    for match in _RE_FENCE_JSON.finditer(text):
        raw = _try_parse_json(match.group(1))
        if raw is None:
            continue
        call = _to_openai_tool_call(raw)
        if call:
            results.append(call)
    return results


def _extract_bare_json_tool_calls(text: str) -> List[Dict[str, Any]]:
    """
    Scan *text* for bare JSON objects that look like tool calls.

    Uses a balanced-brace extractor to handle nested objects.
    """
    results: List[Dict[str, Any]] = []
    pos = 0
    while pos < len(text):
        idx = text.find("{", pos)
        if idx == -1:
            break
        blob = _extract_balanced_json(text, idx)
        if blob is None:
            pos = idx + 1
            continue
        raw = _try_parse_json(blob)
        if raw is not None:
            # Only treat as a tool call if it has the expected keys
            if ("name" in raw or "tool" in raw) and (
                "arguments" in raw or "parameters" in raw
            ):
                call = _to_openai_tool_call(raw)
                if call:
                    results.append(call)
        pos = idx + len(blob)
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_tool_calls(text: str) -> Optional[List[Dict[str, Any]]]:
    """
    Parse tool/function calls from the raw model output *text*.

    Detection is attempted in priority order:
      1. ``<tool_call>{...}</tool_call>`` XML tags
      2. Markdown ```json {...} ``` code fences
      3. Bare JSON objects matching the tool-call schema

    Parameters
    ----------
    text:
        Raw string output from the model.

    Returns
    -------
    list[dict] or None
        A list of OpenAI-format tool_call objects::

            [
                {
                    "id": "call_abc123...",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Paris"}',
                    },
                }
            ]

        Returns ``None`` if no tool calls were detected.
    """
    if not text or not text.strip():
        return None

    # Strategy 1: XML tags
    calls = _extract_xml_tool_calls(text)
    if calls:
        log.debug("Detected %d tool call(s) via XML tags", len(calls))
        return calls

    # Strategy 2: Markdown fences
    calls = _extract_fence_tool_calls(text)
    if calls:
        log.debug("Detected %d tool call(s) via markdown fences", len(calls))
        return calls

    # Strategy 3: Bare JSON
    calls = _extract_bare_json_tool_calls(text)
    if calls:
        log.debug("Detected %d tool call(s) via bare JSON", len(calls))
        return calls

    return None
