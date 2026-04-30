"""
ouro/engine/prompt_builder.py — Chat prompt construction for MLX models.

Provides:
- build_prompt() — build a formatted prompt string from a messages list
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger("ouro.engine.prompt_builder")

# ---------------------------------------------------------------------------
# Fallback template (Llama-3 style)
# ---------------------------------------------------------------------------

_LLAMA3_BOS = "<|begin_of_text|>"
_LLAMA3_EOM = "<|eot_id|>"
_LLAMA3_START = "<|start_header_id|>"
_LLAMA3_END = "<|end_header_id|>"


def _render_llama3_template(messages: List[Dict[str, Any]]) -> str:
    """
    Render *messages* using a generic Llama-3 chat template.

    This is used as a fallback when the tokenizer does not ship its own
    ``chat_template``.
    """
    parts: List[str] = [_LLAMA3_BOS]

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content") or ""
        parts.append(
            f"{_LLAMA3_START}{role}{_LLAMA3_END}\n\n{content}{_LLAMA3_EOM}"
        )

    # Add generation prompt
    parts.append(f"{_LLAMA3_START}assistant{_LLAMA3_END}\n\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Message normalisation
# ---------------------------------------------------------------------------

def _to_dict(msg: Any) -> Dict[str, Any]:
    """
    Convert a message (Pydantic model or dict) to a clean plain dict.

    Rules:
    - Drop None-valued keys so Jinja2 templates don't choke on null fields.
    - Keep tool_calls as-is if already in list-of-dict form.
    - Serialise ToolCall/FunctionCall Pydantic objects to dicts recursively.
    """
    if isinstance(msg, dict):
        raw = msg
    elif hasattr(msg, "model_dump"):
        # Pydantic v2
        raw = msg.model_dump()
    elif hasattr(msg, "dict"):
        # Pydantic v1
        raw = msg.dict()
    else:
        raw = dict(msg)

    # Drop None values — Jinja templates will error on unexpected nulls
    clean: Dict[str, Any] = {}
    for k, v in raw.items():
        if v is None:
            continue
        # Recursively serialise nested Pydantic objects (e.g. tool_calls list)
        if isinstance(v, list):
            serialised = []
            for item in v:
                if hasattr(item, "model_dump"):
                    serialised.append({ik: iv for ik, iv in item.model_dump().items() if iv is not None})
                elif hasattr(item, "dict"):
                    serialised.append({ik: iv for ik, iv in item.dict().items() if iv is not None})
                else:
                    serialised.append(item)
            clean[k] = serialised
        else:
            clean[k] = v

    return clean


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_prompt(
    tokenizer: Any,
    messages: List[Any],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Build a fully formatted prompt string from a list of chat *messages*.

    Attempts to use the tokenizer's own ``apply_chat_template`` method (which
    honours the model's built-in Jinja2 template).  If the tokenizer does not
    expose a chat template — or if it raises an exception — the function falls
    back to a generic Llama-3 style template and logs a warning.

    Parameters
    ----------
    tokenizer:
        A Hugging Face tokenizer (or any object implementing
        ``apply_chat_template``).
    messages:
        Ordered list of role/content items (dicts or Pydantic ChatMessage).
    tools:
        Optional list of OpenAI-format tool definitions to pass to the
        tokenizer template.

    Returns
    -------
    str
        A fully formatted prompt string ready to be passed to the model.
    """
    # Convert all messages to clean plain dicts — drops None fields so
    # the Jinja template doesn't choke on unexpected null keys.
    clean_messages = [_to_dict(m) for m in messages]

    # ------------------------------------------------------------------
    # Primary path: tokenizer.apply_chat_template
    # ------------------------------------------------------------------
    apply_fn = getattr(tokenizer, "apply_chat_template", None)
    chat_template_attr = getattr(tokenizer, "chat_template", None)

    if apply_fn is not None and chat_template_attr is not None:
        try:
            kwargs: Dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }

            if tools is not None:
                # Pass tools to the template.
                # Disable thinking when tools are present — Qwen3 thinking +
                # tool-calling conflict; the model stops mid-generation.
                kwargs["tools"] = tools
                kwargs["enable_thinking"] = False
                prompt: str = apply_fn(clean_messages, **kwargs)
                return prompt
            else:
                # No tools — disable thinking mode (enable_thinking=False).
                # Thinking mode causes the model to emit a long <think>…</think>
                # block before answering; if max_tokens is hit mid-think the
                # response is empty after stripping.  Non-thinking mode is faster,
                # more predictable, and what most API consumers expect.
                try:
                    prompt = apply_fn(clean_messages, **{**kwargs, "enable_thinking": False})
                    return prompt
                except TypeError:
                    pass  # model doesn't support enable_thinking kwarg

                prompt = apply_fn(clean_messages, **kwargs)
                return prompt

        except Exception as exc:
            log.warning(
                "tokenizer.apply_chat_template raised %s: %s — "
                "falling back to Llama-3 template.",
                type(exc).__name__,
                exc,
            )
    else:
        log.warning(
            "Tokenizer has no chat_template; falling back to Llama-3 style template."
        )

    # ------------------------------------------------------------------
    # Fallback: generic Llama-3 style template
    # ------------------------------------------------------------------
    if tools is not None:
        log.warning(
            "Tool definitions are not supported by the fallback Llama-3 template "
            "and will be ignored."
        )

    return _render_llama3_template(clean_messages)
