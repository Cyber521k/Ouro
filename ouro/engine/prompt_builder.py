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


def _render_llama3_template(messages: List[Dict[str, str]]) -> str:
    """
    Render *messages* using a generic Llama-3 chat template.

    This is used as a fallback when the tokenizer does not ship its own
    ``chat_template``.
    """
    parts: List[str] = [_LLAMA3_BOS]

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(
            f"{_LLAMA3_START}{role}{_LLAMA3_END}\n\n{content}{_LLAMA3_EOM}"
        )

    # Add generation prompt
    parts.append(f"{_LLAMA3_START}assistant{_LLAMA3_END}\n\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_prompt(
    tokenizer: Any,
    messages: List[Dict[str, str]],
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
        Ordered list of role/content dicts, e.g.::

            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user",   "content": "Hello!"},
            ]
    tools:
        Optional list of OpenAI-format tool definitions to pass to the
        tokenizer template.

    Returns
    -------
    str
        A fully formatted prompt string ready to be passed to the model.
    """
    # ------------------------------------------------------------------
    # Primary path: tokenizer.apply_chat_template
    # ------------------------------------------------------------------
    apply_fn = getattr(tokenizer, "apply_chat_template", None)
    chat_template_attr = getattr(tokenizer, "chat_template", None)

    if apply_fn is not None and chat_template_attr is not None:
        try:
            # Convert Pydantic objects → clean dicts, dropping None fields so
            # the Jinja template doesn't choke on unexpected null keys.
            def _to_dict(msg: Any) -> Dict[str, Any]:
                if isinstance(msg, dict):
                    return {k: v for k, v in msg.items() if v is not None}
                # Pydantic v2
                if hasattr(msg, "model_dump"):
                    return {k: v for k, v in msg.model_dump().items() if v is not None}
                # Pydantic v1
                if hasattr(msg, "dict"):
                    return {k: v for k, v in msg.dict().items() if v is not None}
                return dict(msg)

            clean_messages = [_to_dict(m) for m in messages]

            kwargs: Dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if tools is not None:
                kwargs["tools"] = tools
                # Disable thinking when tools are present — Qwen3 thinking +
                # tool-calling conflict; the model stops mid-generation.
                kwargs["enable_thinking"] = False
            else:
                # Qwen3 and similar thinking models: enable_thinking so the
                # template wraps reasoning in <think>…</think> (stripped in
                # the API layer).  Silently ignored by non-thinking models.
                try:
                    apply_fn(clean_messages, **{**kwargs, "enable_thinking": True})
                    kwargs["enable_thinking"] = True
                except Exception:
                    pass  # model doesn't support enable_thinking — skip it

            prompt: str = apply_fn(clean_messages, **kwargs)
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

    # Ensure messages are plain dicts for the fallback template too
    if not all(isinstance(m, dict) for m in messages):
        def _to_plain(msg: Any) -> Dict[str, Any]:
            if isinstance(msg, dict):
                return msg
            if hasattr(msg, "model_dump"):
                return {k: v for k, v in msg.model_dump().items() if v is not None}
            if hasattr(msg, "dict"):
                return {k: v for k, v in msg.dict().items() if v is not None}
            return dict(msg)
        messages = [_to_plain(m) for m in messages]

    return _render_llama3_template(messages)
