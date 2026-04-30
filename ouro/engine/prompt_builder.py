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
            kwargs: Dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if tools is not None:
                kwargs["tools"] = tools

            prompt: str = apply_fn(messages, **kwargs)
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

    return _render_llama3_template(messages)
