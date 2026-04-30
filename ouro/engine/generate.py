"""
ouro/engine/generate.py — Text generation helpers wrapping mlx_lm.

Provides:
- generate()        — full response as a string
- generate_stream() — token-by-token iterator
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

try:
    import mlx_lm
    _MLX_AVAILABLE = True
except ImportError:
    mlx_lm = None  # type: ignore[assignment]
    _MLX_AVAILABLE = False

log = logging.getLogger("ouro.engine.generate")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Generate a completion for *prompt* and return the full response as a string.

    Parameters
    ----------
    model:
        An MLX model object returned by :func:`~ouro.engine.loader.load_model`.
    tokenizer:
        The corresponding tokenizer.
    prompt:
        The (already-formatted) prompt string.
    max_tokens:
        Maximum number of tokens to generate.
    temperature:
        Sampling temperature.  Lower values make the output more deterministic.
    top_p:
        Nucleus-sampling cumulative probability threshold.

    Returns
    -------
    str
        Generated text (excluding the prompt).

    Raises
    ------
    RuntimeError
        If ``mlx_lm`` is not installed.
    """
    if not _MLX_AVAILABLE:
        raise RuntimeError(
            "mlx_lm is not installed.  Install it with: pip install mlx-lm"
        )

    log.debug(
        "generate(): max_tokens=%d temp=%.2f top_p=%.2f prompt_len=%d",
        max_tokens,
        temperature,
        top_p,
        len(prompt),
    )

    # mlx_lm.generate accepts keyword arguments for sampling parameters.
    # The exact kwarg names may differ between releases; we try the most
    # common interface first and fall back gracefully.
    try:
        response: str = mlx_lm.generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
            top_p=top_p,
            verbose=False,
        )
    except TypeError:
        # Older / newer API that doesn't accept verbose= or uses different names
        response = mlx_lm.generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
        )

    return response


def generate_stream(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    **kwargs: Any,
) -> Iterator[str]:
    """
    Stream generated tokens one-by-one as an iterator of strings.

    Each yielded value is a single decoded token string.  Consumers can
    concatenate them to reconstruct the full response.

    Parameters
    ----------
    model, tokenizer, prompt:
        Same as :func:`generate`.
    max_tokens, temperature, top_p:
        Same as :func:`generate`.
    **kwargs:
        Additional keyword arguments forwarded to the underlying
        ``mlx_lm`` streaming function.

    Yields
    ------
    str
        Individual decoded tokens.

    Raises
    ------
    RuntimeError
        If ``mlx_lm`` is not installed.
    """
    if not _MLX_AVAILABLE:
        raise RuntimeError(
            "mlx_lm is not installed.  Install it with: pip install mlx-lm"
        )

    log.debug(
        "generate_stream(): max_tokens=%d temp=%.2f top_p=%.2f",
        max_tokens,
        temperature,
        top_p,
    )

    # mlx_lm.stream_generate is the standard streaming API.
    # Fall back to mlx_lm.generate_step if stream_generate is unavailable.
    stream_fn = getattr(mlx_lm, "stream_generate", None)
    if stream_fn is not None:
        for token_response in stream_fn(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
            top_p=top_p,
            **kwargs,
        ):
            # stream_generate yields objects with a .text attribute
            if hasattr(token_response, "text"):
                yield token_response.text
            else:
                yield str(token_response)
    else:
        # Fallback: use generate_step (lower-level token-by-token iterator)
        generate_step_fn = getattr(mlx_lm, "generate_step", None)
        if generate_step_fn is None:
            # Last resort: run full generate and yield in one chunk
            log.warning(
                "mlx_lm has no streaming API; falling back to full generation."
            )
            yield generate(
                model,
                tokenizer,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return

        # generate_step yields (token_id, logprobs) tuples
        for token_id, _logprobs in generate_step_fn(
            prompt,
            model,
            temp=temperature,
            top_p=top_p,
        ):
            token_str = tokenizer.decode([token_id])
            yield token_str
            max_tokens -= 1
            if max_tokens <= 0:
                break
