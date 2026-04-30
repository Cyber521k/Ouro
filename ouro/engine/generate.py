"""
ouro/engine/generate.py — Text generation helpers wrapping mlx_lm.

Provides:
- generate()        — full response as a string
- generate_stream() — token-by-token iterator

mlx_lm >= 0.21 moved sampling params out of generate_step() into a
`sampler=` callable.  Use mlx_lm.sample_utils.make_sampler() to build
the sampler, then pass sampler= and logits_processors= to stream_generate /
generate_step.  Do NOT pass temp=/top_p= directly — those args were removed.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

try:
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler, make_repetition_penalty
    _MLX_AVAILABLE = True
except ImportError:
    mlx_lm = None  # type: ignore[assignment]
    _MLX_AVAILABLE = False

log = logging.getLogger("ouro.engine.generate")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_kwargs(
    temperature: float,
    top_p: float,
    min_p: float,
    repetition_penalty: float,
) -> dict[str, Any]:
    """Build the kwargs dict for stream_generate / generate_step.

    mlx_lm >=0.21 expects a ``sampler=`` callable (built via make_sampler)
    and an optional ``logits_processors=`` list instead of raw temp/top_p args.
    """
    sampler = make_sampler(
        temp=temperature,
        top_p=top_p,
        min_p=min_p,
    )
    kwargs: dict[str, Any] = {"sampler": sampler}

    if repetition_penalty != 1.0:
        kwargs["logits_processors"] = [make_repetition_penalty(repetition_penalty)]

    return kwargs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 4096,
    temperature: float = 0.6,
    top_p: float = 0.95,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
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

    sampling_kwargs = _build_kwargs(temperature, top_p, min_p, repetition_penalty)

    response: str = mlx_lm.generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
        **sampling_kwargs,
    )

    return response


def generate_stream(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 4096,
    temperature: float = 0.6,
    top_p: float = 0.95,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
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

    sampling_kwargs = _build_kwargs(temperature, top_p, min_p, repetition_penalty)
    sampling_kwargs.update(kwargs)

    stream_fn = getattr(mlx_lm, "stream_generate", None)
    if stream_fn is not None:
        for token_response in stream_fn(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            **sampling_kwargs,
        ):
            if hasattr(token_response, "text"):
                yield token_response.text
            else:
                yield str(token_response)
    else:
        # Fallback: full generate yielded as one chunk
        log.warning("mlx_lm has no streaming API; falling back to full generation.")
        yield generate(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
        )
