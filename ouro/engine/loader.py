"""
ouro/engine/loader.py — MLX model loader with caching and rich spinner.

Wraps mlx_lm.load() with:
- Module-level cache keyed by model_path
- Load-time logging via rich spinner
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Tuple

try:
    import mlx_lm
    _MLX_AVAILABLE = True
except ImportError:
    mlx_lm = None  # type: ignore[assignment]
    _MLX_AVAILABLE = False

try:
    from rich.console import Console
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

log = logging.getLogger("ouro.engine.loader")

# ---------------------------------------------------------------------------
# Module-level model cache
# ---------------------------------------------------------------------------

_MODEL_CACHE: Dict[str, Tuple[Any, Any]] = {}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_model(model_path: str) -> Tuple[Any, Any]:
    """
    Load an MLX model and its tokenizer, returning ``(model, tokenizer)``.

    The result is cached in-process so that subsequent calls with the same
    *model_path* return instantly without re-loading weights from disk.

    Parameters
    ----------
    model_path:
        A local directory path or Hugging Face Hub model ID understood by
        ``mlx_lm.load()``.

    Returns
    -------
    tuple[model, tokenizer]

    Raises
    ------
    RuntimeError
        If ``mlx_lm`` is not installed.
    """
    if not _MLX_AVAILABLE:
        raise RuntimeError(
            "mlx_lm is not installed.  Install it with: pip install mlx-lm"
        )

    # Cache hit — return immediately
    if model_path in _MODEL_CACHE:
        log.debug("Cache hit for model '%s'", model_path)
        return _MODEL_CACHE[model_path]

    # Cache miss — load from disk/hub
    log.info("Loading model '%s' …", model_path)
    start = time.monotonic()

    if _RICH_AVAILABLE:
        from rich.console import Console as _Console
        _console = _Console()
        with _console.status(
            f"[bold cyan]Loading model [bold white]{model_path}[/bold white] …",
            spinner="dots",
        ):
            model, tokenizer = mlx_lm.load(model_path)
    else:
        print(f"Loading model '{model_path}' …")
        model, tokenizer = mlx_lm.load(model_path)

    elapsed = time.monotonic() - start
    log.info("Model '%s' loaded in %.2f s", model_path, elapsed)

    _MODEL_CACHE[model_path] = (model, tokenizer)
    return model, tokenizer


def clear_cache(model_path: str | None = None) -> None:
    """
    Remove one or all entries from the in-process model cache.

    Parameters
    ----------
    model_path:
        If provided, only that entry is removed.  If ``None``, the entire
        cache is cleared (useful to free GPU/Metal memory).
    """
    if model_path is None:
        _MODEL_CACHE.clear()
        log.debug("Model cache cleared (all entries)")
    elif model_path in _MODEL_CACHE:
        del _MODEL_CACHE[model_path]
        log.debug("Removed '%s' from model cache", model_path)


def is_cached(model_path: str) -> bool:
    """Return ``True`` if *model_path* is already in the module-level cache."""
    return model_path in _MODEL_CACHE
