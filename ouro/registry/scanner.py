"""
ouro/registry/scanner.py — Scan the local filesystem for existing MLX models.

Detects models already downloaded by HuggingFace Hub or stored in custom
directories so that Ouro can register them without re-downloading.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("ouro.registry.scanner")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HF_HUB_CACHE = Path("~/.cache/huggingface/hub/").expanduser()
_OURO_HUB_DIR = Path("~/.ouro/models/hub/").expanduser()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_valid_mlx_model(directory: Path) -> bool:
    """Return True if *directory* looks like a valid MLX model dir.

    Criteria:
    - Contains ``config.json``
    - Contains at least one ``*.safetensors`` file
    """
    if not directory.is_dir():
        return False
    if not (directory / "config.json").exists():
        return False
    safetensors = list(directory.glob("*.safetensors"))
    return len(safetensors) > 0


def _dir_size_mb(path: Path, suffix: str = ".safetensors") -> float:
    """Return total size of all *suffix* files under *path* in megabytes.

    Follows symlinks so that symlinked model directories are sized correctly.
    """
    total = 0
    try:
        for f in path.rglob(f"*{suffix}"):
            if f.is_file():
                try:
                    total += f.stat(follow_symlinks=True).st_size
                except OSError:
                    pass
    except OSError:
        pass
    return round(total / (1024 * 1024), 2)


def _path_modified(path: Path) -> str:
    """Return the ISO-8601 modification timestamp of *path* (follows symlinks)."""
    try:
        real = path.resolve()
        mtime = real.stat().st_mtime
        return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    except OSError:
        return ""


def _hf_cache_dir_to_model_id(dir_name: str) -> Optional[str]:
    """Convert a HF hub cache dir name to a ``namespace/repo`` model ID.

    HF Hub cache dirs have the form ``models--namespace--reponame``.
    Returns *None* if the name doesn't match this pattern.
    """
    if not dir_name.startswith("models--"):
        return None
    parts = dir_name[len("models--"):].split("--", 1)
    if len(parts) != 2:
        return None
    namespace, repo = parts
    return f"{namespace}/{repo}"


def _get_registered_ids(hub_dir: Path) -> set[str]:
    """Return the set of model IDs already registered under *hub_dir*.

    Walks two levels deep (namespace/repo) looking for directories that
    contain a ``config.json`` — the same criterion used by
    ``list_installed_models()``.
    """
    registered: set[str] = set()
    if not hub_dir.exists():
        return registered

    for config_json in hub_dir.rglob("config.json"):
        model_dir = config_json.parent
        try:
            model_id = model_dir.relative_to(hub_dir).as_posix()
            registered.add(model_id)
            # Also add the resolved path so we can deduplicate symlinks
            registered.add(str(model_dir.resolve()))
        except ValueError:
            pass
    return registered


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan_for_mlx_models(extra_paths: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Scan default and optional extra paths for MLX model directories.

    Scans:
    - ``~/.cache/huggingface/hub/`` — dirs matching ``models--*``
    - Any paths listed in *extra_paths*

    A directory is considered a valid MLX model if it contains both a
    ``config.json`` file **and** at least one ``*.safetensors`` weight file.

    Models already registered under ``~/.ouro/models/hub/`` are skipped so
    that the returned list contains only *new* (unregistered) models.

    Parameters
    ----------
    extra_paths:
        Optional list of additional directory paths to scan for models.

    Returns
    -------
    list[dict]
        Each entry contains:

        ``id``
            Model ID in ``namespace/repo`` format.
        ``path``
            Absolute path to the model directory.
        ``size_mb``
            Total ``*.safetensors`` size in megabytes.
        ``modified``
            ISO-8601 last-modification timestamp.
        ``source``
            One of ``"hf_cache"``, ``"ouro_hub"``, or ``"custom"``.
    """
    results: List[Dict[str, Any]] = []
    seen_paths: set[str] = set()

    # Collect already-registered IDs and resolved paths to skip duplicates
    registered = _get_registered_ids(_OURO_HUB_DIR)
    log.debug("Already registered model IDs/paths: %s", registered)

    # ------------------------------------------------------------------
    # 1. HuggingFace Hub cache
    # ------------------------------------------------------------------
    if _HF_HUB_CACHE.exists():
        log.debug("Scanning HF Hub cache: %s", _HF_HUB_CACHE)
        for cache_entry in sorted(_HF_HUB_CACHE.iterdir()):
            if not cache_entry.is_dir():
                continue
            model_id = _hf_cache_dir_to_model_id(cache_entry.name)
            if model_id is None:
                continue

            # Model files live inside snapshots/<hash>/
            snapshots_dir = cache_entry / "snapshots"
            if not snapshots_dir.exists():
                log.debug("No snapshots/ dir for %s — skipping", cache_entry.name)
                continue

            # Iterate over snapshot hashes; use the first valid one found
            for snapshot_hash_dir in sorted(snapshots_dir.iterdir()):
                if not snapshot_hash_dir.is_dir():
                    continue

                resolved = str(snapshot_hash_dir.resolve())

                if not _is_valid_mlx_model(snapshot_hash_dir):
                    log.debug(
                        "Snapshot %s does not look like an MLX model — skipping",
                        snapshot_hash_dir,
                    )
                    continue

                if model_id in registered or resolved in registered:
                    log.debug("Model %s already registered — skipping", model_id)
                    break  # No need to check other snapshots

                if resolved in seen_paths:
                    break

                seen_paths.add(resolved)
                results.append(
                    {
                        "id": model_id,
                        "path": resolved,
                        "size_mb": _dir_size_mb(snapshot_hash_dir),
                        "modified": _path_modified(snapshot_hash_dir),
                        "source": "hf_cache",
                    }
                )
                log.info("Found HF cache model: %s at %s", model_id, resolved)
                break  # Only report the first (most recent) valid snapshot

    # ------------------------------------------------------------------
    # 2. Custom / extra paths
    # ------------------------------------------------------------------
    for raw_path in extra_paths or []:
        search_root = Path(raw_path).expanduser().resolve()
        if not search_root.exists():
            log.warning("Extra scan path does not exist: %s", search_root)
            continue

        log.debug("Scanning custom path: %s", search_root)

        # If the path itself is a valid model, treat it directly
        if _is_valid_mlx_model(search_root):
            resolved = str(search_root)
            # Derive an ID from the last two path components if available
            parts = search_root.parts
            model_id = "/".join(parts[-2:]) if len(parts) >= 2 else search_root.name
            if model_id not in registered and resolved not in registered and resolved not in seen_paths:
                seen_paths.add(resolved)
                results.append(
                    {
                        "id": model_id,
                        "path": resolved,
                        "size_mb": _dir_size_mb(search_root),
                        "modified": _path_modified(search_root),
                        "source": "custom",
                    }
                )
                log.info("Found custom model: %s at %s", model_id, resolved)
            continue

        # Otherwise, walk one level of subdirectories looking for models
        for sub in sorted(search_root.iterdir()):
            if not sub.is_dir():
                continue
            if not _is_valid_mlx_model(sub):
                continue

            resolved = str(sub.resolve())
            parts = sub.parts
            model_id = "/".join(parts[-2:]) if len(parts) >= 2 else sub.name

            if model_id in registered or resolved in registered or resolved in seen_paths:
                log.debug("Custom model %s already registered — skipping", model_id)
                continue

            seen_paths.add(resolved)
            results.append(
                {
                    "id": model_id,
                    "path": resolved,
                    "size_mb": _dir_size_mb(sub),
                    "modified": _path_modified(sub),
                    "source": "custom",
                }
            )
            log.info("Found custom model: %s at %s", model_id, resolved)

    return results
