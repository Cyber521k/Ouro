"""
ouro/registry/storage.py — Local model storage helpers.

Provides:
- get_models_dir()        → ~/.ouro/models/hub/
- get_manifests_dir()     → ~/.ouro/models/manifests/
- resolve_model_path()    → locate a model on disk
- list_installed_models() → enumerate locally available models
- delete_model()          → remove a model from disk
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("ouro.registry.storage")

# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------


def get_models_dir() -> Path:
    """Return ``~/.ouro/models/hub/``, creating it if necessary."""
    path = Path("~/.ouro/models/hub/").expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_manifests_dir() -> Path:
    """Return ``~/.ouro/models/manifests/``, creating it if necessary."""
    path = Path("~/.ouro/models/manifests/").expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def resolve_model_path(model_id: str) -> Optional[Path]:
    """
    Resolve *model_id* to an absolute local filesystem path.

    Resolution order
    ----------------
    1. If *model_id* is a path that already exists on disk, return it.
    2. If *model_id* looks like ``namespace/repo``, check
       ``~/.ouro/models/hub/namespace/repo/``.
    3. Check ``~/.ouro/models/manifests/<model_id>.yaml`` for a local alias
       that points to a ``from_model`` path.
    4. Return ``None`` if nothing is found.

    Parameters
    ----------
    model_id:
        A local directory path, a Hub-style ``namespace/repo`` identifier,
        or a named manifest alias.

    Returns
    -------
    Path or None
    """
    # 1. Literal local path
    candidate = Path(model_id).expanduser()
    if candidate.exists():
        return candidate.resolve()

    # 2. Hub-style namespace/repo  (exactly one '/')
    if "/" in model_id and not model_id.startswith("/"):
        hub_path = get_models_dir() / model_id
        if hub_path.exists():
            return hub_path.resolve()

    # 3. Manifest-based alias
    manifest_path = get_manifests_dir() / f"{model_id}.yaml"
    if manifest_path.exists():
        try:
            from ouro.registry.manifest import load_manifest  # local import avoids circular
            manifest = load_manifest(model_id)
            if manifest:
                from_model = manifest.get("from_model")
                if from_model:
                    return resolve_model_path(from_model)
        except Exception as exc:
            log.debug("Manifest resolution failed for '%s': %s", model_id, exc)

    return None


# ---------------------------------------------------------------------------
# Model listing
# ---------------------------------------------------------------------------


def _dir_size_mb(path: Path, suffix: str = ".safetensors") -> float:
    """Return the total size of all *suffix* files under *path* in megabytes."""
    total = sum(f.stat().st_size for f in path.rglob(f"*{suffix}") if f.is_file())
    return round(total / (1024 * 1024), 2)


def _path_modified(path: Path) -> str:
    """Return the ISO-8601 modification timestamp of *path*."""
    try:
        mtime = path.stat().st_mtime
        return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    except OSError:
        return ""


def list_installed_models() -> List[Dict[str, Any]]:
    """
    Enumerate all locally installed models.

    Scans two locations:

    1. ``~/.ouro/models/hub/`` — any directory containing a ``config.json``
       is treated as an installed model.  The model ID is derived from the
       path relative to the hub root (e.g. ``mlx-community/Llama-3-8B``).

    2. ``~/.ouro/models/manifests/`` — each ``.yaml`` file represents a
       named local alias / custom model.

    Returns
    -------
    list[dict]
        Each entry has the following keys:

        ``id``
            String identifier (hub path or manifest name).
        ``path``
            Absolute path string.
        ``size_mb``
            Total size of ``*.safetensors`` files in megabytes.
        ``modified``
            ISO-8601 last-modification timestamp of the directory.
    """
    results: List[Dict[str, Any]] = []
    seen_paths: set = set()

    # --- Hub models ---
    hub_dir = get_models_dir()
    for config_json in sorted(hub_dir.rglob("config.json")):
        model_dir = config_json.parent
        resolved = model_dir.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)

        try:
            model_id = model_dir.relative_to(hub_dir).as_posix()
        except ValueError:
            model_id = str(model_dir)

        results.append(
            {
                "id": model_id,
                "path": str(resolved),
                "size_mb": _dir_size_mb(model_dir),
                "modified": _path_modified(model_dir),
            }
        )

    # --- Manifest-based models ---
    manifests_dir = get_manifests_dir()
    for manifest_file in sorted(manifests_dir.glob("*.yaml")):
        name = manifest_file.stem
        try:
            from ouro.registry.manifest import load_manifest
            manifest = load_manifest(name)
        except Exception:
            manifest = None

        from_model: Optional[str] = manifest.get("from_model") if manifest else None
        resolved_path = resolve_model_path(from_model) if from_model else None

        results.append(
            {
                "id": name,
                "path": str(resolved_path) if resolved_path else str(manifest_file),
                "size_mb": _dir_size_mb(resolved_path) if resolved_path else 0.0,
                "modified": _path_modified(manifest_file),
            }
        )

    return results


# ---------------------------------------------------------------------------
# Model deletion
# ---------------------------------------------------------------------------


def delete_model(model_id: str) -> bool:
    """
    Delete a locally installed model.

    Attempts to remove:
    - The hub directory (``~/.ouro/models/hub/<model_id>``)
    - The manifest file (``~/.ouro/models/manifests/<model_id>.yaml``)

    Parameters
    ----------
    model_id:
        Hub-style ID or manifest name.

    Returns
    -------
    bool
        ``True`` if at least one artefact was removed, ``False`` otherwise.
    """
    deleted = False

    # Hub directory
    hub_path = get_models_dir() / model_id
    if hub_path.exists():
        try:
            shutil.rmtree(hub_path)
            log.info("Deleted hub model directory: %s", hub_path)
            deleted = True
        except OSError as exc:
            log.error("Failed to delete '%s': %s", hub_path, exc)

    # Manifest file
    manifest_path = get_manifests_dir() / f"{model_id}.yaml"
    if manifest_path.exists():
        try:
            manifest_path.unlink()
            log.info("Deleted manifest: %s", manifest_path)
            deleted = True
        except OSError as exc:
            log.error("Failed to delete manifest '%s': %s", manifest_path, exc)

    return deleted
