"""
ouro/registry/manifest.py — Modelfile parsing and manifest persistence.

Provides:
- parse_modelfile()  — parse an Ollama-style Modelfile into a dict
- save_manifest()    — save a manifest to ~/.ouro/models/manifests/<name>.yaml
- load_manifest()    — load a manifest by name
- list_manifests()   — list all saved manifest names
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

log = logging.getLogger("ouro.registry.manifest")

# ---------------------------------------------------------------------------
# Modelfile parsing
# ---------------------------------------------------------------------------

# Matches lines of the form:
#   DIRECTIVE value
# where value may be quoted or unquoted.
_RE_DIRECTIVE = re.compile(
    r"^(?P<directive>[A-Z]+)\s+(?P<value>.+)$",
    re.MULTILINE,
)

# Matches a quoted string: "..." or '...'
_RE_QUOTED = re.compile(r'^["\'](.+)["\']$', re.DOTALL)


def _unquote(value: str) -> str:
    """Strip surrounding quotes from *value* if present."""
    value = value.strip()
    match = _RE_QUOTED.match(value)
    if match:
        return match.group(1)
    return value


def parse_modelfile(path: str) -> Dict[str, Any]:
    """
    Parse an Ollama-style Modelfile and return a structured dict.

    Supported directives
    --------------------
    FROM
        Base model identifier (required).
    SYSTEM
        System prompt string.
    PARAMETER
        A key/value pair added under the ``parameters`` sub-dict.
        Values that look like integers or floats are coerced accordingly.
    TEMPLATE
        A Jinja2 / custom template string.

    Parameters
    ----------
    path:
        Filesystem path to the Modelfile.

    Returns
    -------
    dict
        Keys: ``from_model``, ``system``, ``parameters``, ``template``.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Modelfile not found: {path}")

    text = file_path.read_text(encoding="utf-8")

    result: Dict[str, Any] = {
        "from_model": None,
        "system": None,
        "parameters": {},
        "template": None,
    }

    for match in _RE_DIRECTIVE.finditer(text):
        directive = match.group("directive").upper()
        raw_value = match.group("value").strip()
        value = _unquote(raw_value)

        if directive == "FROM":
            result["from_model"] = value

        elif directive == "SYSTEM":
            result["system"] = value

        elif directive == "PARAMETER":
            # PARAMETER <key> <value>
            parts = raw_value.split(None, 1)
            if len(parts) == 2:
                param_key, param_val_raw = parts
                param_val: Any = _unquote(param_val_raw)
                # Coerce numeric values
                try:
                    param_val = int(param_val)
                except (ValueError, TypeError):
                    try:
                        param_val = float(param_val)
                    except (ValueError, TypeError):
                        pass
                result["parameters"][param_key] = param_val
            else:
                log.warning("Malformed PARAMETER directive: '%s'", raw_value)

        elif directive == "TEMPLATE":
            result["template"] = value

        else:
            log.debug("Unknown Modelfile directive '%s' — ignoring.", directive)

    if result["from_model"] is None:
        log.warning("Modelfile '%s' has no FROM directive.", path)

    return result


# ---------------------------------------------------------------------------
# Manifest persistence
# ---------------------------------------------------------------------------


def _manifests_dir() -> Path:
    """Return the manifests directory, creating it if needed."""
    path = Path("~/.ouro/models/manifests/").expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_manifest(name: str, data: Dict[str, Any]) -> None:
    """
    Persist a manifest dict to ``~/.ouro/models/manifests/<name>.yaml``.

    Parameters
    ----------
    name:
        Logical name for the manifest (used as the file stem).
    data:
        Dict to serialise.  Typically the output of :func:`parse_modelfile`
        or a hand-crafted dict with the same schema.

    Raises
    ------
    RuntimeError
        If PyYAML is not installed.
    """
    if not _YAML_AVAILABLE:
        raise RuntimeError(
            "PyYAML is required to save manifests.  Install it with: pip install pyyaml"
        )

    manifest_path = _manifests_dir() / f"{name}.yaml"
    with manifest_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, default_flow_style=False, allow_unicode=True)
    log.debug("Saved manifest '%s' → %s", name, manifest_path)


def load_manifest(name: str) -> Optional[Dict[str, Any]]:
    """
    Load a manifest by *name* from ``~/.ouro/models/manifests/``.

    Parameters
    ----------
    name:
        Manifest name (without ``.yaml`` extension).

    Returns
    -------
    dict or None
        Parsed manifest data, or ``None`` if the file does not exist or
        cannot be parsed.
    """
    if not _YAML_AVAILABLE:
        log.warning("PyYAML not installed; cannot load manifest '%s'.", name)
        return None

    manifest_path = _manifests_dir() / f"{name}.yaml"
    if not manifest_path.exists():
        return None

    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            return None
        return data
    except Exception as exc:
        log.error("Failed to load manifest '%s': %s", name, exc)
        return None


def list_manifests() -> List[str]:
    """
    Return the names of all saved manifests.

    Returns
    -------
    list[str]
        Sorted list of manifest names (without ``.yaml`` extension).
    """
    manifests_dir = _manifests_dir()
    return sorted(p.stem for p in manifests_dir.glob("*.yaml"))
