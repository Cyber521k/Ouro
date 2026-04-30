"""
ouro/config.py — Settings loader for Ouro MLX-native model runner.

Supports:
- Load from ~/.ouro/config.yaml
- Environment variable overrides: OURO_HOST, OURO_PORT, OURO_MODELS
- Default values: api_host=127.0.0.1, api_port=5215, hub_cache_dir=~/.ouro/models/hub/
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 5215
_DEFAULT_HUB_CACHE_DIR = str(Path("~/.ouro/models/hub/").expanduser())
_CONFIG_PATH = Path("~/.ouro/config.yaml").expanduser()


@dataclass
class OuroConfig:
    """Main configuration object for Ouro.

    Priority (highest to lowest):
        1. Environment variables
        2. ~/.ouro/config.yaml
        3. Built-in defaults
    """

    default_model: Optional[str] = None
    api_host: str = _DEFAULT_HOST
    api_port: int = _DEFAULT_PORT
    hub_cache_dir: str = _DEFAULT_HUB_CACHE_DIR
    scan_paths: List[str] = field(default_factory=list)
    # List of model IDs to load at startup — all served simultaneously
    models: List[str] = field(default_factory=list)
    # HuggingFace token — enables gated models and higher download rate limits
    hf_token: Optional[str] = None

    # -----------------------------------------------------------------------
    # Derived helpers
    # -----------------------------------------------------------------------

    @property
    def hub_cache_path(self) -> Path:
        return Path(self.hub_cache_dir).expanduser()

    @property
    def api_base_url(self) -> str:
        return f"http://{self.api_host}:{self.api_port}"


def _load_yaml_config(path: Path) -> dict:
    """Load a YAML config file and return a dict.  Returns {} on any failure."""
    if not path.exists():
        return {}
    if not _YAML_AVAILABLE:
        # Soft-fail: yaml not installed, skip file-based config
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        return {}


def _nested_get(d: dict, *keys: str):
    """Safely traverse nested dict keys; return None if any key is missing."""
    for key in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(key)  # type: ignore[assignment]
    return d


def load_config(config_path: Optional[Path] = None) -> OuroConfig:
    """
    Build and return an :class:`OuroConfig` instance, merging (in order):

    1. Built-in defaults
    2. ~/.ouro/config.yaml  (or *config_path* if provided)
    3. Environment variables (OURO_HOST, OURO_PORT, OURO_MODELS)
    """
    path = config_path if config_path is not None else _CONFIG_PATH
    file_cfg = _load_yaml_config(path)

    # Start from defaults, then overlay file values
    cfg = OuroConfig(
        default_model=file_cfg.get("default_model", None),
        api_host=file_cfg.get("api_host", _DEFAULT_HOST),
        api_port=int(file_cfg.get("api_port", _DEFAULT_PORT)),
        hub_cache_dir=file_cfg.get("hub_cache_dir", _DEFAULT_HUB_CACHE_DIR),
        scan_paths=list(file_cfg.get("scan_paths", [])),
        models=list(file_cfg.get("models", [])),
        hf_token=_nested_get(file_cfg, "huggingface", "token"),
    )

    # Environment variable overrides
    env_host = os.environ.get("OURO_HOST")
    if env_host:
        cfg.api_host = env_host

    env_port = os.environ.get("OURO_PORT")
    if env_port:
        try:
            cfg.api_port = int(env_port)
        except ValueError:
            pass  # Keep existing value on bad input

    env_models = os.environ.get("OURO_MODELS")
    if env_models:
        cfg.hub_cache_dir = env_models

    env_hf_token = os.environ.get("HF_TOKEN")
    if env_hf_token:
        cfg.hf_token = env_hf_token

    # Propagate HF token to environment so huggingface-hub picks it up
    if cfg.hf_token:
        os.environ.setdefault("HF_TOKEN", cfg.hf_token)

    return cfg


def save_config(cfg: OuroConfig, config_path: Optional[Path] = None) -> None:
    """Persist an :class:`OuroConfig` to YAML."""
    if not _YAML_AVAILABLE:
        raise RuntimeError(
            "PyYAML is required to save config files.  Install it with: pip install pyyaml"
        )
    path = config_path if config_path is not None else _CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    data: dict = {
        "api_host": cfg.api_host,
        "api_port": cfg.api_port,
        "hub_cache_dir": cfg.hub_cache_dir,
    }
    if cfg.default_model is not None:
        data["default_model"] = cfg.default_model
    if cfg.scan_paths:
        data["scan_paths"] = list(cfg.scan_paths)
    if cfg.hf_token:
        data["huggingface"] = {"token": cfg.hf_token}

    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, default_flow_style=False)


# ---------------------------------------------------------------------------
# Module-level singleton (lazy-loaded)
# ---------------------------------------------------------------------------

_config: Optional[OuroConfig] = None


def get_config() -> OuroConfig:
    """Return the global :class:`OuroConfig` singleton, loading it on first call."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    """Reset the cached singleton (useful in tests)."""
    global _config
    _config = None
