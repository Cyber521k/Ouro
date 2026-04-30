"""
ouro/utils.py — Console helpers and PID file management for Ouro.

Provides:
- Console helpers using rich (console, print_error, print_success, print_info)
- PID file management: write_pid_file, read_pid_file, delete_pid_file, list_pid_files
- PID files stored in ~/.ouro/pids/<name>.json
- Structured logging setup
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.theme import Theme
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Console setup
# ---------------------------------------------------------------------------

_OURO_THEME = None
if _RICH_AVAILABLE:
    _OURO_THEME = Theme(
        {
            "success": "bold green",
            "error": "bold red",
            "info": "bold cyan",
            "warning": "bold yellow",
        }
    )

if _RICH_AVAILABLE:
    console = Console(theme=_OURO_THEME, highlight=True)
    err_console = Console(theme=_OURO_THEME, stderr=True, highlight=True)
else:
    # Minimal fallback when rich is not installed
    class _PlainConsole:  # type: ignore[no-redef]
        def print(self, *args: Any, **kwargs: Any) -> None:
            print(*args)

    console = _PlainConsole()  # type: ignore[assignment]
    err_console = _PlainConsole()  # type: ignore[assignment]


def print_error(message: str) -> None:
    """Print an error message to stderr in red."""
    if _RICH_AVAILABLE:
        err_console.print(f"[error]✗ {message}[/error]")
    else:
        print(f"ERROR: {message}", file=sys.stderr)


def print_success(message: str) -> None:
    """Print a success message to stdout in green."""
    if _RICH_AVAILABLE:
        console.print(f"[success]✓ {message}[/success]")
    else:
        print(f"OK: {message}")


def print_info(message: str) -> None:
    """Print an informational message to stdout in cyan."""
    if _RICH_AVAILABLE:
        console.print(f"[info]ℹ {message}[/info]")
    else:
        print(f"INFO: {message}")


def print_warning(message: str) -> None:
    """Print a warning message to stdout in yellow."""
    if _RICH_AVAILABLE:
        console.print(f"[warning]⚠ {message}[/warning]")
    else:
        print(f"WARNING: {message}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------

def setup_logging(
    level: int = logging.INFO,
    logger_name: str = "ouro",
) -> logging.Logger:
    """
    Configure and return the root Ouro logger.

    Uses :class:`rich.logging.RichHandler` when rich is available, falling back
    to a standard :class:`logging.StreamHandler` otherwise.
    """
    logger = logging.getLogger(logger_name)

    if logger.handlers:
        # Already configured — return existing logger
        return logger

    logger.setLevel(level)

    if _RICH_AVAILABLE:
        handler: logging.Handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            show_path=False,
            markup=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    logger.addHandler(handler)
    logger.propagate = False
    return logger


# Module-level logger
log = setup_logging()

# ---------------------------------------------------------------------------
# PID file management
# ---------------------------------------------------------------------------

_PIDS_DIR = Path("~/.ouro/pids").expanduser()


def _pids_dir() -> Path:
    """Return the PID files directory, creating it if necessary."""
    _PIDS_DIR.mkdir(parents=True, exist_ok=True)
    return _PIDS_DIR


def _pid_file_path(name: str) -> Path:
    return _pids_dir() / f"{name}.json"


def write_pid_file(
    name: str,
    pid: int,
    port: int,
    model: Optional[str] = None,
) -> Path:
    """
    Write a PID file for a named Ouro server process.

    Parameters
    ----------
    name:
        Logical name for the process (e.g. ``"ouro-server"``).
    pid:
        OS process ID.
    port:
        TCP port the server is listening on.
    model:
        Optional model identifier loaded by this process.

    Returns
    -------
    Path
        Path to the written PID file.
    """
    data: Dict[str, Any] = {
        "pid": pid,
        "port": port,
        "model": model,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    path = _pid_file_path(name)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    log.debug("Wrote PID file: %s", path)
    return path


def read_pid_file(name: str) -> Optional[Dict[str, Any]]:
    """
    Read a PID file by name.

    Returns
    -------
    dict or None
        Parsed PID data with keys ``pid``, ``port``, ``model``, ``started_at``,
        or ``None`` if the file does not exist or cannot be parsed.
    """
    path = _pid_file_path(name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def delete_pid_file(name: str) -> bool:
    """
    Delete the PID file for *name*.

    Returns
    -------
    bool
        ``True`` if the file was deleted, ``False`` if it did not exist.
    """
    path = _pid_file_path(name)
    if path.exists():
        try:
            path.unlink()
            log.debug("Deleted PID file: %s", path)
            return True
        except OSError:
            return False
    return False


def list_pid_files() -> List[Dict[str, Any]]:
    """
    Return all PID file records as a list of dicts.

    Each dict contains at minimum ``name`` (derived from the file stem) plus
    whatever keys were written by :func:`write_pid_file`.
    """
    results: List[Dict[str, Any]] = []
    pids_dir = _pids_dir()
    for pid_path in sorted(pids_dir.glob("*.json")):
        try:
            data = json.loads(pid_path.read_text(encoding="utf-8"))
            data["name"] = pid_path.stem
            results.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return results
