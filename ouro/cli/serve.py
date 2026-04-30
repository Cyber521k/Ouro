"""
ouro/cli/serve.py — Start / stop an OpenAI-compatible HTTP server for a model.

Usage:
    ouro serve <model> [--host 127.0.0.1] [--port 8000]
    ouro stop  <model>
"""

from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

console = Console()

# Directory where PID files are stored: ~/.ouro/pids/<model_safe_name>.json
OURO_PIDS_DIR = Path.home() / ".ouro" / "pids"

# Default server settings
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def _safe_model_name(model: str) -> str:
    """Convert a model ID to a filesystem-safe filename."""
    return model.replace("/", "_").replace(":", "_")


def _pid_file(model: str) -> Path:
    return OURO_PIDS_DIR / f"{_safe_model_name(model)}.json"


def _write_pid_file(model: str, pid: int, host: str, port: int) -> None:
    import time

    OURO_PIDS_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "model": model,
        "pid": pid,
        "host": host,
        "port": port,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _pid_file(model).write_text(json.dumps(data, indent=2))


def _remove_pid_file(model: str) -> None:
    pf = _pid_file(model)
    try:
        pf.unlink(missing_ok=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# serve command
# ---------------------------------------------------------------------------

def serve_command(
    model: str = typer.Argument(..., help="Model name or path to serve"),
    host: str = typer.Option(DEFAULT_HOST, "--host", help="Bind host"),
    port: int = typer.Option(DEFAULT_PORT, "--port", help="Bind port"),
) -> None:
    """
    [bold]Serve[/bold] a model as an OpenAI-compatible HTTP API (foreground).

    The server runs in the foreground.  Press Ctrl-C to stop.
    A PID file is written to [cyan]~/.ouro/pids/[/cyan] and removed on exit.
    """
    try:
        import uvicorn  # type: ignore[import]
    except ImportError:
        console.print("[red]Error:[/red] uvicorn is not installed. Run: pip install uvicorn")
        raise typer.Exit(code=1)

    try:
        from ouro.registry import storage  # type: ignore[import]
    except ImportError:
        console.print("[red]Error:[/red] ouro.registry.storage module not found.")
        raise typer.Exit(code=1)

    try:
        from ouro.engine import loader as engine_loader  # type: ignore[import]
    except ImportError as exc:
        console.print(f"[red]Error importing engine loader:[/red] {exc}")
        raise typer.Exit(code=1)

    try:
        from ouro.api.app import create_app  # type: ignore[import]
    except ImportError as exc:
        console.print(f"[red]Error importing API app:[/red] {exc}")
        raise typer.Exit(code=1)

    # Resolve model path
    try:
        model_path = storage.resolve_model_path(model)
    except Exception as exc:
        console.print(f"[red]Could not resolve model path:[/red] {exc}")
        raise typer.Exit(code=1)

    # Load model
    console.print(f"[cyan]Loading[/cyan] [bold]{model}[/bold] …")
    try:
        loaded_model, tokenizer = engine_loader.load_model(model_path)
    except Exception as exc:
        console.print(f"[red]Failed to load model:[/red] {exc}")
        raise typer.Exit(code=1)

    # Build FastAPI app
    try:
        fastapi_app = create_app(loaded_model, tokenizer, model_name=model)
    except Exception as exc:
        console.print(f"[red]Failed to create API app:[/red] {exc}")
        raise typer.Exit(code=1)

    # Write PID file
    pid = os.getpid()
    _write_pid_file(model, pid, host, port)

    console.print(f"[bold green]Ouro serving[/bold green] [bold]{model}[/bold] on http://{host}:{port}/v1")
    console.print("Press [bold]Ctrl-C[/bold] to stop.\n")

    try:
        uvicorn.run(fastapi_app, host=host, port=port, log_level="info")
    finally:
        _remove_pid_file(model)
        console.print(f"\n[yellow]Server for '{model}' stopped.[/yellow]")


# ---------------------------------------------------------------------------
# stop command
# ---------------------------------------------------------------------------

def stop_command(
    model: str = typer.Argument(..., help="Model name of the running server to stop"),
) -> None:
    """
    [bold]Stop[/bold] a running Ouro serve process.

    Sends SIGTERM to the process recorded in the PID file and removes the file.
    """
    pf = _pid_file(model)

    if not pf.exists():
        console.print(f"[red]No PID file found for model '[bold]{model}[/bold]'. Is it running?")
        raise typer.Exit(code=1)

    try:
        data = json.loads(pf.read_text())
        pid: int = data["pid"]
    except Exception as exc:
        console.print(f"[red]Failed to read PID file:[/red] {exc}")
        raise typer.Exit(code=1)

    try:
        os.kill(pid, signal.SIGTERM)
        console.print(f"[green]✓ Sent SIGTERM to process {pid} (model: {model}).[/green]")
    except ProcessLookupError:
        console.print(f"[yellow]Process {pid} not found — it may have already exited.[/yellow]")
    except PermissionError:
        console.print(f"[red]Permission denied sending signal to process {pid}.[/red]")
        raise typer.Exit(code=1)
    finally:
        _remove_pid_file(model)
        console.print(f"[dim]PID file for '{model}' removed.[/dim]")
