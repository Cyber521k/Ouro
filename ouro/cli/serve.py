"""
ouro/cli/serve.py — Start the Ouro multi-model server.

All models listed in ~/.ouro/config.yaml are loaded at startup and served
simultaneously on a single port — no model argument needed, just like Ollama.

Usage:
    ouro serve [--host 127.0.0.1] [--port 5215]
    ouro stop
"""
from __future__ import annotations

import json
import os
import signal
from pathlib import Path

import typer
from rich.console import Console

console = Console()

OURO_PID_FILE = Path.home() / ".ouro" / "ouro.pid"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5215


# ---------------------------------------------------------------------------
# PID helpers
# ---------------------------------------------------------------------------

def _write_pid(pid: int, host: str, port: int) -> None:
    import time
    OURO_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    OURO_PID_FILE.write_text(
        json.dumps({"pid": pid, "host": host, "port": port,
                    "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    )


def _read_pid() -> dict | None:
    if not OURO_PID_FILE.exists():
        return None
    try:
        return json.loads(OURO_PID_FILE.read_text())
    except Exception:
        return None


def _remove_pid() -> None:
    OURO_PID_FILE.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# serve command
# ---------------------------------------------------------------------------

def serve_command(
    host: str = typer.Option(DEFAULT_HOST, "--host", help="Bind host"),
    port: int = typer.Option(DEFAULT_PORT, "--port", help="Bind port"),
    background: bool = typer.Option(False, "--background", "-b",
                                     help="Fork to background (daemonize)"),
) -> None:
    """
    [bold]Start[/bold] the Ouro server — all configured models loaded at once.

    Add models to serve in [cyan]~/.ouro/config.yaml[/cyan]:

        [dim]models:[/dim]
          [dim]- mlx-community/Qwen3-8B-4bit[/dim]
          [dim]- mlx-community/Llama-3.2-3B-Instruct-4bit[/dim]

    The server runs on [cyan]http://{host}:{port}/v1[/cyan] and exposes an
    OpenAI-compatible REST API.
    """
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        console.print("[red]Error:[/red] uvicorn is not installed.  Run: pip install uvicorn")
        raise typer.Exit(code=1)

    from ouro.config import get_config
    cfg = get_config()

    if not cfg.models:
        console.print(
            "[yellow]Warning:[/yellow] No models in [cyan]~/.ouro/config.yaml[/cyan].\n"
            "Add a [bold]models[/bold] list, e.g.:\n\n"
            "  [dim]models:[/dim]\n"
            "  [dim]  - mlx-community/Qwen3-8B-4bit[/dim]\n"
        )
        raise typer.Exit(code=1)

    console.print(f"[bold green]Ouro[/bold green] starting on [cyan]http://{host}:{port}/v1[/cyan]")
    console.print(f"Loading [bold]{len(cfg.models)}[/bold] model(s):")
    for m in cfg.models:
        console.print(f"  [dim]•[/dim] {m}")
    console.print("Press [bold]Ctrl-C[/bold] to stop.\n")

    if background:
        _daemonize(host, port)
        return

    # Foreground — write PID then run
    _write_pid(os.getpid(), host, port)
    try:
        from ouro.api.server import run_server
        run_server(host=host, port=port)
    finally:
        _remove_pid()
        console.print("\n[yellow]Ouro server stopped.[/yellow]")


def _daemonize(host: str, port: int) -> None:
    """Fork to background, redirect stdio, write PID file."""
    pid = os.fork()
    if pid > 0:
        # Parent — print info and exit
        console.print(f"[green]Ouro daemon started (PID {pid})[/green]")
        console.print(f"  Logs: [cyan]~/.ouro/ouro.log[/cyan]")
        console.print(f"  Stop: [bold]ouro stop[/bold]")
        _write_pid(pid, host, port)
        raise typer.Exit(code=0)

    # Child — become session leader
    os.setsid()

    # Redirect stdio to log file
    log_path = Path.home() / ".ouro" / "ouro.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as logf:
        os.dup2(logf.fileno(), 1)
        os.dup2(logf.fileno(), 2)

    _write_pid(os.getpid(), host, port)

    from ouro.api.server import run_server
    try:
        run_server(host=host, port=port)
    finally:
        _remove_pid()


# ---------------------------------------------------------------------------
# stop command
# ---------------------------------------------------------------------------

def stop_command() -> None:
    """
    [bold]Stop[/bold] a running Ouro server.
    """
    info = _read_pid()
    if not info:
        console.print("[red]No running Ouro server found.[/red]  (PID file missing)")
        raise typer.Exit(code=1)

    pid: int = info["pid"]
    try:
        os.kill(pid, signal.SIGTERM)
        console.print(f"[green]✓ Sent SIGTERM to Ouro server (PID {pid})[/green]")
    except ProcessLookupError:
        console.print(f"[yellow]Process {pid} not found — it may have already exited.[/yellow]")
    except PermissionError:
        console.print(f"[red]Permission denied sending signal to PID {pid}.[/red]")
        raise typer.Exit(code=1)
    finally:
        _remove_pid()
