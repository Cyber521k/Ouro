"""
ouro/cli/ps.py — List running Ouro serve processes.

Usage:
    ouro ps
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

console = Console()

OURO_PIDS_DIR = Path.home() / ".ouro" / "pids"


def ps_command() -> None:
    """
    [bold]List[/bold] running Ouro serve processes.

    Reads PID files from [cyan]~/.ouro/pids/[/cyan] and checks whether each
    process is still alive.
    """
    try:
        import psutil  # type: ignore[import]
        has_psutil = True
    except ImportError:
        has_psutil = False
        console.print("[yellow]Warning:[/yellow] psutil not installed; alive-check unavailable.")

    if not OURO_PIDS_DIR.exists():
        console.print("[yellow]No running Ouro servers.[/yellow]")
        return

    pid_files = list(OURO_PIDS_DIR.glob("*.json"))

    if not pid_files:
        console.print("[yellow]No running Ouro servers.[/yellow]")
        return

    table = Table(title="Running Ouro Servers", show_lines=True)
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("PID", justify="right", style="bold")
    table.add_column("Port", justify="right", style="green")
    table.add_column("Started", style="magenta")
    table.add_column("Alive", justify="center")

    any_alive = False

    for pf in sorted(pid_files):
        try:
            data = json.loads(pf.read_text())
        except Exception:
            continue

        model = data.get("model", pf.stem)
        pid = data.get("pid", "?")
        port = data.get("port", "?")
        started = data.get("started", "?")

        if has_psutil and isinstance(pid, int):
            import psutil  # type: ignore[import]
            alive = psutil.pid_exists(pid)
        else:
            # Fallback: try sending signal 0
            alive = _check_pid_alive(pid) if isinstance(pid, int) else None

        if alive is True:
            alive_str = "[green]✓[/green]"
            any_alive = True
        elif alive is False:
            alive_str = "[red]✗[/red]"
        else:
            alive_str = "[dim]?[/dim]"

        table.add_row(str(model), str(pid), str(port), str(started), alive_str)

    if table.row_count == 0:
        console.print("[yellow]No running Ouro servers.[/yellow]")
        return

    console.print(table)

    if not any_alive:
        console.print(
            "[dim]Tip: stale PID files can be cleaned up with [bold]ouro stop <model>[/bold].[/dim]"
        )


def _check_pid_alive(pid: int) -> bool:
    """Fallback alive check using os.kill(pid, 0) when psutil is unavailable."""
    import os

    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we lack permission to signal it
        return True
    except Exception:
        return False
