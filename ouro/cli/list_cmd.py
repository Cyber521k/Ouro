"""
ouro/cli/list_cmd.py — List installed Ouro models.

Usage:
    ouro list
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def list_command() -> None:
    """
    [bold]List[/bold] all installed models.

    Displays a table with model name, path, size, and last-modified time.
    """
    try:
        from ouro.registry import storage  # type: ignore[import]
    except ImportError:
        console.print("[red]Error:[/red] ouro.registry.storage module not found.")
        raise typer.Exit(code=1)

    models = storage.list_installed_models()

    if not models:
        console.print("[yellow]No models installed.[/yellow]")
        return

    table = Table(title="Installed Models", show_lines=True)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Path", style="dim")
    table.add_column("Size (MB)", justify="right", style="green")
    table.add_column("Modified", style="magenta")

    for model in models:
        name = model.get("id", "")
        path = model.get("path", "")
        size_mb = model.get("size_mb", "")
        modified = model.get("modified", "")

        table.add_row(
            str(name),
            str(path),
            f"{size_mb:.1f}" if isinstance(size_mb, float) else str(size_mb),
            str(modified),
        )

    console.print(table)
