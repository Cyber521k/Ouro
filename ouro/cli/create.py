"""
ouro/cli/create.py — Create a custom model from a Modelfile.

Usage:
    ouro create <name> -f <modelfile_path>
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

console = Console()


def create_command(
    name: str = typer.Argument(..., help="Name to assign to the new model"),
    modelfile: Path = typer.Option(
        ...,
        "--file",
        "-f",
        help="Path to the Modelfile",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """
    [bold]Create[/bold] a custom model from a Modelfile.

    Parses the Modelfile and saves the resulting manifest so the model can be
    referenced by [cyan]<name>[/cyan] in other Ouro commands.
    """
    try:
        from ouro.registry import manifest  # type: ignore[import]
    except ImportError:
        console.print("[red]Error:[/red] ouro.registry.manifest module not found.")
        raise typer.Exit(code=1)

    console.print(f"[cyan]Parsing[/cyan] Modelfile: {modelfile}")

    try:
        parsed = manifest.parse_modelfile(modelfile)
    except Exception as exc:
        console.print(f"[red]Failed to parse Modelfile:[/red] {exc}")
        raise typer.Exit(code=1)

    try:
        manifest.save_manifest(name, parsed)
    except Exception as exc:
        console.print(f"[red]Failed to save manifest:[/red] {exc}")
        raise typer.Exit(code=1)

    console.print(f"[green]✓ Created model[/green] [bold]{name}[/bold]")
