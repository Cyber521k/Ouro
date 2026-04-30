"""
ouro/cli/rm.py — Remove an installed model.

Usage:
    ouro rm <model_id> [--force]
"""

from __future__ import annotations

import typer
from rich.console import Console

console = Console()


def rm_command(
    model_id: str = typer.Argument(..., help="Name or ID of the model to remove"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt"),
) -> None:
    """
    [bold]Remove[/bold] an installed model.

    Deletes the model files from local storage. Use [yellow]--force[/yellow] to skip
    the confirmation prompt.
    """
    try:
        from ouro.registry import storage  # type: ignore[import]
    except ImportError:
        console.print("[red]Error:[/red] ouro.registry.storage module not found.")
        raise typer.Exit(code=1)

    if not force:
        confirm = typer.confirm(f"Are you sure you want to delete model '{model_id}'?")
        if not confirm:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(code=0)

    try:
        storage.delete_model(model_id)
        console.print(f"[green]✓ Model '[bold]{model_id}[/bold]' removed successfully.[/green]")
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] Model '[bold]{model_id}[/bold]' not found.")
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"[red]Error removing model:[/red] {exc}")
        raise typer.Exit(code=1)
