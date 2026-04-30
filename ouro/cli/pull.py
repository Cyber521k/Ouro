"""
ouro/cli/pull.py — Download a model from HuggingFace Hub.

Usage:
    ouro pull <repo_id> [--revision main]
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.spinner import Spinner
from rich.live import Live
from rich.text import Text

console = Console()

# Default base directory for Ouro models
OURO_MODELS_DIR = Path.home() / ".ouro" / "models" / "hub"


def _model_cache_path(repo_id: str) -> Path:
    """Return the local cache path for a given HuggingFace repo_id."""
    parts = repo_id.split("/", 1)
    if len(parts) == 2:
        namespace, repo = parts
    else:
        namespace, repo = "local", parts[0]
    return OURO_MODELS_DIR / namespace / repo


def pull_command(
    repo_id: str = typer.Argument(..., help="HuggingFace repo ID, e.g. mlx-community/Qwen2.5-7B-Instruct"),
    revision: str = typer.Option("main", "--revision", help="Git revision / branch / tag to download"),
) -> None:
    """
    [bold]Pull[/bold] a model from HuggingFace Hub.

    Downloads the specified model and saves it to [cyan]~/.ouro/models/hub/<namespace>/<repo>/[/cyan].
    If the model is already cached locally the download is skipped.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        console.print("[red]Error:[/red] huggingface_hub is not installed. Run: pip install huggingface-hub")
        raise typer.Exit(code=1)

    cache_path = _model_cache_path(repo_id)

    # Check if already cached (directory exists and is non-empty)
    if cache_path.exists() and any(cache_path.iterdir()):
        console.print(f"[green]Already cached[/green] → {cache_path}")
        return

    console.print(f"[cyan]Pulling[/cyan] [bold]{repo_id}[/bold] (revision: {revision}) …")
    OURO_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    spinner_text = Text(f"Downloading {repo_id}…")

    with Live(Spinner("dots", text=spinner_text), console=console, refresh_per_second=10):
        try:
            local_dir = snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=str(cache_path),
            )
        except Exception as exc:  # pragma: no cover
            console.print(f"[red]Download failed:[/red] {exc}")
            raise typer.Exit(code=1)

    console.print(f"[green]✓ Saved to[/green] {local_dir}")
