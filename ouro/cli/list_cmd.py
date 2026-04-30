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


def _auto_scan_and_register() -> int:
    """
    Silently scan for unregistered MLX models and symlink them in.

    Returns the number of newly registered models.
    """
    import logging
    import os
    from pathlib import Path

    log = logging.getLogger("ouro.cli.list_cmd")
    hub_dir = Path("~/.ouro/models/hub/").expanduser()

    try:
        from ouro.registry.scanner import scan_for_mlx_models
        from ouro.config import get_config

        cfg = get_config()
        extra = list(cfg.scan_paths)
        discovered = scan_for_mlx_models(extra_paths=extra if extra else None)
    except Exception as exc:
        log.debug("Auto-scan failed: %s", exc)
        return 0

    registered = 0
    for model in discovered:
        model_id: str = model.get("id", "")
        src_path = Path(model.get("path", ""))
        if not src_path.exists():
            continue
        dest = hub_dir / model_id
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            continue
        try:
            os.symlink(str(src_path), str(dest))
            registered += 1
            log.debug("Auto-registered model: %s → %s", model_id, src_path)
        except OSError as exc:
            log.debug("Auto-register symlink failed for %s: %s", model_id, exc)

    return registered


def list_command() -> None:
    """
    [bold]List[/bold] all installed models.

    Displays a table with model name, path, size, and last-modified time.
    Automatically detects and registers any MLX models found in the
    HuggingFace Hub cache ([cyan]~/.cache/huggingface/hub/[/cyan]) or any
    extra paths configured via [cyan]scan_paths[/cyan] in
    [cyan]~/.ouro/config.yaml[/cyan].
    """
    # Auto-scan silently so newly-downloaded HF cache models appear immediately
    new_count = _auto_scan_and_register()
    if new_count:
        console.print(f"[dim]Auto-scan: registered {new_count} new model(s) from local cache.[/dim]")

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
