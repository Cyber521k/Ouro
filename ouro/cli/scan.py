"""
ouro/cli/scan.py — Auto-detect and optionally register MLX models.

Usage:
    ouro scan [--no-register] [--paths /some/path]
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

log = logging.getLogger("ouro.cli.scan")
console = Console()

# ---------------------------------------------------------------------------
# Hub directory (same default as storage.py)
# ---------------------------------------------------------------------------

_OURO_HUB_DIR = Path("~/.ouro/models/hub/").expanduser()


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


def scan_command(
    register: bool = typer.Option(
        True,
        "--register/--no-register",
        help="Symlink discovered models into ~/.ouro/models/hub/ so they appear in `ouro list`.",
    ),
    paths: Optional[List[str]] = typer.Option(
        None,
        "--paths",
        help="Additional directories to scan for MLX models (can be repeated).",
    ),
) -> None:
    """
    [bold]Scan[/bold] for locally installed MLX models and optionally register them.

    Scans [cyan]~/.cache/huggingface/hub/[/cyan] (and any extra [cyan]--paths[/cyan]) for
    model directories that contain a ``config.json`` and at least one
    ``*.safetensors`` weight file.  Models already present in
    [cyan]~/.ouro/models/hub/[/cyan] are skipped.

    With [bold]--register[/bold] (the default), newly discovered models are symlinked
    into [cyan]~/.ouro/models/hub/<namespace>/<repo>/[/cyan] so they are immediately
    available to [bold]ouro list[/bold], [bold]ouro run[/bold], etc. without re-downloading.
    """
    from ouro.registry.scanner import scan_for_mlx_models  # local import

    extra: List[str] = list(paths) if paths else []

    # Also pull in any paths configured in ouro config
    try:
        from ouro.config import get_config
        cfg = get_config()
        extra = list(extra) + list(cfg.scan_paths)
    except Exception as exc:
        log.debug("Could not load ouro config scan_paths: %s", exc)

    console.print("[cyan]Scanning for MLX models…[/cyan]")
    discovered = scan_for_mlx_models(extra_paths=extra if extra else None)

    if not discovered:
        console.print("[yellow]No new MLX models found.[/yellow]")
        console.print("\n[dim]Summary:[/dim] Found [bold]0[/bold] new models, registered [bold]0[/bold]")
        return

    # ------------------------------------------------------------------
    # Display results table
    # ------------------------------------------------------------------
    table = Table(title="Discovered MLX Models (not yet in ouro hub)", show_lines=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Source", style="magenta")
    table.add_column("Size (MB)", justify="right", style="green")
    table.add_column("Modified", style="dim")
    table.add_column("Path", style="dim")

    for model in discovered:
        table.add_row(
            str(model.get("id", "")),
            str(model.get("source", "")),
            f"{model.get('size_mb', 0.0):.1f}",
            str(model.get("modified", "")),
            str(model.get("path", "")),
        )

    console.print(table)

    # ------------------------------------------------------------------
    # Register (symlink) discovered models
    # ------------------------------------------------------------------
    registered_count = 0

    if register:
        console.print("\n[cyan]Registering models via symlinks…[/cyan]")
        for model in discovered:
            model_id: str = model.get("id", "")
            src_path = Path(model.get("path", ""))

            if not src_path.exists():
                console.print(f"[yellow]  ⚠ Skipping {model_id}: source path no longer exists.[/yellow]")
                continue

            # Build symlink destination: ~/.ouro/models/hub/<namespace>/<repo>
            dest = _OURO_HUB_DIR / model_id
            dest.parent.mkdir(parents=True, exist_ok=True)

            if dest.exists() or dest.is_symlink():
                console.print(f"[dim]  Already exists: {model_id}[/dim]")
                continue

            try:
                os.symlink(str(src_path), str(dest))
                console.print(f"[green]  ✓ Registered:[/green] {model_id} → {src_path}")
                registered_count += 1
            except OSError as exc:
                console.print(f"[red]  ✗ Failed to register {model_id}:[/red] {exc}")
                log.error("symlink failed for %s: %s", model_id, exc)
    else:
        console.print("\n[dim](--no-register: skipping symlink creation)[/dim]")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    found_count = len(discovered)
    console.print(
        f"\n[dim]Summary:[/dim] Found [bold]{found_count}[/bold] new model{'s' if found_count != 1 else ''}, "
        f"registered [bold]{registered_count}[/bold]"
    )
