"""
ouro/cli/main.py — Root Typer application for Ouro CLI.

All subcommands are registered here.
"""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(
    name="ouro",
    help="[bold cyan]Ouro[/bold cyan] — MLX-native model runner",
    rich_markup_mode="rich",
    no_args_is_help=True,
)

console = Console()

# ---------------------------------------------------------------------------
# Import and register subcommands
# ---------------------------------------------------------------------------

from ouro.cli.pull import pull_command  # noqa: E402
from ouro.cli.list_cmd import list_command  # noqa: E402
from ouro.cli.rm import rm_command  # noqa: E402
from ouro.cli.run import run_command  # noqa: E402
from ouro.cli.serve import serve_command, stop_command  # noqa: E402
from ouro.cli.create import create_command  # noqa: E402
from ouro.cli.ps import ps_command  # noqa: E402

app.command("pull")(pull_command)
app.command("list")(list_command)
app.command("rm")(rm_command)
app.command("run")(run_command)
app.command("serve")(serve_command)
app.command("stop")(stop_command)
app.command("create")(create_command)
app.command("ps")(ps_command)


def main() -> None:
    """Entry point for the Ouro CLI."""
    app()


if __name__ == "__main__":
    main()
