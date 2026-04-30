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
    no_args_is_help=False,
    invoke_without_command=True,
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
from ouro.cli.scan import scan_command  # noqa: E402
from ouro.cli.service import install_command, uninstall_command, status_command  # noqa: E402

app.command("pull")(pull_command)
app.command("list")(list_command)
app.command("rm")(rm_command)
app.command("run")(run_command)
app.command("serve")(serve_command)
app.command("stop")(stop_command)
app.command("create")(create_command)
app.command("ps")(ps_command)
app.command("scan")(scan_command)

# Service management (launchd auto-start)
service_app = typer.Typer(
    name="service",
    help="Install/uninstall Ouro as a macOS login service (auto-start on boot).",
    rich_markup_mode="rich",
)
app.add_typer(service_app, name="service")
service_app.command("install")(install_command)
service_app.command("uninstall")(uninstall_command)
service_app.command("status")(status_command)


@app.callback()
def main_callback(ctx: typer.Context) -> None:
    """Ouro — MLX-native model runner for Apple Silicon."""
    if ctx.invoked_subcommand is None:
        from ouro.cli.welcome import show_welcome
        show_welcome()


def main() -> None:
    """Entry point for the Ouro CLI."""
    app()


if __name__ == "__main__":
    main()
