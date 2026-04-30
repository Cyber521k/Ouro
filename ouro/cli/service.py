"""
ouro/cli/service.py — Install / uninstall Ouro as a macOS login service.

Usage:
    ouro service install   [--host 127.0.0.1] [--port 5215]
    ouro service uninstall
    ouro service status
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import typer
from rich.console import Console

console = Console()

LABEL = "com.ouro.server"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def _ouro_bin() -> str:
    """Locate the ouro executable."""
    import shutil
    path = shutil.which("ouro")
    if path:
        return path
    # Try common venv/homebrew paths
    for candidate in [
        "/opt/homebrew/bin/ouro",
        str(Path.home() / ".local" / "bin" / "ouro"),
        str(Path.home() / ".venv" / "bin" / "ouro"),
    ]:
        if Path(candidate).exists():
            return candidate
    raise RuntimeError(
        "Cannot locate the 'ouro' binary.  Make sure it is installed: pip install -e ~/Ouro"
    )


def _build_plist(ouro_bin: str, host: str, port: int) -> str:
    log_dir = Path.home() / ".ouro" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build PATH that includes Homebrew, local bin, and the ouro binary's parent
    ouro_parent = str(Path(ouro_bin).parent)
    path_dirs = [
        ouro_parent,
        "/opt/homebrew/bin",
        "/usr/local/bin",
        "/usr/bin",
        "/bin",
        "/usr/sbin",
        "/sbin",
        str(Path.home() / ".local" / "bin"),
    ]
    path_str = ":".join(dict.fromkeys(path_dirs))  # deduplicate, preserve order

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{ouro_bin}</string>
        <string>serve</string>
        <string>--host</string>
        <string>{host}</string>
        <string>--port</string>
        <string>{port}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log_dir}/ouro.out.log</string>
    <key>StandardErrorPath</key>
    <string>{log_dir}/ouro.err.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{path_str}</string>
        <key>HOME</key>
        <string>{Path.home()}</string>
    </dict>
</dict>
</plist>
"""


def _launchctl(*args: str) -> tuple[int, str]:
    result = subprocess.run(
        ["launchctl", *args],
        capture_output=True, text=True
    )
    return result.returncode, (result.stdout + result.stderr).strip()


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------

def install_command(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host"),
    port: int = typer.Option(5215, "--port", help="Bind port"),
) -> None:
    """
    [bold]Install[/bold] Ouro as a macOS login item (launchd agent).

    Ouro will start automatically when you log in — no terminal required.
    All models listed in [cyan]~/.ouro/config.yaml[/cyan] will be loaded.

    Logs:
      [cyan]~/.ouro/logs/ouro.out.log[/cyan]
      [cyan]~/.ouro/logs/ouro.err.log[/cyan]
    """
    try:
        ouro_bin = _ouro_bin()
    except RuntimeError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    plist_content = _build_plist(ouro_bin, host, port)

    # Write the plist
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLIST_PATH.write_text(plist_content)
    console.print(f"[green]✓[/green] Plist written → [cyan]{PLIST_PATH}[/cyan]")

    # Unload stale entry if present
    _launchctl("unload", str(PLIST_PATH))

    # Load (registers + starts immediately)
    rc, out = _launchctl("load", "-w", str(PLIST_PATH))
    if rc != 0 and out:
        console.print(f"[yellow]launchctl load warning:[/yellow] {out}")
    else:
        console.print(f"[green]✓[/green] Service loaded and started")

    console.print(
        f"\n[bold green]Ouro is now a login service![/bold green]\n"
        f"  Endpoint : [cyan]http://{host}:{port}/v1[/cyan]\n"
        f"  Logs     : [cyan]~/.ouro/logs/[/cyan]\n"
        f"  Stop now : [bold]ouro service uninstall[/bold]\n"
        f"  Status   : [bold]ouro service status[/bold]\n"
    )


# ---------------------------------------------------------------------------
# uninstall
# ---------------------------------------------------------------------------

def uninstall_command() -> None:
    """
    [bold]Uninstall[/bold] the Ouro login service (stops it and removes the plist).
    """
    if not PLIST_PATH.exists():
        console.print("[yellow]No Ouro service plist found — already removed?[/yellow]")
        raise typer.Exit(code=0)

    rc, out = _launchctl("unload", "-w", str(PLIST_PATH))
    if out:
        console.print(f"[dim]{out}[/dim]")

    PLIST_PATH.unlink(missing_ok=True)
    console.print(f"[green]✓[/green] Ouro service uninstalled.")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def status_command() -> None:
    """
    [bold]Show[/bold] the status of the Ouro login service.
    """
    rc, out = _launchctl("list", LABEL)
    plist_exists = PLIST_PATH.exists()

    if rc != 0 or not out or "Could not find service" in out:
        state = "[red]not running[/red]"
    else:
        state = "[green]running[/green]"

    console.print(f"Service label : [bold]{LABEL}[/bold]")
    console.print(f"Plist         : {'[green]exists[/green]' if plist_exists else '[red]missing[/red]'} → {PLIST_PATH}")
    console.print(f"Status        : {state}")
    if out and rc == 0:
        console.print(f"\nlaunchctl info:\n[dim]{out}[/dim]")
