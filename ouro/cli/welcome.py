"""
ouro/cli/welcome.py — Ouro welcome/banner screen.

Displayed when `ouro` is invoked with no arguments.
Replaces the default --help with a Hermes-Agent-style TUI dashboard.
"""

from __future__ import annotations

from typing import Any, Dict, List

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text


# ---------------------------------------------------------------------------
# ASCII art assets
# ---------------------------------------------------------------------------

# Eye + snakes ASCII art (detailed — two snakes coiled around a central eye)
EYE_SNAKES_ART = r"""
      /\   ,'`-.        .-'`\   /\
     /  `~(  ( o)~~)~~((o )  )~'  \
    / ,-'  |  `~=~'    `~=~'  | `-.\ 
   / /  _.-\   /~~~~~~~~~~~~~~\  /-._\ \
  | | ,'   _> |  .==========.  | <_   `.| 
  | |/  ,-'   | //  O U R O \\ |   `-, \| 
  |  \ /  ,-.  \|| ,========. ||/  ,-. / |
  |  |Y  / __|  || |  #  #  | ||  |__ \ Y|
 /   ||  \(  )\ || |  # # #  | || /(  )/  ||   \
|  ,-'|   `--'  || |#########| ||  `--'   |`-.  |
|  \  |  snake  || \=========/ ||  snake  |  /  |
|   \ | ~~~~~~~  \\  `-------'  // ~~~~~~~| /   |
 \   `|  (o  o)   `\___________/'  (o  o) |'   /
  \   \   \  /   ,~~~~|=====|~~~~.   \  /   /  /
   \   `.  \/  ,'  /  |     |  \  `.  \/  ,'  /
    \    `-.  /  /   ||  @  ||   \  \  .-'   /
     \      `/ ,'    ||.   .||    `. \/`    /
      `\    / /  _   |' `-' `|   _  \ \   /'
        `--' | ( )  / \     / \  ( ) | `--'
             |  `-./   `---'   \.-'  |
             \    /  ,-. .-. ,-.\ .  /
              \  / ,' `Y' `Y' `. \  /
               \/  /   |   |   \ \/
               /  / ,--'   `--. \ \
              /  /  \  OURO  /  \ \
             /  /    `------'    \ \
            `--'                  `--'"""

# Fallback big "OURO" title if pyfiglet is unavailable
OURO_FALLBACK_TITLE = r"""
  ___  _   _ ____   ___  
 / _ \| | | |  _ \ / _ \ 
| | | | | | | |_) | | | |
| |_| | |_| |  _ <| |_| |
 \___/ \___/|_| \_\\___/ 
"""


def _get_ouro_title() -> str:
    """Return a big block-font 'OURO' title string."""
    try:
        import pyfiglet  # type: ignore[import]
        # Try preferred fonts, fall back gracefully
        for font in ("banner3", "doom", "block", "standard"):
            try:
                rendered = pyfiglet.figlet_format("OURO", font=font)
                if rendered.strip():
                    return rendered
            except Exception:
                continue
    except ImportError:
        pass
    return OURO_FALLBACK_TITLE


def _get_installed_models() -> List[Dict[str, Any]]:
    """Return list of installed models from the registry, or empty list on error."""
    try:
        from ouro.registry import storage  # type: ignore[import]
        return storage.list_installed_models()
    except Exception:
        return []


def _get_version() -> str:
    """Return the Ouro package version."""
    try:
        from importlib.metadata import version
        return version("ouro")
    except Exception:
        return "0.1.0"


# ---------------------------------------------------------------------------
# Main welcome renderer
# ---------------------------------------------------------------------------

def show_welcome() -> None:
    """Render the Hermes-Agent-style Ouro welcome screen."""
    console = Console()

    amber = "bold #FFA500"
    gold = "#D4A017"
    brown_border = "#8B4513"
    dim_gold = "dim yellow"

    # -----------------------------------------------------------------------
    # 1. Big OURO title
    # -----------------------------------------------------------------------
    title_text = _get_ouro_title()
    title = Text(title_text, style=amber, justify="center")
    console.print(title)

    # Sub-tagline
    tagline = Text(
        "  ◈  MLX-Native Model Runner for Apple Silicon  ◈  ",
        style=gold,
        justify="center",
    )
    console.print(tagline)
    console.print()

    # -----------------------------------------------------------------------
    # 2. Left panel: ASCII art logo
    # -----------------------------------------------------------------------
    logo_text = Text(EYE_SNAKES_ART, style=amber, justify="center")
    logo_panel = Panel(
        logo_text,
        title="[dim yellow]∴ ouro ∴[/dim yellow]",
        border_style=brown_border,
        padding=(0, 1),
        expand=False,
    )

    # -----------------------------------------------------------------------
    # 3. Right panel: System info + capabilities
    # -----------------------------------------------------------------------
    version = _get_version()

    info_lines = Text(justify="left")
    info_lines.append(f"Ouro  v{version}\n", style=amber)
    info_lines.append("─" * 30 + "\n", style=dim_gold)
    info_lines.append("\n")
    info_lines.append("Platform  ", style="dim")
    info_lines.append("Apple Silicon (MLX-native)\n", style=gold)
    info_lines.append("Backend   ", style="dim")
    info_lines.append("mlx-lm / GGUF\n", style=gold)
    info_lines.append("API       ", style="dim")
    info_lines.append("OpenAI-compatible REST\n", style=gold)
    info_lines.append("\n")
    info_lines.append("COMMANDS\n", style=amber)
    info_lines.append("─" * 30 + "\n", style=dim_gold)

    commands = [
        ("ouro pull  <model>", "Download a model from HuggingFace"),
        ("ouro run   <model>", "Run a model interactively"),
        ("ouro serve <model>", "Start OpenAI-compatible API server"),
        ("ouro list         ", "List all installed models"),
        ("ouro ps           ", "Show running model servers"),
        ("ouro stop  <model>", "Stop a running server"),
        ("ouro scan         ", "Scan cache for MLX models"),
        ("ouro rm    <model>", "Remove a model from disk"),
        ("ouro create <name>", "Create a model from a Modelfile"),
    ]

    for cmd, desc in commands:
        info_lines.append(f"  {cmd}  ", style="cyan")
        info_lines.append(f"{desc}\n", style="white")

    info_panel = Panel(
        info_lines,
        title=f"[bold yellow]◈ System Info[/bold yellow]",
        border_style=brown_border,
        padding=(0, 2),
        expand=True,
    )

    # Render the two panels side-by-side
    console.print(Columns([logo_panel, info_panel], equal=False, expand=True))
    console.print()

    # -----------------------------------------------------------------------
    # 4. Installed models table
    # -----------------------------------------------------------------------
    console.print(Rule(
        title="[bold yellow]◈ Installed Models[/bold yellow]",
        style=brown_border,
    ))
    console.print()

    models = _get_installed_models()

    if not models:
        no_models = Text(
            "  No models installed — run  ouro pull <model>  to download one.\n",
            style="dim",
            justify="center",
        )
        no_models_panel = Panel(
            no_models,
            border_style=dim_gold,
            padding=(0, 2),
        )
        console.print(no_models_panel)
    else:
        table = Table(
            show_header=True,
            header_style=amber,
            border_style=brown_border,
            show_lines=False,
            expand=True,
            row_styles=["", "dim"],
        )
        table.add_column("Model Name", style="cyan", no_wrap=False, ratio=4)
        table.add_column("Size (GB)", justify="right", style="green", ratio=1)
        table.add_column("Modified", style="magenta", ratio=2)

        for model in models:
            name = str(model.get("id", ""))
            size_mb = model.get("size_mb", 0)
            modified = str(model.get("modified", ""))

            if isinstance(size_mb, (int, float)) and size_mb:
                size_str = f"{size_mb / 1024:.2f} GB"
            else:
                size_str = "—"

            table.add_row(name, size_str, modified)

        console.print(table)

    console.print()

    # -----------------------------------------------------------------------
    # 5. Footer
    # -----------------------------------------------------------------------
    n = len(models)
    model_word = "model" if n == 1 else "models"
    footer = Text(justify="center")
    footer.append(f"  {n} {model_word} available", style=gold)
    footer.append("  ·  ", style="dim")
    footer.append("ouro run <model>", style="cyan")
    footer.append(" to start", style="dim")
    footer.append("  ·  ", style="dim")
    footer.append("ouro pull <model>", style="cyan")
    footer.append(" to download  \n", style="dim")

    console.print(Panel(footer, border_style=dim_gold, padding=(0, 0)))
