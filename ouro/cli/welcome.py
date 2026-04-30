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

# Block-style spiral/eye logo
EYE_SNAKES_ART = r"""
                                                               
                                                               
                                                               
                                                               
                                                               
                ██████████████████████                         
          ██████████████████████████████████                   
         ██   █████████████████████████████████░               
        ███  ░████████████████████████████████████             
        ██████ ████████           ▓█████████████████           
       ████████ █████                    █████████████         
      ███████    ░███                       ████████████       
  ██████████     ████                         ███████████      
█ ░██████      ██████                           ██████████     
 ██████  ██   ███████                            ██████████    
     ▒  ██    ██████                              ██████████   
       ██░  ██████                                 █████████   
      ███                                           █████████  
      ███                                            ████████  
      ███                                             ████████ 
     ███                                              ████████ 
     ██▒                                               ████████
    ███                                                ████████
    ███                                                ████████
    ███                                                ████████
    ███                                                ████████
    ███                                                ███████ 
    ████                                              ████████ 
    ▓███                                             ████████░ 
     ███░                                            ████████  
     ████                                           ████████   
     ▒████                                         ████████    
      ▓████                                       ████████     
       █████                                     ████████      
        ▓█████                                 █████████       
         ░█████                               █████████        
           ██████                          ███████████         
            ████████                   ██████████████          
              ████████████     ▒░█████████████████             
                 ███████████████████████████████               
                     ▓██████████████████████                   
                                                               
                                                               
                                                               
                                                               
                                                               """


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


def _model_perf_bar(size_mb: float, ram_gb: float) -> Text:
    """Return a small color bar showing how well a model will run on this machine.

    Layout:  [██░░░░]
      red    = won't run  (model > 90% RAM)
      orange = slow       (model 60–90% RAM)
      green  = fast       (model < 60% RAM)
    """
    bar_width = 6
    model_gb = size_mb / 1024 if size_mb else 0

    if ram_gb <= 0:
        # Can't determine — show neutral grey bar
        bar = Text()
        bar.append(" [", style="dim")
        bar.append("░" * bar_width, style="dim")
        bar.append("]", style="dim")
        return bar

    ratio = model_gb / ram_gb  # 0.0 → 1.0+

    if ratio >= 0.9:
        # Red — won't run
        filled = bar_width
        color = "bold red"
        label = " ✗"
        label_style = "bold red"
    elif ratio >= 0.6:
        # Orange — will run slow
        filled = round(bar_width * ratio)
        color = "bold yellow"
        label = " ~"
        label_style = "bold yellow"
    else:
        # Green — will run fast
        filled = round(bar_width * ratio)
        color = "bold green"
        label = " ✓"
        label_style = "bold green"

    empty = bar_width - filled
    bar = Text()
    bar.append(" [", style="dim")
    bar.append("█" * filled, style=color)
    bar.append("░" * empty, style="dim")
    bar.append("]", style="dim")
    bar.append(label, style=label_style)
    return bar


def _get_machine_info() -> dict:
    """Auto-detect machine hardware, OS, CPU, RAM, and GPU/chip info."""
    import platform
    import sys

    info: dict = {}

    # OS / platform
    system = platform.system()
    info["os"] = f"{system} {platform.release()}"

    # CPU
    info["cpu"] = platform.processor() or platform.machine()

    # RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        info["ram"] = f"{ram_gb:.0f} GB"
    except ImportError:
        info["ram"] = "unknown"

    # Apple Silicon detection (chip name + GPU cores)
    info["chip"] = None
    info["gpu"] = None
    if system == "Darwin":
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=3
            )
            chip = result.stdout.strip()
            if chip:
                info["chip"] = chip
        except Exception:
            pass

        # Try system_profiler for Apple Silicon GPU core count
        try:
            import subprocess, json
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType", "-json"],
                capture_output=True, text=True, timeout=5
            )
            data = json.loads(result.stdout)
            hw = data.get("SPHardwareDataType", [{}])[0]
            # chip name from system_profiler (more reliable on M-series)
            chip_sp = hw.get("chip_type") or hw.get("cpu_type", "")
            if chip_sp and not info["chip"]:
                info["chip"] = chip_sp
            elif chip_sp:
                info["chip"] = chip_sp  # prefer system_profiler name
            # GPU cores
            gpu_cores = hw.get("number_processors", "")
            # Look for graphics info
            result2 = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True, text=True, timeout=5
            )
            disp = json.loads(result2.stdout)
            gpu_data = disp.get("SPDisplaysDataType", [{}])[0]
            gpu_name = gpu_data.get("sppci_model", "") or gpu_data.get("_name", "")
            if gpu_name:
                info["gpu"] = gpu_name
        except Exception:
            pass

    # Python version
    info["python"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    return info


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

    tagline = Text(
        "  ◈  MLX-Native Model Runner for Apple Silicon  ◈  ",
        style=gold,
        justify="center",
    )
    console.print(tagline)
    console.print()

    # -----------------------------------------------------------------------
    # 2. Logo panel (full width)
    # -----------------------------------------------------------------------
    logo_text = Text(EYE_SNAKES_ART, style=amber, justify="center")
    logo_panel = Panel(
        logo_text,
        title="[dim yellow]∴ ouro ∴[/dim yellow]",
        border_style=brown_border,
        padding=(0, 1),
        expand=True,
    )
    console.print(logo_panel)
    console.print()

    # -----------------------------------------------------------------------
    # 3. System info panel
    # -----------------------------------------------------------------------
    version = _get_version()
    machine = _get_machine_info()

    info_lines = Text(justify="left")
    info_lines.append(f"Ouro  v{version}\n", style=amber)
    info_lines.append("─" * 30 + "\n", style=dim_gold)
    info_lines.append("\n")
    info_lines.append("OS        ", style="dim")
    info_lines.append(f"{machine['os']}\n", style=gold)
    info_lines.append("Chip      ", style="dim")
    info_lines.append(f"{machine['chip'] or machine['cpu']}\n", style=gold)
    info_lines.append("RAM       ", style="dim")
    info_lines.append(f"{machine['ram']}\n", style=gold)
    if machine.get("gpu"):
        info_lines.append("GPU       ", style="dim")
        info_lines.append(f"{machine['gpu']}\n", style=gold)
    info_lines.append("Python    ", style="dim")
    info_lines.append(f"{machine['python']}\n", style=gold)
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
        title="[bold yellow]◈ System Info[/bold yellow]",
        border_style=brown_border,
        padding=(0, 2),
        expand=True,
    )
    console.print(info_panel)
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
        # Get RAM for perf bar
        try:
            import psutil
            _ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            _ram_gb = 0

        for model in models:
            name = str(model.get("id", ""))
            size_mb = model.get("size_mb", 0)
            modified = str(model.get("modified", ""))

            if isinstance(size_mb, (int, float)) and size_mb:
                size_str = f"{size_mb / 1024:.2f} GB"
            else:
                size_str = "—"

            line = Text()
            line.append("  ● ", style="bold yellow")
            line.append(f"{name}", style="cyan")
            line.append(f"    {size_str}", style="green")
            line.append(f"    {modified}", style="dim magenta")
            line.append_text(_model_perf_bar(size_mb if isinstance(size_mb, (int, float)) else 0, _ram_gb))
            console.print(line)

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
