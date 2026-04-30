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


def _estimate_size_from_name(model_id: str) -> float:
    """Estimate model size in GB from its name using param count + quantization bits."""
    import re
    name = model_id.lower()

    # extract parameter count (e.g. 7b, 13b, 70b, 0.5b, 1.7b, 3b, 30b)
    param_match = re.search(r'(\d+\.?\d*)\s*b(?:[^a-z]|$)', name)
    params_b = float(param_match.group(1)) if param_match else 7.0  # default 7B

    # extract bits from quant suffix
    if 'bf16' in name or 'fp16' in name:
        bits = 16
    elif 'fp32' in name:
        bits = 32
    elif 'mxfp4' in name or 'nvfp4' in name or '4bit' in name or 'q4' in name or 'dq4' in name or '4-bit' in name:
        bits = 4
    elif '8bit' in name or 'q8' in name or '8-bit' in name or 'mxfp8' in name:
        bits = 8
    elif '6bit' in name or 'q6' in name:
        bits = 6
    elif '3bit' in name or 'q3' in name or 'dq3' in name:
        bits = 3
    elif '2bit' in name or 'q2' in name or 'dq2' in name:
        bits = 2
    else:
        bits = 4  # most mlx-community models are 4bit

    # bytes per param + ~10% overhead
    size_gb = round((params_b * 1e9 * bits / 8) / (1024 ** 3) * 1.1, 2)
    return max(size_gb, 0.1)


def _hf_cache_path() -> "Path":
    """Return the path to the local HuggingFace model cache file."""
    from pathlib import Path
    cache_dir = Path.home() / ".ouro" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "hf_mlx_models.json"


def _fetch_hf_models() -> List[Dict[str, Any]]:
    """Fetch mlx-community models from HuggingFace API and return a list of dicts."""
    import urllib.request
    import json

    results: Dict[str, Any] = {}  # keyed by full model_id to deduplicate
    seen_ids: set = set()
    page = 1
    per_page = 100

    while True:
        url = (
            f"https://huggingface.co/api/models"
            f"?author=mlx-community&filter=mlx&limit={per_page}&offset={(page-1)*per_page}"
            f"&sort=downloads&direction=-1"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ouro/0.1"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode())
        except Exception:
            break

        if not data:
            break

        for m in data:
            model_id = m.get("id", "")
            # Skip non-MLX repos: require the "mlx" tag to be present
            tags = m.get("tags", [])
            if not any(t.lower() == "mlx" or t.lower().startswith("mlx-") for t in tags):
                continue
            # estimate size from model name (e.g. 7B-4bit -> ~4GB, 27B-4bit -> ~14GB)
            size_gb = _estimate_size_from_name(model_id)

            seen_ids.add(model_id)
            results[model_id] = {
                "id": model_id.replace("mlx-community/", ""),
                "full_id": model_id,
                "size_gb": size_gb,
                "downloads": m.get("downloads", 0),
                "pull_cmd": f"ouro pull {model_id}",
            }

        if len(data) < per_page:
            break
        page += 1
        if page > 5:  # cap at 500 models
            break

    return list(results.values())


def _load_hf_cache() -> List[Dict[str, Any]]:
    """Load HuggingFace model cache from disk, or return empty list."""
    import json
    path = _hf_cache_path()
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return []


def _save_hf_cache(models: List[Dict[str, Any]]) -> None:
    """Save HuggingFace model list to local cache file."""
    import json
    import time
    path = _hf_cache_path()
    try:
        with open(path, "w") as f:
            json.dump({"fetched_at": time.time(), "models": models}, f)
    except Exception:
        pass


def _hf_cache_is_stale() -> bool:
    """Return True if cache is missing or older than 24 hours."""
    import json
    import time
    path = _hf_cache_path()
    try:
        with open(path, "r") as f:
            data = json.load(f)
        fetched_at = data.get("fetched_at", 0)
        return (time.time() - fetched_at) > 86400  # 24 hours
    except Exception:
        return True


def _get_hf_models_cached() -> List[Dict[str, Any]]:
    """Return HuggingFace mlx-community models, refreshing cache if stale."""
    import json
    path = _hf_cache_path()

    if _hf_cache_is_stale():
        # Fetch fresh data
        fresh = _fetch_hf_models()
        if fresh:
            _save_hf_cache(fresh)
            return fresh
        # If fetch failed, fall back to stale cache
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return data.get("models", [])
        except Exception:
            return []
    else:
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return data.get("models", [])
        except Exception:
            return []


def _get_recommended_models(ram_gb: float) -> List[Dict[str, Any]]:
    """Return mlx-community models from HuggingFace that will run fast on this machine.

    Fetches live from HF API and caches locally for 24 hours.
    Filters to models whose size_gb < 60% of available RAM (green zone).
    Falls back to a hardcoded list if HF is unreachable and cache is empty.
    """
    fast_threshold = ram_gb * 0.60

    # Try live/cached HF data first
    hf_models = _get_hf_models_cached()

    if hf_models:
        results = []
        for m in hf_models:
            size_gb = m.get("size_gb", 0)
            # Skip models with unknown size or that are too large
            if size_gb <= 0 or size_gb > fast_threshold:
                continue
            results.append({
                "id": m["id"],
                "size_gb": size_gb,
                "pull_cmd": m["pull_cmd"],
                "downloads": m.get("downloads", 0),
            })
        # Sort by downloads descending, take top 15
        results.sort(key=lambda x: x["downloads"], reverse=True)
        return results[:15]

    # Fallback hardcoded list if HF unreachable
    FALLBACK = [
        ("gemma-3-1b-it-qat-4bit",      0.7),
        ("gemma-3-4b-it-qat-4bit",      2.5),
        ("Qwen3.5-9B-OptiQ-4bit",       5.5),
        ("gemma-3-12b-it-qat-4bit",     7.5),
        ("Qwen2.5-14B-Instruct-4bit",   8.5),
        ("gpt-oss-20b-MXFP4-Q8",       13.0),
    ]
    return [
        {
            "id": name,
            "size_gb": size,
            "pull_cmd": f"ouro pull mlx-community/{name}",
            "downloads": 0,
        }
        for name, size in FALLBACK
        if size <= fast_threshold
    ]


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
    # 4. Recommended models (HuggingFace mlx-community, fast on this machine)
    # -----------------------------------------------------------------------
    try:
        import psutil
        _ram_gb_rec = psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        _ram_gb_rec = 0

    # Show a brief status if cache is stale (will fetch from HF)
    if _hf_cache_is_stale():
        console.print(Text("  ⟳ Refreshing model list from HuggingFace...", style="dim yellow"))

    recommended = _get_recommended_models(_ram_gb_rec)

    console.print(Rule(
        title="[bold yellow]◈ Recommended Models (runs fast on your machine)[/bold yellow]",
        style=brown_border,
    ))
    console.print()

    if recommended:
        for rec in recommended:
            size_str = f"{rec['size_gb']:.1f} GB"
            bar = _model_perf_bar(rec["size_gb"] * 1024, _ram_gb_rec)
            line = Text()
            line.append("  ● ", style="bold green")
            line.append(f"{rec['id']}", style="cyan")
            line.append(f"    {size_str}", style="green")
            line.append("    ")
            line.append(f"{rec['pull_cmd']}", style="dim yellow")
            line.append_text(bar)
            console.print(line)
    else:
        console.print(Text("  No recommendations available for your RAM size.", style="dim"))

    console.print()

    # -----------------------------------------------------------------------
    # 5. Installed models table
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
