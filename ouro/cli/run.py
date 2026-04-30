"""
ouro/cli/run.py — Run a model in interactive REPL or one-shot mode.

Usage:
    ouro run <model> [prompt]
    ouro run <model> --system "You are helpful." --temperature 0.7 --max-tokens 512
"""

from __future__ import annotations

import sys
from typing import Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

console = Console()

# ---------------------------------------------------------------------------
# REPL help text
# ---------------------------------------------------------------------------

REPL_HELP = """\
[bold cyan]Ouro REPL Commands[/bold cyan]
  [green]/bye[/green]          — Exit the REPL
  [green]/clear[/green]        — Reset conversation history (model stays loaded)
  [green]/system <msg>[/green] — Set or override the system prompt
  [green]/help[/green]         — Show this help message
"""


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------

def run_command(
    model: str = typer.Argument(..., help="Model name or path to run"),
    prompt: Optional[str] = typer.Argument(None, help="One-shot prompt; omit for interactive REPL"),
    system: Optional[str] = typer.Option(None, "--system", help="System prompt"),
    temperature: float = typer.Option(0.6, "--temperature", help="Sampling temperature"),
    max_tokens: int = typer.Option(8192, "--max-tokens", help="Maximum tokens to generate"),
    top_p: float = typer.Option(0.95, "--top-p", help="Nucleus sampling probability"),
    min_p: float = typer.Option(0.0, "--min-p", help="Min-p sampling (0.0 = disabled)"),
    repetition_penalty: float = typer.Option(1.0, "--rep-penalty", help="Repetition penalty (1.0 = off)"),
    no_thinking: bool = typer.Option(False, "--no-thinking", help="For Qwen3: disable <think> reasoning phase"),
) -> None:
    """
    [bold]Run[/bold] a model interactively or in one-shot mode.

    If a [cyan]prompt[/cyan] argument is provided the model generates a single response
    and exits.  Otherwise an interactive REPL is started.
    """
    # ------------------------------------------------------------------
    # Lazy imports so the file always parses even without MLX installed
    # ------------------------------------------------------------------
    try:
        from ouro.registry import storage  # type: ignore[import]
    except ImportError:
        console.print("[red]Error:[/red] ouro.registry.storage module not found.")
        raise typer.Exit(code=1)

    try:
        from ouro.engine import loader as engine_loader  # type: ignore[import]
        from ouro.engine import prompt_builder  # type: ignore[import]
        from ouro.engine import generate as engine_generate  # type: ignore[import]
    except ImportError as exc:
        console.print(f"[red]Error importing engine modules:[/red] {exc}")
        raise typer.Exit(code=1)

    # ------------------------------------------------------------------
    # Resolve model path & load
    # ------------------------------------------------------------------
    try:
        model_path = storage.resolve_model_path(model)
    except Exception as exc:
        console.print(f"[red]Could not resolve model path:[/red] {exc}")
        raise typer.Exit(code=1)

    console.print(f"[cyan]Loading model[/cyan] [bold]{model}[/bold] from {model_path} …")

    try:
        loaded_model, tokenizer = engine_loader.load_model(model_path)
    except Exception as exc:
        console.print(f"[red]Failed to load model:[/red] {exc}")
        raise typer.Exit(code=1)

    gen_params: dict = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }
    if min_p > 0.0:
        gen_params["min_p"] = min_p
    if repetition_penalty != 1.0:
        gen_params["repetition_penalty"] = repetition_penalty

    def _inject_thinking(msgs: list) -> list:
        """For Qwen3 thinking models: append /no_think to the last user message to skip reasoning."""
        if not no_thinking:
            return msgs
        result = list(msgs)
        for i in range(len(result) - 1, -1, -1):
            if result[i].get("role") == "user":
                result[i] = dict(result[i])
                content = result[i].get("content", "")
                if not content.endswith("/no_think"):
                    result[i]["content"] = content + " /no_think"
                break
        return result

    def _stream_response(msgs: list) -> str:
        """Build prompt, stream tokens to stdout, return full response text."""
        built_prompt = prompt_builder.build_prompt(tokenizer, _inject_thinking(msgs))
        token_stream = engine_generate.generate_stream(loaded_model, tokenizer, built_prompt, **gen_params)
        full_text = ""
        in_think = False
        console.print("[bold cyan]Assistant:[/bold cyan] ", end="")
        for token in token_stream:
            full_text += token
            # Dim the <think>...</think> block so it's visually distinct but still visible
            if "<think>" in token:
                in_think = True
            if in_think:
                console.print(f"[dim]{token}[/dim]", end="")
            else:
                console.print(token, end="")
            if "</think>" in token:
                in_think = False
        console.print()  # newline after last token
        return full_text

    # ------------------------------------------------------------------
    # One-shot mode
    # ------------------------------------------------------------------
    if prompt is not None:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            _stream_response(messages)
        except Exception as exc:
            console.print(f"[red]Generation error:[/red] {exc}")
            raise typer.Exit(code=1)
        return

    # ------------------------------------------------------------------
    # Interactive REPL
    # ------------------------------------------------------------------
    console.print(f"\n[bold cyan]Ouro[/bold cyan] — model: [bold]{model}[/bold]")
    console.print("Type [green]/help[/green] for commands, [green]/bye[/green] to exit.\n")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    while True:
        try:
            user_input = typer.prompt("You", prompt_suffix=" › ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Exiting.[/yellow]")
            break

        stripped = user_input.strip()

        # ---- Built-in REPL commands ----
        if stripped == "/bye":
            console.print("[yellow]Bye![/yellow]")
            break

        if stripped == "/clear":
            # Keep system message if present, drop everything else
            messages = [m for m in messages if m.get("role") == "system"]
            console.print("[dim]Conversation history cleared.[/dim]")
            continue

        if stripped.startswith("/system "):
            new_system = stripped[len("/system "):].strip()
            # Replace or insert system message at position 0
            messages = [m for m in messages if m.get("role") != "system"]
            messages.insert(0, {"role": "system", "content": new_system})
            console.print(f"[dim]System prompt updated.[/dim]")
            continue

        if stripped == "/help":
            console.print(REPL_HELP)
            continue

        if not stripped:
            continue

        # ---- Regular user message ----
        messages.append({"role": "user", "content": stripped})

        try:
            response = _stream_response(messages)
        except Exception as exc:
            console.print(f"[red]Generation error:[/red] {exc}")
            # Remove last user message so conversation stays consistent
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": response})
