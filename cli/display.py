"""Rich display panels for real-time pipeline streaming."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

# Agent role → display color
AGENT_COLORS = {
    "actor": "blue",
    "checker": "yellow",
    "dafny_verifier": "magenta",
    "policy": "green",
}


def show_agent_start(agent: str) -> None:
    """Display an agent starting panel."""
    color = AGENT_COLORS.get(agent, "white")
    console.print(
        Panel(
            f"[bold {color}]Agent [{agent.upper()}] starting...[/]",
            border_style=color,
        )
    )


def show_agent_output(agent: str, data: dict[str, Any]) -> None:
    """Display an agent's output summary."""
    color = AGENT_COLORS.get(agent, "white")
    lines: list[str] = []
    for key, value in data.items():
        lines.append(f"  {key}: {value}")

    console.print(
        Panel(
            "\n".join(lines) if lines else "Done",
            title=f"[bold {color}]{agent.upper()} Output[/]",
            border_style=color,
        )
    )


def show_iteration_complete(data: dict[str, Any]) -> None:
    """Display iteration completion summary."""
    iteration = data.get("iteration", "?")
    all_pass = data.get("all_pass", False)
    summary = data.get("summary", "")

    status_icon = "[bold green]PASS[/]" if all_pass else "[bold red]FAIL[/]"

    console.print(
        Panel(
            f"Status: {status_icon}\n{summary}",
            title=f"Iteration {iteration} Complete",
            border_style="green" if all_pass else "red",
        )
    )


def show_pipeline_complete(data: dict[str, Any]) -> None:
    """Display final pipeline result."""
    status = data.get("status", "unknown")
    iterations = data.get("iterations", 0)
    run_id = data.get("run_id", "")

    if status in ("completed", "awaiting_approval"):
        console.print(
            Panel(
                f"[bold green]Pipeline SUCCEEDED[/]\n"
                f"Iterations: {iterations}\n"
                f"Run ID: {run_id}",
                title="Pipeline Complete",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                f"[bold red]Pipeline {status.upper()}[/]\n"
                f"Iterations: {iterations}\n"
                f"Run ID: {run_id}",
                title="Pipeline Complete",
                border_style="red",
            )
        )


def show_error(message: str) -> None:
    console.print(f"[bold red]Error:[/] {message}")
