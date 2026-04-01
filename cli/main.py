"""HPEMA CLI — interactive REPL + one-shot commands.

Usage:
    python -m cli.main                 # Interactive REPL (default)
    python -m cli.main generate ...    # One-shot generation
    python -m cli.main audit ...       # Query audit trail
"""

from __future__ import annotations

import json
from typing import Optional

import typer
from rich.console import Console

from backend.api.schemas.pipeline import PipelineStage
from cli.config import (
    DEFAULT_BACKEND_URL,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_STANDARD,
)
from cli.display import DisplayManager, show_error

app = typer.Typer(
    name="hpema",
    help="HPEMA — Hierarchical Policy-Enforced Multi-Agent code generation pipeline",
    invoke_without_command=True,
)
console = Console()


@app.callback()
def default(ctx: typer.Context) -> None:
    """Launch interactive REPL if no subcommand is given."""
    if ctx.invoked_subcommand is None:
        from cli.repl import start_repl
        start_repl()


@app.command()
def generate(
    requirement: str = typer.Option(..., "--requirement", "-r", help="Natural language requirement"),
    standard: str = typer.Option(DEFAULT_STANDARD, "--standard", "-s", help="Safety standard"),
    language: str = typer.Option(DEFAULT_LANGUAGE, "--language", "-l", help="Target language (Python, C, SPARK_Ada)"),
    max_iterations: int = typer.Option(DEFAULT_MAX_ITERATIONS, "--max-iterations", "-m", help="Max iterations"),
    stage: str = typer.Option("policy", "--stage", help="Pipeline stage: actor, checker, policy"),
) -> None:
    """Run the pipeline once (non-interactive)."""
    from cli.runner import run_pipeline

    try:
        pipeline_stage = PipelineStage(stage)
    except ValueError:
        show_error(f"Unknown stage: {stage}. Options: actor, checker, policy")
        raise typer.Exit(1)

    console.print(f"\n  [bold]HPEMA Pipeline[/]")
    console.print(f"  [dim]Standard: {standard}  |  Language: {language}  |  Max iterations: {max_iterations}  |  Stage: {stage}[/]")

    if pipeline_stage != PipelineStage.POLICY:
        from cli.display import show_disconnected_warning
        show_disconnected_warning(pipeline_stage)

    display = DisplayManager()

    try:
        state = run_pipeline(
            requirement=requirement,
            standard=standard,
            language=language,
            max_iterations=max_iterations,
            display=display,
            stage=pipeline_stage,
        )
    except KeyboardInterrupt:
        console.print("\n  [yellow]Pipeline interrupted.[/]")
        raise typer.Exit(1)
    except Exception as e:
        show_error(str(e))
        raise typer.Exit(1)


@app.command()
def audit(
    run_id: str = typer.Option(..., "--run-id", help="Pipeline run UUID"),
    backend_url: str = typer.Option(DEFAULT_BACKEND_URL, "--backend", help="Backend API URL"),
) -> None:
    """View the audit trail for a pipeline run (via backend API)."""
    import httpx

    url = f"{backend_url}/api/v1/audit/run/{run_id}"

    try:
        response = httpx.get(url, timeout=30.0)
        if response.status_code != 200:
            show_error(f"Backend returned {response.status_code}")
            raise typer.Exit(1)

        data = response.json()
        entries = data.get("entries", [])

        if not entries:
            console.print(f"  No audit entries found for run {run_id}")
            return

        console.print(f"\n  [bold]Audit Trail — Run {run_id}[/]")
        console.print(f"  Total entries: {len(entries)}\n")

        for entry in entries:
            ts = entry.get("timestamp", "")
            evt = entry.get("event_type", "")
            agent = entry.get("agent", "")
            console.print(f"  [{ts}] {evt}" + (f" ({agent})" if agent else ""))

    except httpx.ConnectError:
        show_error(f"Cannot connect to backend at {backend_url}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
