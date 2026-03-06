"""HPEMA CLI — typer + rich interface for the generation pipeline.

Usage:
    hpema generate --requirement "Implement altitude hold controller" --standard DO_178C
    hpema audit --run-id <uuid>
    hpema dashboard
"""

from __future__ import annotations

import json
from typing import Optional
from uuid import UUID

import httpx
import typer
from rich.console import Console

from cli.config import (
    DEFAULT_BACKEND_URL,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_STANDARD,
)
from cli.display import (
    show_agent_output,
    show_agent_start,
    show_error,
    show_iteration_complete,
    show_pipeline_complete,
)

app = typer.Typer(
    name="hpema",
    help="HPEMA — Hierarchical Policy-Enforced Multi-Agent code generation pipeline",
)
console = Console()


@app.command()
def generate(
    requirement: str = typer.Option(..., "--requirement", "-r", help="Natural language requirement"),
    standard: str = typer.Option(DEFAULT_STANDARD, "--standard", "-s", help="Safety standard (DO_178C, MISRA_C, NASA)"),
    language: str = typer.Option(DEFAULT_LANGUAGE, "--language", "-l", help="Target language (C, SPARK_Ada)"),
    max_iterations: int = typer.Option(DEFAULT_MAX_ITERATIONS, "--max-iterations", "-m", help="Max pipeline iterations"),
    backend_url: str = typer.Option(DEFAULT_BACKEND_URL, "--backend", help="Backend API URL"),
) -> None:
    """Run the full generation pipeline with real-time streaming."""
    console.print(f"\n[bold]HPEMA Pipeline[/]")
    console.print(f"  Requirement: {requirement}")
    console.print(f"  Standard: {standard}")
    console.print(f"  Language: {language}")
    console.print(f"  Max iterations: {max_iterations}\n")

    payload = {
        "requirement_text": requirement,
        "safety_standard": standard,
        "target_language": language,
        "max_iterations": max_iterations,
    }

    url = f"{backend_url}/api/v1/pipeline/run"

    try:
        with httpx.stream("POST", url, json=payload, timeout=300.0) as response:
            if response.status_code != 200:
                show_error(f"Backend returned {response.status_code}")
                raise typer.Exit(1)

            # Parse SSE events
            event_type = None
            for line in response.iter_lines():
                if line.startswith("event: "):
                    event_type = line[7:]
                elif line.startswith("data: ") and event_type:
                    data = json.loads(line[6:])
                    _handle_event(event_type, data)
                    event_type = None

    except httpx.ConnectError:
        show_error(f"Cannot connect to backend at {backend_url}. Is it running?")
        raise typer.Exit(1)


def _handle_event(event_type: str, data: dict) -> None:
    """Route SSE events to the appropriate display handler."""
    agent = data.get("agent")

    if event_type == "agent_start" and agent:
        show_agent_start(agent)
    elif event_type == "agent_output" and agent:
        show_agent_output(agent, data.get("data", {}))
    elif event_type == "iteration_complete":
        show_iteration_complete(data.get("data", {}))
    elif event_type == "pipeline_complete":
        show_pipeline_complete(data.get("data", {}))
    elif event_type == "agent_error":
        show_error(str(data.get("data", {}).get("error", "Unknown error")))


@app.command()
def audit(
    run_id: str = typer.Option(..., "--run-id", help="Pipeline run UUID"),
    backend_url: str = typer.Option(DEFAULT_BACKEND_URL, "--backend", help="Backend API URL"),
) -> None:
    """View the audit trail for a pipeline run."""
    url = f"{backend_url}/api/v1/audit/run/{run_id}"

    try:
        response = httpx.get(url, timeout=30.0)
        if response.status_code != 200:
            show_error(f"Backend returned {response.status_code}")
            raise typer.Exit(1)

        data = response.json()
        entries = data.get("entries", [])

        if not entries:
            console.print(f"No audit entries found for run {run_id}")
            return

        console.print(f"\n[bold]Audit Trail — Run {run_id}[/]")
        console.print(f"Total entries: {len(entries)}\n")

        for entry in entries:
            ts = entry.get("timestamp", "")
            evt = entry.get("event_type", "")
            agent = entry.get("agent", "")
            console.print(f"  [{ts}] {evt}" + (f" ({agent})" if agent else ""))

    except httpx.ConnectError:
        show_error(f"Cannot connect to backend at {backend_url}")
        raise typer.Exit(1)


@app.command()
def dashboard(
    port: int = typer.Option(8501, "--port", help="Streamlit port"),
) -> None:
    """Launch the Streamlit audit dashboard."""
    import subprocess
    import sys

    console.print(f"[bold]Launching HPEMA Dashboard on port {port}...[/]")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "dashboard/app.py", "--server.port", str(port)],
    )


if __name__ == "__main__":
    app()
