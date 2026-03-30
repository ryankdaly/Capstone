"""Interactive REPL — Claude-Code-style interface for HPEMA.

Launch with: python -m cli.main
Type requirements naturally, use /commands to configure.
"""

from __future__ import annotations

import sys
from uuid import UUID

from rich.console import Console
from rich.table import Table

from backend.api.schemas.pipeline import PipelineState
from backend.config import load_config
from cli.display import (
    DisplayManager,
    console,
    show_banner,
    show_config,
    show_error,
    show_help,
)
from cli.runner import run_pipeline


class Session:
    """Holds REPL state across commands."""

    def __init__(self) -> None:
        config = load_config()
        self.standard: str = config.policies.default_standard
        self.language: str = "C"
        self.max_iterations: int = config.pipeline.max_iterations
        self.model: str = config.models.actor.model
        self.config_source: str = ""
        self.last_state: PipelineState | None = None

    @property
    def last_run_id(self) -> UUID | None:
        return self.last_state.run_id if self.last_state else None


def _handle_command(line: str, session: Session) -> bool:
    """Handle a /command. Returns True if the REPL should continue."""
    parts = line.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/quit", "/exit", "/q"):
        console.print("\n  [dim]Goodbye.[/]\n")
        return False

    elif cmd == "/help":
        show_help()

    elif cmd == "/config":
        show_config(session.standard, session.language, session.max_iterations, session.model)

    elif cmd == "/standard":
        if not arg:
            console.print(f"  Current standard: [bold]{session.standard}[/]")
            console.print("  [dim]Options: DO_178C, MISRA_C, NASA, Boeing_SDP[/]")
        else:
            session.standard = arg
            console.print(f"  Standard set to: [bold]{arg}[/]")

    elif cmd == "/language":
        if not arg:
            console.print(f"  Current language: [bold]{session.language}[/]")
            console.print("  [dim]Options: C, SPARK_Ada[/]")
        else:
            session.language = arg
            console.print(f"  Language set to: [bold]{arg}[/]")

    elif cmd == "/iterations":
        if not arg:
            console.print(f"  Current max iterations: [bold]{session.max_iterations}[/]")
        else:
            try:
                session.max_iterations = int(arg)
                console.print(f"  Max iterations set to: [bold]{arg}[/]")
            except ValueError:
                show_error("Iterations must be a number")

    elif cmd == "/last":
        _show_last_run(session)

    elif cmd == "/audit":
        _show_audit(session)

    else:
        show_error(f"Unknown command: {cmd}. Type /help for options.")

    return True


def _show_last_run(session: Session) -> None:
    """Show full output of the last pipeline run."""
    if session.last_state is None:
        console.print("  [dim]No runs yet. Type a requirement to start.[/]")
        return

    state = session.last_state
    console.print(f"\n  [bold]Last Run:[/] {state.run_id}")
    console.print(f"  Status: {state.status.value}")
    console.print(f"  Iterations: {len(state.iterations)}")

    if state.final_code:
        console.print(f"\n  [bold]Final Code:[/]")
        from rich.syntax import Syntax
        lang_map = {"C": "c", "SPARK_Ada": "ada"}
        lang = lang_map.get(session.language, "c")
        syntax = Syntax(state.final_code, lang, theme="monokai", line_numbers=True, padding=1)
        console.print(syntax)

    if state.final_proof:
        console.print(f"\n  [bold]Dafny Specification:[/]")
        syntax = Syntax(state.final_proof, "csharp", theme="monokai", padding=1)
        console.print(syntax)

    console.print()


def _show_audit(session: Session) -> None:
    """Show traceability matrix for the last run."""
    if session.last_state is None:
        console.print("  [dim]No runs yet.[/]")
        return

    from backend.services.audit.traceability import build_traceability_matrix

    state = session.last_state
    matrix = build_traceability_matrix(state)

    table = Table(title=f"Traceability Matrix — {state.run_id}", padding=(0, 1))
    table.add_column("Requirement", style="bold")
    table.add_column("Code")
    table.add_column("Tests")
    table.add_column("Proof")
    table.add_column("Policy")

    for row in matrix.rows:
        tests_str = f"{len(row.test_cases)} cases" if row.test_cases else "—"
        refs = ", ".join(row.standard_references[:3]) if row.standard_references else ""
        policy_str = row.policy_verdict
        if refs:
            policy_str += f"\n[dim]{refs}[/]"

        table.add_row(
            row.requirement[:40],
            row.code_artifact,
            tests_str,
            row.formal_proof_status,
            policy_str,
        )

    console.print()
    console.print(table)
    console.print()


def start_repl() -> None:
    """Launch the interactive REPL."""
    session = Session()

    # Detect config source
    import os
    env_config = os.environ.get("HPEMA_CONFIG", "hpema_config.yaml")
    session.config_source = env_config

    show_banner(
        config_source=env_config,
        model=session.model,
        standard=session.standard,
        language=session.language,
    )

    # Check LLM connectivity
    config = load_config()
    endpoint = config.models.actor.endpoint
    try:
        import httpx
        r = httpx.get(f"{endpoint.rstrip('/').rsplit('/v1', 1)[0]}/health", timeout=3.0)
        if r.status_code == 200:
            console.print(f"  [green]LLM connected[/] at {endpoint}\n")
        else:
            console.print(f"  [yellow]LLM responded with {r.status_code}[/] at {endpoint}\n")
    except Exception:
        console.print(f"  [red]LLM not reachable[/] at {endpoint}")
        console.print(f"  [dim]Start vLLM first, or check hpema_config[/]\n")

    # REPL loop
    while True:
        try:
            line = console.input("[bold bright_blue]hpema >[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n  [dim]Goodbye.[/]\n")
            break

        if not line:
            continue

        if line.startswith("/"):
            if not _handle_command(line, session):
                break
            continue

        # Treat as a requirement — run the pipeline
        console.print()
        console.print(f"  [dim]Standard: {session.standard}  |  Language: {session.language}  |  Max iterations: {session.max_iterations}[/]")

        display = DisplayManager()

        try:
            state = run_pipeline(
                requirement=line,
                standard=session.standard,
                language=session.language,
                max_iterations=session.max_iterations,
                display=display,
            )
            session.last_state = state
        except KeyboardInterrupt:
            console.print("\n  [yellow]Pipeline interrupted.[/]")
        except Exception as e:
            show_error(str(e))
