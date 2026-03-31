"""Interactive REPL — Claude-Code-style interface for HPEMA.

Launch with: python -m cli.main
Type requirements naturally, use /commands to configure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from uuid import UUID

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from backend.api.schemas.pipeline import PipelineStage, PipelineState
from backend.config import load_config
from cli.display import (
    DisplayManager,
    console,
    show_banner,
    show_checker_detail,
    show_config,
    show_dafny_detail,
    show_disconnected_warning,
    show_error,
    show_help,
)
from cli.runner import run_pipeline


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    """One pipeline run in the session history."""
    requirement: str
    state: PipelineState
    standard: str
    language: str
    stage: str
    elapsed_seconds: float


class Session:
    """Holds REPL state across commands."""

    def __init__(self) -> None:
        config = load_config()
        self.standard: str = config.policies.default_standard
        self.language: str = "C"
        self.max_iterations: int = config.pipeline.max_iterations
        self.model: str = config.models.actor.model
        self.config_source: str = ""
        self.started_at: float = time.monotonic()

        # Pipeline stage — controls how far the pipeline runs
        try:
            self.stage: PipelineStage = PipelineStage(config.pipeline.stage)
        except ValueError:
            self.stage = PipelineStage.POLICY

        # Run history (most recent last)
        self.history: list[RunRecord] = []

    @property
    def last_state(self) -> PipelineState | None:
        return self.history[-1].state if self.history else None

    @property
    def last_run_id(self) -> UUID | None:
        return self.last_state.run_id if self.last_state else None

    @property
    def is_disconnected(self) -> bool:
        return self.stage != PipelineStage.POLICY


# ---------------------------------------------------------------------------
# Command handling
# ---------------------------------------------------------------------------

def _handle_command(line: str, session: Session) -> bool:
    """Handle a /command. Returns True if the REPL should continue."""
    parts = line.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/quit", "/exit", "/q"):
        _show_goodbye(session)
        return False

    elif cmd == "/help":
        show_help()

    elif cmd == "/config":
        show_config(session.standard, session.language, session.max_iterations, session.model, session.stage)

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

    elif cmd == "/history":
        n = 15
        if arg:
            try:
                n = int(arg)
            except ValueError:
                show_error("Usage: /history [N]")
                return True
        _show_history(session, n)

    elif cmd == "/stage":
        if not arg:
            console.print(f"  Current stage: [bold]{session.stage.value}[/]")
            console.print("  [dim]Options: actor, checker, policy (full pipeline)[/]")
        else:
            try:
                session.stage = PipelineStage(arg.lower())
                console.print(f"  Stage set to: [bold]{session.stage.value}[/]")
                if session.is_disconnected:
                    show_disconnected_warning(session.stage)
            except ValueError:
                show_error(f"Unknown stage: {arg}. Options: actor, checker, policy")

    elif cmd == "/checker":
        _show_checker(session, arg)

    elif cmd == "/dafny":
        _show_dafny(session, arg)

    elif cmd == "/audit":
        _show_audit(session)

    else:
        show_error(f"Unknown command: {cmd}. Type /help for options.")

    return True


# ---------------------------------------------------------------------------
# /last — detailed view of most recent run
# ---------------------------------------------------------------------------

def _show_last_run(session: Session) -> None:
    """Show detailed output of the last pipeline run."""
    if not session.history:
        console.print("  [dim]No runs yet. Type a requirement to start.[/]")
        return

    record = session.history[-1]
    state = record.state

    # Header panel
    status_color = "green" if state.status.value in ("completed", "awaiting_approval") else "red"
    header_lines = [
        f"  [bold]Requirement:[/]  {record.requirement}",
        f"  [bold]Status:[/]       [{status_color}]{state.status.value.upper()}[/]",
        f"  [bold]Run ID:[/]       {state.run_id}",
        f"  [bold]Standard:[/]     {record.standard}",
        f"  [bold]Language:[/]      {record.language}",
        f"  [bold]Stage:[/]        {record.stage}",
        f"  [bold]Iterations:[/]   {len(state.iterations)}",
        f"  [bold]Time:[/]         {record.elapsed_seconds:.1f}s",
    ]

    console.print()
    console.print(Panel(
        "\n".join(header_lines),
        title="[bold]Last Run Details[/]",
        border_style="bright_blue",
        padding=(1, 1),
    ))

    # Code output
    if state.final_code:
        lang_map = {"C": "c", "SPARK_Ada": "ada"}
        lang = lang_map.get(record.language, "c")
        code_lines = len(state.final_code.splitlines())
        console.print(f"\n  [bold]Source Code[/] [dim]({code_lines} lines)[/]")
        syntax = Syntax(state.final_code, lang, theme="monokai", line_numbers=True, padding=1)
        console.print(syntax)

    # Dafny spec
    if state.final_proof:
        dafny_lines = len(state.final_proof.splitlines())
        console.print(f"\n  [bold]Dafny Specification[/] [dim]({dafny_lines} lines)[/]")
        syntax = Syntax(state.final_proof, "csharp", theme="monokai", line_numbers=True, padding=1)
        console.print(syntax)
    else:
        console.print(f"\n  [bold]Dafny Specification[/]  [dim]— not generated[/]")

    # Reasoning trace (from the last iteration's actor output)
    if state.iterations:
        last_iter = state.iterations[-1]
        if last_iter.code_candidate and last_iter.code_candidate.reasoning_trace:
            trace = last_iter.code_candidate.reasoning_trace
            console.print(f"\n  [bold]Actor Reasoning:[/]")
            console.print(f"  [dim]{trace[:500]}[/]")

        # Checker verdict
        if last_iter.checker_report:
            cr = last_iter.checker_report
            v_label = cr.verdict.value.upper()
            v_color = "green" if cr.verdict.value == "pass" else "red"
            console.print(f"\n  [bold]Checker:[/] [{v_color}]{v_label}[/] — {len(cr.issues)} issue(s), {len(cr.test_cases)} test case(s)")

        # Dafny verification
        if last_iter.verification_result:
            vr = last_iter.verification_result
            v_color = "green" if vr.verified else "red"
            v_label = "VERIFIED" if vr.verified else "FAILED"
            console.print(f"  [bold]Dafny:[/]   [{v_color}]{v_label}[/]  ({vr.execution_time_seconds:.1f}s)")

        # Policy verdict
        if last_iter.policy_verdict:
            pv = last_iter.policy_verdict
            v_color = "green" if pv.compliant else "red"
            v_label = "COMPLIANT" if pv.compliant else "NON-COMPLIANT"
            console.print(f"  [bold]Policy:[/]  [{v_color}]{v_label}[/] — Risk: {pv.risk_level.value.upper()}, {len(pv.violations)} violation(s)")

    console.print()


# ---------------------------------------------------------------------------
# /history N — show recent run summaries
# ---------------------------------------------------------------------------

def _show_history(session: Session, n: int) -> None:
    """Show the last N runs as a summary table."""
    if not session.history:
        console.print("  [dim]No runs yet.[/]")
        return

    runs = session.history[-n:]

    table = Table(
        title=f"Run History (last {len(runs)} of {len(session.history)})",
        padding=(0, 1),
        show_lines=True,
    )
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("Requirement", max_width=40)
    table.add_column("Status", width=10, justify="center")
    table.add_column("Code", width=8, justify="center")
    table.add_column("Dafny", width=8, justify="center")
    table.add_column("Checker", width=10, justify="center")
    table.add_column("Policy", width=12, justify="center")
    table.add_column("Time", width=6, justify="right")

    offset = len(session.history) - len(runs)
    for i, record in enumerate(runs, start=1):
        state = record.state
        idx = offset + i

        # Status
        passed = state.status.value in ("completed", "awaiting_approval")
        status_str = f"[green]OK[/]" if passed else f"[red]FAIL[/]"

        # Code lines
        code_str = f"{len(state.final_code.splitlines())}L" if state.final_code else "[dim]—[/]"

        # Dafny
        dafny_str = "[green]Yes[/]" if state.final_proof else "[dim]No[/]"

        # Checker (from last iteration)
        checker_str = "[dim]—[/]"
        policy_str = "[dim]—[/]"
        if state.iterations:
            last_iter = state.iterations[-1]
            if last_iter.checker_report:
                cv = last_iter.checker_report.verdict.value
                c_color = "green" if cv == "pass" else "red"
                checker_str = f"[{c_color}]{cv.upper()}[/]"
            if last_iter.policy_verdict:
                pc = "PASS" if last_iter.policy_verdict.compliant else "FAIL"
                p_color = "green" if last_iter.policy_verdict.compliant else "red"
                policy_str = f"[{p_color}]{pc}[/]"

        # Time
        time_str = f"{record.elapsed_seconds:.1f}s"

        # Truncate requirement
        req = record.requirement[:38]
        if len(record.requirement) > 38:
            req += ".."

        table.add_row(str(idx), req, status_str, code_str, dafny_str, checker_str, policy_str, time_str)

    console.print()
    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# Goodbye — session statistics
# ---------------------------------------------------------------------------

def _show_goodbye(session: Session) -> None:
    """Show session statistics on exit."""
    session_time = time.monotonic() - session.started_at
    total_runs = len(session.history)

    if total_runs == 0:
        console.print()
        console.print(Panel(
            "[dim]No runs this session. See you next time.[/]",
            title="[bold]Goodbye[/]",
            border_style="bright_blue",
            padding=(1, 2),
        ))
        console.print()
        return

    # Compute stats
    succeeded = sum(
        1 for r in session.history
        if r.state.status.value in ("completed", "awaiting_approval")
    )
    failed = total_runs - succeeded

    total_code_lines = 0
    runs_with_dafny = 0
    runs_with_tests = 0
    runs_with_checker = 0
    runs_with_policy = 0
    total_iterations = 0
    total_pipeline_time = 0.0

    for record in session.history:
        state = record.state
        total_pipeline_time += record.elapsed_seconds
        total_iterations += len(state.iterations)

        if state.final_code:
            total_code_lines += len(state.final_code.splitlines())
        if state.final_proof:
            runs_with_dafny += 1

        if state.iterations:
            last_iter = state.iterations[-1]
            if last_iter.checker_report:
                runs_with_checker += 1
                if last_iter.checker_report.test_cases:
                    runs_with_tests += 1
            if last_iter.policy_verdict:
                runs_with_policy += 1

    # Percentages (avoid div by zero)
    pct_dafny = (runs_with_dafny / total_runs * 100) if total_runs else 0
    pct_tests = (runs_with_tests / total_runs * 100) if total_runs else 0
    pct_checker = (runs_with_checker / total_runs * 100) if total_runs else 0
    pct_policy = (runs_with_policy / total_runs * 100) if total_runs else 0
    pct_success = (succeeded / total_runs * 100) if total_runs else 0

    # Format session time
    mins, secs = divmod(int(session_time), 60)
    if mins > 0:
        session_str = f"{mins}m {secs}s"
    else:
        session_str = f"{secs}s"

    # Build stats table
    stats = Table(show_header=False, box=None, padding=(0, 2))
    stats.add_column("Label", style="bold")
    stats.add_column("Value")
    stats.add_column("Bar", width=20)

    stats.add_row("Session Duration", session_str, "")
    stats.add_row("Pipeline Runs", f"{total_runs}", "")
    stats.add_row("  Succeeded", f"[green]{succeeded}[/]", _bar(pct_success, "green"))
    stats.add_row("  Failed", f"[red]{failed}[/]" if failed else "[dim]0[/]", _bar(100 - pct_success, "red") if failed else "")
    stats.add_row("Total Iterations", f"{total_iterations}", "")
    stats.add_row("Code Generated", f"{total_code_lines} lines", "")
    stats.add_row("Pipeline Time", f"{total_pipeline_time:.1f}s", "")
    stats.add_row("", "", "")
    stats.add_row("Coverage", "", "")
    stats.add_row("  Dafny Specs", f"{runs_with_dafny}/{total_runs}", _bar(pct_dafny, "magenta"))
    stats.add_row("  Test Cases", f"{runs_with_tests}/{total_runs}", _bar(pct_tests, "yellow"))
    stats.add_row("  Checker Review", f"{runs_with_checker}/{total_runs}", _bar(pct_checker, "yellow"))
    stats.add_row("  Policy Audit", f"{runs_with_policy}/{total_runs}", _bar(pct_policy, "green"))

    console.print()
    console.print(Panel(
        stats,
        title="[bold]Session Summary[/]",
        border_style="bright_blue",
        padding=(1, 2),
    ))
    console.print()


def _bar(pct: float, color: str, width: int = 15) -> str:
    """Render a simple percentage bar."""
    filled = int(pct / 100 * width)
    empty = width - filled
    return f"[{color}]{'█' * filled}[/{color}][dim]{'░' * empty}[/dim] {pct:.0f}%"


# ---------------------------------------------------------------------------
# /checker [N] and /dafny [N] — verbose agent inspection
# ---------------------------------------------------------------------------

def _show_checker(session: Session, arg: str) -> None:
    """Show verbose checker report for an iteration of the last run."""
    if not session.history:
        console.print("  [dim]No runs yet.[/]")
        return

    state = session.history[-1].state
    if not state.iterations:
        console.print("  [dim]No iterations in last run.[/]")
        return

    iter_idx = _parse_iteration_arg(arg, len(state.iterations))
    if iter_idx is None:
        return
    show_checker_detail(state.iterations[iter_idx], iter_idx + 1)


def _show_dafny(session: Session, arg: str) -> None:
    """Show verbose Dafny result for an iteration of the last run."""
    if not session.history:
        console.print("  [dim]No runs yet.[/]")
        return

    state = session.history[-1].state
    if not state.iterations:
        console.print("  [dim]No iterations in last run.[/]")
        return

    iter_idx = _parse_iteration_arg(arg, len(state.iterations))
    if iter_idx is None:
        return
    show_dafny_detail(state.iterations[iter_idx], iter_idx + 1)


def _parse_iteration_arg(arg: str, total: int) -> int | None:
    """Parse an optional iteration number argument. Returns 0-based index or None on error."""
    if not arg:
        return total - 1  # default: last iteration

    try:
        n = int(arg)
    except ValueError:
        show_error(f"Expected iteration number, got: {arg}")
        return None

    if n < 1 or n > total:
        show_error(f"Iteration {n} not found. Last run had {total} iteration(s).")
        return None

    return n - 1


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# REPL entry point
# ---------------------------------------------------------------------------

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
        stage=session.stage,
    )

    # Show large warning if running in disconnected (partial pipeline) mode
    if session.is_disconnected:
        show_disconnected_warning(session.stage)

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
            _show_goodbye(session)
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
        run_start = time.monotonic()

        try:
            state = run_pipeline(
                requirement=line,
                standard=session.standard,
                language=session.language,
                max_iterations=session.max_iterations,
                display=display,
                stage=session.stage,
            )
            elapsed = time.monotonic() - run_start
            if state is not None:
                session.history.append(RunRecord(
                    requirement=line,
                    state=state,
                    standard=session.standard,
                    language=session.language,
                    stage=session.stage.value,
                    elapsed_seconds=elapsed,
                ))
        except KeyboardInterrupt:
            console.print("\n  [yellow]Pipeline interrupted.[/]")
        except Exception as e:
            show_error(str(e))
