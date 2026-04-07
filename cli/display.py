"""Claude-Code-style display for the HPEMA pipeline.

Renders agent activity with live spinners, syntax-highlighted code,
color-coded verdicts, and a final summary panel.
"""

from __future__ import annotations

import time
from typing import Any

from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from backend.api.schemas.pipeline import PipelineStage, StreamEvent, StreamEventType

console = Console()

# ---------------------------------------------------------------------------
# HPEMA logo — ANSI Shadow block-letter style, blue top-to-bottom gradient
# ---------------------------------------------------------------------------

_LOGO_LINES = [
    "██╗  ██╗██████╗ ███████╗███╗   ███╗ █████╗ ",
    "██║  ██║██╔══██╗██╔════╝████╗ ████║██╔══██╗",
    "███████║██████╔╝█████╗  ██╔████╔██║███████║",
    "██╔══██║██╔═══╝ ██╔══╝  ██║╚██╔╝██║██╔══██║",
    "██║  ██║██║     ███████╗██║ ╚═╝ ██║██║  ██║",
    "╚═╝  ╚═╝╚═╝     ╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝",
]
# True-color gradient: light sky blue → royal blue → navy shadow
_LOGO_COLORS = [
    "#87CEFF",
    "#6AAFD6",
    "#4D92BE",
    "#3175A6",
    "#1C5A8E",
    "#112E55",
]

# Agent role → (display name, color)
AGENT_STYLE = {
    "actor": ("Actor", "blue"),
    "checker": ("Checker", "yellow"),
    "dafny_verifier": ("Dafny Verifier", "magenta"),
    "test_runner": ("Test Runner", "cyan"),
    "policy": ("Policy", "green"),
}

VERDICT_STYLE = {
    "pass": ("PASS", "bold green"),
    "fail": ("FAIL", "bold red"),
    "warn": ("WARN", "bold yellow"),
}


class DisplayManager:
    """Handles all terminal output for a pipeline run."""

    def __init__(self) -> None:
        self._agent_start_times: dict[str, float] = {}
        self._iteration_start: float = 0.0
        self._pipeline_start: float = time.monotonic()
        self._current_iteration: int = 0
        # Store full agent data for /last command
        self.last_actor_output: dict[str, Any] = {}
        self.last_checker_output: dict[str, Any] = {}
        self.last_dafny_output: dict[str, Any] = {}
        self.last_policy_output: dict[str, Any] = {}
        # Live spinner — active between AGENT_START and the first output event
        self._live: Live | None = None

    # ------------------------------------------------------------------
    # Spinner helpers
    # ------------------------------------------------------------------

    def _start_spinner(self, label: str, color: str) -> None:
        """Start an animated spinner line. Replaces any existing spinner."""
        self._stop_spinner()
        spinner = Spinner("dots", text=Text.from_markup(f"  [bold {color}]{label}[/]"), style=color)
        self._live = Live(spinner, refresh_per_second=12, transient=True, console=console)
        self._live.start()

    def _stop_spinner(self) -> None:
        """Stop and erase the current spinner. No-op if none is running."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    def cleanup(self) -> None:
        """Force-stop any running spinner. Call from exception handlers."""
        self._stop_spinner()

    def handle_event(self, event: StreamEvent) -> None:
        """Route a stream event to the appropriate display method."""
        handlers = {
            StreamEventType.AGENT_START: self._on_agent_start,
            StreamEventType.AGENT_OUTPUT: self._on_agent_output,
            StreamEventType.AGENT_ERROR: self._on_agent_error,
            StreamEventType.TEST_RUN: self._on_test_run,
            StreamEventType.ITERATION_COMPLETE: self._on_iteration_complete,
            StreamEventType.PIPELINE_COMPLETE: self._on_pipeline_complete,
        }
        handler = handlers.get(event.event_type)
        if handler:
            handler(event)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_agent_start(self, event: StreamEvent) -> None:
        agent = event.agent or "unknown"
        name, color = AGENT_STYLE.get(agent, (agent, "white"))
        self._agent_start_times[agent] = time.monotonic()

        if agent == "checker":
            # Checker and Dafny run in parallel — one spinner covers both
            console.print()
            self._start_spinner("Checker  +  Dafny Verifier  working...", "yellow")
            return
        if agent == "dafny_verifier":
            # Spinner already running from checker start — nothing to do
            return
        if agent == "test_runner":
            self._start_spinner("Test Runner  executing pytest...", "cyan")
            return

        console.print()
        self._start_spinner(f"{name}  working...", color)

    def _on_agent_output(self, event: StreamEvent) -> None:
        self._stop_spinner()
        agent = event.agent or "unknown"
        data = event.data
        name, color = AGENT_STYLE.get(agent, (agent, "white"))
        elapsed = self._elapsed(agent)

        if agent == "actor":
            self._render_actor(name, color, data, elapsed)
        elif agent == "checker":
            self._render_checker(name, color, data, elapsed)
        elif agent == "dafny_verifier":
            self._render_dafny(name, color, data, elapsed)
        elif agent == "policy":
            self._render_policy(name, color, data, elapsed)

    def _on_agent_error(self, event: StreamEvent) -> None:
        self._stop_spinner()
        error = event.data.get("error", "Unknown error")
        console.print()
        console.print(
            Panel(
                f"[bold red]{error}[/]",
                title="[bold red]Pipeline Error[/]",
                border_style="red",
            )
        )

    def _on_test_run(self, event: StreamEvent) -> None:
        self._stop_spinner()
        data = event.data
        elapsed = self._elapsed("test_runner")
        executed = data.get("executed", False)
        total = data.get("total", 0)
        passed = data.get("passed", 0)
        failed = data.get("failed", 0)
        errors = data.get("errors", 0)
        test_results = data.get("test_results", [])

        if not executed:
            console.print(
                Panel(
                    f"[dim]{total} test(s) generated but not executed[/]",
                    title="[bold cyan]Test Runner[/]",
                    border_style="dim",
                    padding=(0, 1),
                )
            )
            return

        all_pass = failed == 0 and errors == 0 and total > 0
        border = "green" if all_pass else "red" if (failed > 0 or errors > 0) else "yellow"

        parts: list[str] = []
        if total == 0 and errors == 0:
            parts.append(f"[bold yellow]NO TESTS COLLECTED[/] — check pytest output with /checker")
            border = "yellow"
        elif all_pass:
            parts.append(f"[bold green]ALL PASSED[/] — {passed}/{total} tests")
        elif errors > 0:
            parts.append(f"[bold red]ERROR[/] — pytest collection or execution failed")
        else:
            parts.append(f"[bold red]{failed} FAILED[/] — {passed} passed, {total} total")

        # Show individual test results
        if test_results:
            for tr in test_results:
                name = tr.get("name", "?")
                if tr.get("passed"):
                    parts.append(f"  [green]✓[/] {name}")
                else:
                    err = tr.get("error_message", "")
                    parts.append(f"  [red]✗[/] {name}")
                    if err:
                        parts.append(f"    [dim]{err[:100]}[/]")

        console.print(
            Panel(
                "\n".join(parts),
                title=f"[bold cyan]Test Runner[/] [dim]{elapsed}[/]",
                border_style=border,
                padding=(0, 1),
            )
        )

    def _on_iteration_complete(self, event: StreamEvent) -> None:
        self._stop_spinner()
        data = event.data
        iteration = data.get("iteration", "?")
        all_pass = data.get("all_pass", False)
        summary = data.get("summary", "")
        max_iter = self._current_iteration  # will be set properly

        console.print()
        if all_pass:
            console.print(
                f"  [bold green]--- Iteration {iteration}: ALL CHECKS PASSED ---[/]"
            )
        else:
            console.print(
                f"  [bold red]--- Iteration {iteration}: FAILED ---[/]"
            )
            if summary:
                # Show truncated feedback summary
                lines = summary.split(" | ")
                for line in lines[:3]:
                    console.print(f"  [dim]{line.strip()}[/]")
            console.print(f"  [dim]Composing feedback for next iteration...[/]")

    def _on_pipeline_complete(self, event: StreamEvent) -> None:
        self._stop_spinner()
        data = event.data
        status = data.get("status", "unknown")
        iterations = data.get("iterations", 0)
        run_id = data.get("run_id", "")
        stage = data.get("stage", "policy")
        total_time = time.monotonic() - self._pipeline_start

        console.print()

        stage_note = ""
        if stage != "policy":
            stage_note = f"\n  [yellow]Stage:       {stage} (disconnected)[/]"

        if status in ("completed", "awaiting_approval"):
            console.print(
                Panel(
                    f"[bold green]Pipeline SUCCEEDED[/]\n"
                    f"  Iterations:  {iterations}\n"
                    f"  Total time:  {total_time:.1f}s\n"
                    f"  Run ID:      {run_id}\n"
                    f"  Audit log:   logs/audit/{run_id}.jsonl"
                    f"{stage_note}\n\n"
                    f"  [dim]Type /last to see full output, /audit for traceability[/]",
                    title="[bold green]Pipeline Complete[/]",
                    border_style="green",
                    padding=(1, 2),
                )
            )
        else:
            console.print(
                Panel(
                    f"[bold red]Pipeline {status.upper()}[/]\n"
                    f"  Iterations:  {iterations}\n"
                    f"  Total time:  {total_time:.1f}s\n"
                    f"  Run ID:      {run_id}\n"
                    f"  Audit log:   logs/audit/{run_id}.jsonl"
                    f"{stage_note}",
                    title="[bold red]Pipeline Complete[/]",
                    border_style="red",
                    padding=(1, 2),
                )
            )

    # ------------------------------------------------------------------
    # Agent-specific renderers
    # ------------------------------------------------------------------

    def _render_actor(self, name: str, color: str, data: dict, elapsed: str) -> None:
        self.last_actor_output = data
        language = data.get("language", "C")
        has_dafny = data.get("has_dafny", False)
        source_code = data.get("source_code", "")
        dafny_spec = data.get("dafny_spec", "")
        reasoning = data.get("reasoning_trace", "")

        # Count lines
        code_lines = len(source_code.splitlines()) if source_code else 0

        parts: list[str] = []
        dafny_label = " + Dafny spec" if has_dafny else ""
        parts.append(f"[bold]Generated {code_lines} lines of {language}{dafny_label}[/]")

        content = "\n".join(parts)

        console.print(
            Panel(
                content,
                title=f"[bold {color}]{name}[/] [dim]{elapsed}[/]",
                border_style=color,
                padding=(0, 1),
            )
        )

        # Show syntax-highlighted code preview (truncated to 20 lines)
        if source_code:
            preview = "\n".join(source_code.splitlines()[:20])
            if code_lines > 20:
                preview += f"\n... ({code_lines - 20} more lines)"
            lang_map = {"Python": "python", "C": "c", "SPARK_Ada": "ada", "SPARK Ada": "ada"}
            syntax = Syntax(
                preview,
                lang_map.get(language, "c"),
                theme="monokai",
                line_numbers=True,
                padding=1,
            )
            console.print(syntax)

    def _render_checker(self, name: str, color: str, data: dict, elapsed: str) -> None:
        self.last_checker_output = data
        verdict_val = data.get("verdict", "unknown")
        issues_count = data.get("issues", 0)
        issues_list = data.get("issues_detail", [])
        test_cases = data.get("test_cases", [])

        verdict_label, verdict_style = VERDICT_STYLE.get(
            verdict_val, (verdict_val.upper(), "bold white")
        )

        parts: list[str] = []
        parts.append(f"[{verdict_style}]{verdict_label}[/] — {issues_count} issue(s)")

        if issues_list:
            for issue in issues_list[:5]:
                sev = issue.get("severity", "info").upper()
                desc = issue.get("description", "")
                sev_color = "red" if sev == "CRITICAL" else "yellow" if sev == "MAJOR" else "dim"
                parts.append(f"  [{sev_color}][{sev}][/] {desc}")

        if test_cases:
            parts.append(f"\n  Test cases generated: {len(test_cases)}")

        console.print(
            Panel(
                "\n".join(parts),
                title=f"[bold {color}]{name}[/] [dim]{elapsed}[/]",
                border_style="green" if verdict_val == "pass" else "red",
                padding=(0, 1),
            )
        )

    def _render_dafny(self, name: str, color: str, data: dict, elapsed: str) -> None:
        self.last_dafny_output = data
        verified = data.get("verified", False)
        failing = data.get("failing_assertions", [])

        if verified:
            content = "[bold green]VERIFIED[/] — all assertions hold"
        else:
            parts = ["[bold red]FAILED[/]"]
            if failing:
                for fa in failing[:3]:
                    parts.append(f"  [red]{fa}[/]")
            content = "\n".join(parts)

        console.print(
            Panel(
                content,
                title=f"[bold {color}]{name}[/] [dim]{elapsed}[/]",
                border_style="green" if verified else "red",
                padding=(0, 1),
            )
        )

    def _render_policy(self, name: str, color: str, data: dict, elapsed: str) -> None:
        self.last_policy_output = data
        compliant = data.get("compliant", False)
        risk_level = data.get("risk_level", "unknown")
        violations = data.get("violations", [])
        recommendations = data.get("recommendations", [])

        if compliant:
            parts = [f"[bold green]COMPLIANT[/] — Risk: [green]{risk_level.upper()}[/]"]
        else:
            parts = [f"[bold red]NON-COMPLIANT[/] — Risk: [red]{risk_level.upper()}[/]"]

        if violations:
            parts.append("")
            for v in violations[:5]:
                rule = v.get("rule_id", "")
                desc = v.get("description", "")
                parts.append(f"  [red][{rule}][/] {desc}")

        if recommendations:
            parts.append("")
            parts.append("[dim]Recommendations:[/]")
            for rec in recommendations[:3]:
                parts.append(f"  [dim]- {rec}[/]")

        console.print(
            Panel(
                "\n".join(parts),
                title=f"[bold {color}]{name}[/] [dim]{elapsed}[/]",
                border_style="green" if compliant else "red",
                padding=(0, 1),
            )
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _elapsed(self, agent: str) -> str:
        start = self._agent_start_times.get(agent)
        if start is None:
            return ""
        return f"{time.monotonic() - start:.1f}s"


def show_banner(
    config_source: str = "",
    model: str = "",
    standard: str = "DO_178C",
    language: str = "Python",
    stage: PipelineStage = PipelineStage.POLICY,
) -> None:
    """Show the startup banner with gradient HPEMA logo."""
    # --- Gradient logo ---
    logo = Text(justify="center")
    for i, (line, color) in enumerate(zip(_LOGO_LINES, _LOGO_COLORS)):
        logo.append(line, style=f"bold {color}")
        if i < len(_LOGO_LINES) - 1:
            logo.append("\n")

    subtitle = Text(
        "Hierarchical Policy-Enforced Multi-Agent  ·  v0.1.0",
        style="dim",
        justify="center",
    )

    # --- Info block ---
    stage_labels = {
        PipelineStage.ACTOR:   "Actor only",
        PipelineStage.CHECKER: "Actor + Checker",
        PipelineStage.POLICY:  "Full pipeline",
    }
    stage_label = stage_labels.get(stage, stage.value)
    stage_style = "yellow" if stage != PipelineStage.POLICY else "green"

    info = Text(justify="left")
    info.append("\n")
    if model:
        info.append("  Model    ", style="dim")
        info.append(f"{model}\n")
    if config_source:
        info.append("  Config   ", style="dim")
        info.append(f"{config_source}\n")
    info.append("  Standard ", style="dim")
    info.append(f"{standard}  ", style="bold")
    info.append("  Language ", style="dim")
    info.append(f"{language}  ", style="bold")
    info.append("  Prover ", style="dim")
    info.append("Dafny\n", style="bold")
    info.append("  Stage    ", style="dim")
    info.append(f"{stage_label}\n", style=f"bold {stage_style}")
    info.append("\n  ")
    info.append("Type a requirement to begin", style="dim")
    info.append("  or  ", style="dim")
    info.append("/help", style="bold cyan")

    console.print()
    console.print(Panel(
        Group(
            Align.center(logo),
            Align.center(subtitle),
            info,
        ),
        border_style="bright_blue",
        padding=(1, 3),
    ))
    console.print()


def show_disconnected_warning(stage: PipelineStage) -> None:
    """Show a warning when running in partial pipeline mode."""
    if stage == PipelineStage.ACTOR:
        skipped = ["Checker Agent", "Dafny Verifier", "Policy Agent", "Feedback Loop"]
        lines = [
            f"[bold yellow]DISCONNECTED MODE — stage: {stage.value.upper()}[/]",
            "",
            "Single inference only. No verification, no feedback loop.",
            "",
            "[bold]Skipped components:[/]",
        ]
        for s in skipped:
            lines.append(f"  [red]x[/] {s}")
        lines.append("")
        lines.append("[dim]Set stage: /stage checker  or  /stage policy[/]")

        console.print(
            Panel(
                "\n".join(lines),
                title="[bold yellow]!! WARNING !![/]",
                border_style="yellow",
                padding=(1, 2),
            )
        )
    elif stage == PipelineStage.CHECKER:
        lines = [
            f"[bold cyan]PARTIAL PIPELINE — stage: {stage.value.upper()}[/]",
            "",
            "[green]Active:[/]  Actor + Checker Agent + Dafny Verifier + Feedback Loop",
            "[yellow]Skipped:[/] Policy Agent (RAG compliance audit)",
            "",
            "Dafny failures and Checker issues will feed back to the Actor.",
            "[dim]Set stage: /stage policy  for full pipeline[/]",
        ]

        console.print(
            Panel(
                "\n".join(lines),
                title="[bold cyan]Partial Pipeline[/]",
                border_style="cyan",
                padding=(1, 2),
            )
        )


def show_help() -> None:
    """Show REPL help."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Command", style="bold cyan")
    table.add_column("Description")
    table.add_row("<requirement>", "Type any requirement to start the pipeline")
    table.add_row("/standard <name>", "Set safety standard (DO_178C, MISRA_C, NASA, Boeing_SDP)")
    table.add_row("/language <name>", "Set target language (Python, C, SPARK_Ada)")
    table.add_row("/iterations <n>", "Set max pipeline iterations")
    table.add_row("/stage <name>", "Set pipeline stage (actor, checker, policy)")
    table.add_row("/run-tests <on|off>", "Toggle pytest execution of checker tests (default: on)")
    table.add_row("/last", "Detailed view of the last run (code, spec, verdicts)")
    table.add_row("/checker [N]", "Verbose checker report for iteration N (default: last)")
    table.add_row("/dafny [N]", "Verbose Dafny verification for iteration N (default: last)")
    table.add_row("/pytest [N]", "Full pytest output for iteration N (default: last)")
    table.add_row("/history [N]", "Show last N runs as a summary table (default: 15)")
    table.add_row("/audit", "Show traceability matrix for the last run")
    table.add_row("/config", "Show current configuration")
    table.add_row("/help", "Show this help")
    table.add_row("/quit", "Exit")
    console.print()
    console.print(table)
    console.print()


def show_config(
    standard: str, language: str, max_iterations: int, model: str = "",
    stage: PipelineStage = PipelineStage.POLICY,
    run_tests: bool = True,
) -> None:
    """Show current config."""
    console.print()
    console.print(f"  [bold]Standard:[/]    {standard}")
    console.print(f"  [bold]Language:[/]     {language}")
    console.print(f"  [bold]Iterations:[/]  {max_iterations}")
    if model:
        console.print(f"  [bold]Model:[/]       {model}")
    stage_color = "yellow" if stage != PipelineStage.POLICY else "green"
    console.print(f"  [bold]Stage:[/]       [{stage_color}]{stage.value}[/]")
    tests_color = "green" if run_tests else "red"
    tests_label = "ON" if run_tests else "OFF"
    console.print(f"  [bold]Run Tests:[/]   [{tests_color}]{tests_label}[/]")
    console.print()


def show_checker_detail(iteration: "IterationRecord", iter_num: int) -> None:
    """Show verbose checker output for an iteration."""
    from backend.api.schemas.pipeline import IterationRecord  # noqa: F811

    cr = iteration.checker_report
    if cr is None:
        console.print(f"\n  [dim]No checker report for iteration {iter_num}.[/]\n")
        return

    verdict_label, verdict_style = VERDICT_STYLE.get(
        cr.verdict.value, (cr.verdict.value.upper(), "bold white")
    )

    parts: list[str] = [
        f"  [bold]Verdict:[/]  [{verdict_style}]{verdict_label}[/]",
        f"  [bold]Issues:[/]   {len(cr.issues)}",
        f"  [bold]Tests:[/]    {len(cr.test_cases)}",
    ]

    if cr.issues:
        parts.append("")
        parts.append("  [bold]Issues:[/]")
        for i, issue in enumerate(cr.issues, 1):
            sev = issue.severity.value.upper()
            sev_color = "red" if sev == "CRITICAL" else "yellow" if sev == "MAJOR" else "dim"
            line_ref = f" (line {issue.line_reference})" if issue.line_reference else ""
            parts.append(f"    [{sev_color}]{i}. [{sev}]{line_ref}[/] {issue.description}")
            if issue.suggested_fix:
                parts.append(f"       [dim]Fix: {issue.suggested_fix}[/]")

    if cr.test_cases:
        parts.append("")
        parts.append("  [bold]Test Cases:[/]")
        for j, tc in enumerate(cr.test_cases, 1):
            # Truncate long test cases for display
            tc_display = tc[:120] + "..." if len(tc) > 120 else tc
            parts.append(f"    [cyan]{j}.[/] {tc_display}")

    if cr.reasoning_trace:
        parts.append("")
        parts.append("  [bold]Reasoning:[/]")
        parts.append(f"  [dim]{cr.reasoning_trace[:500]}[/]")

    console.print()
    console.print(Panel(
        "\n".join(parts),
        title=f"[bold yellow]Checker Report — Iteration {iter_num}[/]",
        border_style="yellow",
        padding=(1, 1),
    ))
    console.print()


def show_dafny_detail(iteration: "IterationRecord", iter_num: int) -> None:
    """Show verbose Dafny output for an iteration."""
    from backend.api.schemas.pipeline import IterationRecord  # noqa: F811
    from rich.syntax import Syntax as _Syntax

    vr = iteration.verification_result
    if vr is None:
        console.print(f"\n  [dim]No Dafny result for iteration {iter_num}.[/]\n")
        return

    v_color = "green" if vr.verified else "red"
    v_label = "VERIFIED" if vr.verified else "FAILED"

    parts: list[str] = [
        f"  [bold]Status:[/]    [{v_color}][bold]{v_label}[/][/]",
        f"  [bold]Prover:[/]    {vr.prover}",
        f"  [bold]Time:[/]      {vr.execution_time_seconds:.1f}s",
    ]

    if vr.failing_assertions:
        parts.append("")
        parts.append("  [bold]Failing Assertions:[/]")
        for fa in vr.failing_assertions:
            parts.append(f"    [red]{fa}[/]")

    if vr.solver_output:
        parts.append("")
        parts.append("  [bold]Solver Output:[/]")

    console.print()
    console.print(Panel(
        "\n".join(parts),
        title=f"[bold magenta]Dafny Verification — Iteration {iter_num}[/]",
        border_style="magenta",
        padding=(1, 1),
    ))

    # Show solver output separately (can be long)
    if vr.solver_output:
        # Truncate very long output
        output = vr.solver_output
        if len(output.splitlines()) > 40:
            lines = output.splitlines()
            output = "\n".join(lines[:40]) + f"\n... ({len(lines) - 40} more lines)"
        console.print(_Syntax(output, "text", theme="monokai", padding=1))

    # Show the Dafny spec if available
    if iteration.code_candidate and iteration.code_candidate.dafny_spec:
        spec = iteration.code_candidate.dafny_spec
        console.print(f"\n  [bold]Dafny Spec ({len(spec.splitlines())} lines):[/]")
        console.print(_Syntax(spec, "csharp", theme="monokai", line_numbers=True, padding=1))
    else:
        console.print(f"\n  [dim]No Dafny spec was generated this iteration.[/]")

    console.print()


def show_pytest_detail(iteration: "IterationRecord", iter_num: int) -> None:
    """Show verbose pytest execution output for an iteration."""
    from backend.api.schemas.pipeline import IterationRecord  # noqa: F811

    tr = iteration.test_result
    if tr is None:
        console.print(f"\n  [dim]No test result for iteration {iter_num}.[/]\n")
        return

    if not tr.executed:
        parts = [
            f"  [bold]Status:[/]  [yellow]NOT EXECUTED[/]",
            f"  [bold]Reason:[/]  {tr.pytest_output or 'Tests disabled or unavailable'}",
        ]
        if iteration.checker_report and iteration.checker_report.test_cases:
            parts.append(f"\n  [bold]Stored Test Cases ({len(iteration.checker_report.test_cases)}):[/]")
            for j, tc in enumerate(iteration.checker_report.test_cases, 1):
                parts.append(f"    [cyan]{j}.[/] {tc[:120]}")
        console.print()
        console.print(Panel(
            "\n".join(parts),
            title=f"[bold cyan]Pytest — Iteration {iter_num}[/]",
            border_style="yellow",
            padding=(1, 1),
        ))
        console.print()
        return

    all_pass = tr.failed == 0 and tr.errors == 0 and tr.total > 0
    border = "green" if all_pass else "red"

    parts: list[str] = [
        f"  [bold]Total:[/]   {tr.total}",
        f"  [bold]Passed:[/]  [green]{tr.passed}[/]",
        f"  [bold]Failed:[/]  [red]{tr.failed}[/]" if tr.failed else f"  [bold]Failed:[/]  0",
        f"  [bold]Errors:[/]  [red]{tr.errors}[/]" if tr.errors else f"  [bold]Errors:[/]  0",
        f"  [bold]Time:[/]    {tr.execution_time_seconds:.2f}s",
    ]

    if tr.test_results:
        parts.append("")
        parts.append("  [bold]Results:[/]")
        for t in tr.test_results:
            if t.passed:
                parts.append(f"    [green]✓[/] {t.name}")
            else:
                parts.append(f"    [red]✗[/] {t.name}")
                if t.error_message:
                    parts.append(f"      [dim]{t.error_message}[/]")

    console.print()
    console.print(Panel(
        "\n".join(parts),
        title=f"[bold cyan]Pytest — Iteration {iter_num}[/]",
        border_style=border,
        padding=(1, 1),
    ))

    # Show raw pytest output
    if tr.pytest_output:
        from rich.syntax import Syntax as _Syntax
        output = tr.pytest_output
        if len(output.splitlines()) > 50:
            lines = output.splitlines()
            output = "\n".join(lines[:50]) + f"\n... ({len(lines) - 50} more lines)"
        console.print(f"\n  [bold]Raw pytest output:[/]")
        console.print(_Syntax(output, "text", theme="monokai", padding=1))

    console.print()


def wait_for_layers_ready(config: Any, stage: PipelineStage, max_wait: int = 300) -> None:
    """Animate health checks for every LLM layer until all are ready or Ctrl+C.

    - Local endpoints (localhost / vLLM): polled via /health until 200 OK.
    - External API endpoints (OpenAI, Groq, Nvidia …): marked ready immediately
      — no polling needed, keys are validated at inference time.
    - Policy layer when stage < POLICY: shown as Disconnected, skipped.

    Press Ctrl+C at any time to skip and proceed to the REPL.
    """
    import httpx

    _FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    layer_cfgs = [
        ("Actor",   config.models.actor),
        ("Checker", config.models.checker),
        ("Policy",  config.models.policy),
    ]

    # None = pending, True = ready, False = unreachable, "disconnected" = skipped
    statuses: dict[str, Any] = {}
    for name, _ in layer_cfgs:
        if name == "Policy" and stage != PipelineStage.POLICY:
            statuses[name] = "disconnected"
        else:
            statuses[name] = None

    def _is_local(ep: str) -> bool:
        return any(h in ep for h in ("localhost", "127.0.0.1", "0.0.0.0"))

    def _check(ep: str) -> bool:
        if not _is_local(ep):
            return True  # external API — assume reachable
        try:
            base = ep.rstrip("/").rsplit("/v1", 1)[0]
            r = httpx.get(f"{base}/health", timeout=3.0)
            return r.status_code == 200
        except Exception:
            return False

    def _build_panel(frame: str, elapsed: float) -> Panel:
        table = Table(box=None, show_header=False, padding=(0, 2))
        table.add_column(width=3,  justify="center")  # icon
        table.add_column(width=9)                      # name
        table.add_column(width=40)                     # endpoint
        table.add_column(width=18)                     # state

        for name, cfg in layer_cfgs:
            ep     = cfg.endpoint
            status = statuses[name]

            if status == "disconnected":
                icon, ep_s, state = "[dim]○[/]", f"[dim]{ep}[/]", "[dim]Disconnected[/]"
            elif status is True:
                icon, ep_s, state = "[bold green]✓[/]", ep, "[bold green]Ready[/]"
            elif status is False:
                icon, ep_s, state = "[bold red]✗[/]", f"[red]{ep}[/]", "[bold red]Unreachable[/]"
            else:
                icon  = f"[bold blue]{frame}[/]"
                ep_s  = ep
                state = "[blue]Connecting...[/]"

            table.add_row(icon, f"[bold]{name}[/]", ep_s, state)

        hint = Text("\n  Press Ctrl+C to skip and proceed", style="dim")
        elapsed_s = f"{elapsed:.0f}s"

        return Panel(
            Group(table, hint),
            title=f"[bold bright_blue]Layer Status[/]  [dim]{elapsed_s}[/]",
            border_style="bright_blue",
            padding=(1, 2),
        )

    start    = time.monotonic()
    frame_i  = 0
    all_done = False

    try:
        with Live(console=console, refresh_per_second=8) as live:
            while True:
                elapsed = time.monotonic() - start
                frame   = _FRAMES[frame_i % len(_FRAMES)]
                frame_i += 1

                # Check every pending layer
                for name, cfg in layer_cfgs:
                    if statuses[name] is None:
                        statuses[name] = _check(cfg.endpoint)

                live.update(_build_panel(frame, elapsed))

                pending  = [n for n, s in statuses.items() if s is None]
                all_done = not pending

                if all_done or elapsed > max_wait:
                    break

                time.sleep(2.0)

    except KeyboardInterrupt:
        console.print("\n  [yellow]Connectivity check skipped — proceeding.[/]\n")
        return

    # Print the final resolved state (stays visible after Live exits)
    console.print(_build_panel("✓", time.monotonic() - start))
    console.print()


def show_error(message: str) -> None:
    console.print(f"\n  [bold red]Error:[/] {message}\n")
