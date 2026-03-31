"""Claude-Code-style display for the HPEMA pipeline.

Renders agent activity with live spinners, syntax-highlighted code,
color-coded verdicts, and a final summary panel.
"""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from backend.api.schemas.pipeline import PipelineStage, StreamEvent, StreamEventType

console = Console()

# Agent role → (display name, color)
AGENT_STYLE = {
    "actor": ("Actor", "blue"),
    "checker": ("Checker", "yellow"),
    "dafny_verifier": ("Dafny Verifier", "magenta"),
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

    def handle_event(self, event: StreamEvent) -> None:
        """Route a stream event to the appropriate display method."""
        handlers = {
            StreamEventType.AGENT_START: self._on_agent_start,
            StreamEventType.AGENT_OUTPUT: self._on_agent_output,
            StreamEventType.AGENT_ERROR: self._on_agent_error,
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
            # Checker and Dafny start together (parallel) — show combined
            console.print()
            console.print(
                f"  [{color}]>[/] [bold {color}]{name}[/] reviewing code...",
            )
            return
        if agent == "dafny_verifier":
            console.print(
                f"  [{color}]>[/] [bold {color}]{name}[/] verifying specification...",
            )
            return

        console.print()
        console.print(
            f"  [{color}]>[/] [bold {color}]{name}[/] working...",
        )

    def _on_agent_output(self, event: StreamEvent) -> None:
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
        error = event.data.get("error", "Unknown error")
        console.print()
        console.print(
            Panel(
                f"[bold red]{error}[/]",
                title="[bold red]Pipeline Error[/]",
                border_style="red",
            )
        )

    def _on_iteration_complete(self, event: StreamEvent) -> None:
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
            lang_map = {"C": "c", "SPARK_Ada": "ada", "SPARK Ada": "ada"}
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
    language: str = "C",
    stage: PipelineStage = PipelineStage.POLICY,
) -> None:
    """Show the startup banner."""
    lines = [
        "[bold]HPEMA[/] v0.1.0",
        "[dim]Hierarchical Policy-Enforced Multi-Agent[/]",
        "",
    ]
    if model:
        lines.append(f"  Model:    {model}")
    if config_source:
        lines.append(f"  Config:   {config_source}")
    lines.append(f"  Standard: {standard}")
    lines.append(f"  Language: {language}")
    lines.append(f"  Prover:   Dafny")

    # Show stage info
    stage_labels = {
        PipelineStage.ACTOR: "Actor only (disconnected)",
        PipelineStage.CHECKER: "Actor + Checker (disconnected)",
        PipelineStage.POLICY: "Full pipeline",
    }
    stage_label = stage_labels.get(stage, stage.value)
    if stage != PipelineStage.POLICY:
        lines.append(f"  Stage:    [bold yellow]{stage_label}[/]")
    else:
        lines.append(f"  Stage:    {stage_label}")

    lines.append("")
    lines.append("  [dim]Type a requirement to begin, or /help[/]")

    console.print(
        Panel(
            "\n".join(lines),
            border_style="bright_blue",
            padding=(1, 2),
        )
    )


def show_disconnected_warning(stage: PipelineStage) -> None:
    """Show a large, unmissable warning when running in disconnected mode."""
    skipped: list[str] = []
    if stage == PipelineStage.ACTOR:
        skipped = ["Checker Agent", "Dafny Verifier", "Policy Agent", "Feedback Loop"]
    elif stage == PipelineStage.CHECKER:
        skipped = ["Policy Agent", "Feedback Loop"]

    lines = [
        f"[bold yellow]DISCONNECTED MODE — stage: {stage.value.upper()}[/]",
        "",
        "The pipeline will stop after the configured stage.",
        "Downstream agents are SKIPPED — results are NOT verified.",
        "",
        "[bold]Skipped components:[/]",
    ]
    for s in skipped:
        lines.append(f"  [red]x[/] {s}")
    lines.append("")
    lines.append("[dim]Set stage: /stage policy  (or edit pipeline.stage in config YAML)[/]")

    console.print(
        Panel(
            "\n".join(lines),
            title="[bold yellow]!! WARNING !![/]",
            border_style="yellow",
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
    table.add_row("/language <name>", "Set target language (C, SPARK_Ada)")
    table.add_row("/iterations <n>", "Set max pipeline iterations")
    table.add_row("/stage <name>", "Set pipeline stage (actor, checker, policy)")
    table.add_row("/last", "Show full output of the last run")
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
    console.print()


def show_error(message: str) -> None:
    console.print(f"\n  [bold red]Error:[/] {message}\n")
