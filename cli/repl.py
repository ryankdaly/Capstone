"""Interactive REPL — Claude-Code-style interface for HPEMA.

Launch with: python -m cli.main
Type requirements naturally, use /commands to configure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from backend.api.schemas.pipeline import PipelineStage, PipelineState
from backend.config import load_config

# ---------------------------------------------------------------------------
# prompt_toolkit — optional, degrades gracefully to plain input if missing
# ---------------------------------------------------------------------------

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.styles import Style
    _PT_AVAILABLE = True
except ImportError:
    _PT_AVAILABLE = False


# One-line description shown next to each command in the completion menu
_CMD_META: dict[str, str] = {
    "/standard":   "set safety standard",
    "/language":   "set target language",
    "/stage":      "set pipeline stage",
    "/mode":       "switch Build/Chat mode (or Shift+Tab)",
    "/run-tests":  "toggle pytest execution",
    "/iterations": "set max iterations",
    "/last":       "show last run details",
    "/checker":    "verbose checker report [N]",
    "/dafny":      "verbose Dafny output [N]",
    "/pytest":     "full pytest output [N]",
    "/history":    "show run history [N]",
    "/audit":      "traceability matrix",
    "/config":     "show config or set config file path",
    "/setup":      "configure API keys and model endpoints",
    "/help":       "show all commands",
    "/scroll":     "terminal scrollmode",
    "/quit":       "exit HPEMA",
    "/exit":       "exit HPEMA",
}

_CMD_ARGS: dict[str, list[str]] = {
    "/standard":  ["DO_178C", "MISRA_C", "NASA", "Boeing_SDP"],
    "/language":  ["Python", "C", "SPARK_Ada"],
    "/stage":     ["actor", "checker", "policy"],
    "/mode":      ["build", "chat"],
    "/run-tests": ["on", "off"],
}


if _PT_AVAILABLE:
    class _HpemaCompleter(Completer):
        """Tab/ghost-text completer for HPEMA slash commands."""

        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            if not text.startswith("/"):
                return  # free-text requirement — no completions

            parts = text.split(maxsplit=1)

            if len(parts) == 1:
                # Complete the command name itself
                word = parts[0]
                for cmd in sorted(_CMD_META):
                    if cmd.startswith(word) and cmd != word:
                        yield Completion(
                            cmd[len(word):],
                            display=cmd,
                            display_meta=_CMD_META.get(cmd, ""),
                        )
            elif len(parts) == 2:
                # Complete the argument
                cmd, arg_prefix = parts[0].lower(), parts[1]
                for opt in _CMD_ARGS.get(cmd, []):
                    if opt.lower().startswith(arg_prefix.lower()):
                        yield Completion(opt[len(arg_prefix):], display=opt)


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
    show_logo,
    show_pytest_detail,
    show_startup_info,
    wait_for_layers_ready,
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
        self.language: str = "Python"
        self.max_iterations: int = config.pipeline.max_iterations
        self.model: str = config.models.actor.model
        self.config_source: str = ""
        self.started_at: float = time.monotonic()

        # Pipeline stage — controls how far the pipeline runs
        try:
            self.stage: PipelineStage = PipelineStage(config.pipeline.stage)
        except ValueError:
            self.stage = PipelineStage.POLICY

        # Test execution flag
        self.run_tests: bool = True

        # Run history (most recent last)
        self.history: list[RunRecord] = []

        # Input mode — "build" runs the full pipeline; "chat" is a direct LLM conversation
        self.mode: str = "build"

        # True while a pipeline or chat call is in-flight (drives toolbar suggestions)
        self.agents_running: bool = False

        # Conversation history for Chat mode
        self.chat_history: list[dict] = []

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
# Config hot-swap
# ---------------------------------------------------------------------------

def _set_config(path: str, session: Session) -> None:
    """Set HPEMA_CONFIG to *path* and offer an immediate restart."""
    import os
    import sys

    path = path.strip()
    if not os.path.isfile(path):
        show_error(f"Config file not found: {path}")
        return

    os.environ["HPEMA_CONFIG"] = path
    session.config_source = path
    console.print(f"  Config path set to: [bold]{path}[/]")
    console.print("  [dim]Takes effect on next startup. Restart now to apply immediately.[/]")

    if not _PT_AVAILABLE:
        answer = console.input("  Restart now? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            return
    else:
        from prompt_toolkit.formatted_text import HTML
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.styles import Style

        choice = {"val": 0}  # 0 = Yes, 1 = No

        kb = KeyBindings()

        @kb.add("left")
        @kb.add("right")
        @kb.add("tab")
        def _swap(event):
            choice["val"] = 1 - choice["val"]
            event.app.invalidate()

        @kb.add("enter")
        @kb.add("c-m")
        def _confirm(event):
            event.app.exit(result=choice["val"])

        @kb.add("c-c")
        def _cancel(event):
            choice["val"] = 1  # treat Ctrl-C as No
            event.app.exit(result=1)

        from prompt_toolkit import PromptSession as _PS
        _confirm_session = _PS(
            key_bindings=kb,
            style=Style.from_dict({"prompt": "bold #89b4fa"}),
        )

        def _toolbar_fn():
            yes_style = "bold #a6e3a1" if choice["val"] == 0 else ""
            no_style  = "bold #f38ba8" if choice["val"] == 1 else ""
            return HTML(
                f'<style bg="#313244"> '
                f'Restart HPEMA now?  '
                f'<style fg="{"#a6e3a1" if choice["val"] == 0 else "#6c7086"}">[ Yes ]</style>'
                f'  '
                f'<style fg="{"#f38ba8" if choice["val"] == 1 else "#6c7086"}">[ No ]</style>'
                f'  <style fg="#6c7086">← → to select · Enter to confirm</style>'
                f' </style>'
            )

        try:
            result = _confirm_session.prompt(
                HTML('<b><style fg="#89b4fa">  Restart?</style></b> '),
                bottom_toolbar=_toolbar_fn,
                default="",
            )
            # result is "" (user typed Enter) — actual choice in choice["val"]
        except (EOFError, KeyboardInterrupt):
            choice["val"] = 1

        if choice["val"] != 0:
            console.print("  [dim]Restart skipped — new config will apply on next manual start.[/]")
            return

    # --- Restart ---
    console.print("  [bold]Restarting HPEMA...[/]")
    # Find .env relative to the project root (same dir as this file's package)
    project_root = str(Path(__file__).resolve().parent.parent)
    env_file = os.path.join(project_root, ".env")

    if os.path.isfile(env_file):
        restart_cmd = f"source {env_file} && HPEMA_CONFIG={path} {sys.executable} -m cli.main"
    else:
        restart_cmd = f"HPEMA_CONFIG={path} {sys.executable} -m cli.main"

    os.execv("/bin/bash", ["/bin/bash", "-c", restart_cmd])
    # os.execv replaces the process — code below never reached


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

    elif cmd == "/setup":
        _run_setup(session)

    elif cmd == "/help":
        show_help()

    elif cmd == "/config":
        if not arg:
            show_config(session.standard, session.language, session.max_iterations, session.model, session.stage, session.run_tests)
        else:
            _set_config(arg, session)
            return True  # may have restarted; if not, continue REPL

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
            console.print("  [dim]Options: Python, C, SPARK_Ada[/]")
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

    elif cmd == "/mode":
        if not arg:
            mode_color = "green" if session.mode == "build" else "cyan"
            console.print(f"  Current mode: [{mode_color}]{session.mode.upper()}[/]")
            console.print("  [dim]build — runs the full pipeline  |  chat — direct conversation with Actor[/]")
            console.print("  [dim]Tip: Shift+Tab toggles instantly[/]")
        elif arg.lower() in ("build", "chat"):
            session.mode = arg.lower()
            mode_color = "green" if session.mode == "build" else "cyan"
            console.print(f"  Mode: [{mode_color}]{session.mode.upper()}[/]")
        else:
            show_error(f"Unknown mode: {arg}. Options: build, chat")

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
    
    elif cmd == "/scroll":
        _scroll_mode(session)

    elif cmd == "/dafny":
        _show_dafny(session, arg)

    elif cmd == "/pytest":
        _show_pytest(session, arg)

    elif cmd == "/run-tests":
        if not arg:
            status = "[green]ON[/]" if session.run_tests else "[red]OFF[/]"
            console.print(f"  Test execution: {status}")
            console.print("  [dim]Usage: /run-tests on  or  /run-tests off[/]")
        elif arg.lower() in ("on", "true", "1", "yes"):
            session.run_tests = True
            console.print("  Test execution: [green]ON[/] — pytest will run checker tests")
        elif arg.lower() in ("off", "false", "0", "no"):
            session.run_tests = False
            console.print("  Test execution: [red]OFF[/] — tests stored but not executed")
        else:
            show_error("Usage: /run-tests on  or  /run-tests off")

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
        lang_map = {"Python": "python", "C": "c", "SPARK_Ada": "ada"}
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

        # Test execution
        if last_iter.test_result:
            tr = last_iter.test_result
            if tr.executed:
                t_color = "green" if tr.failed == 0 else "red"
                console.print(f"  [bold]Tests:[/]   [{t_color}]{tr.passed}/{tr.total} passed[/]  ({tr.execution_time_seconds:.1f}s)")
            else:
                console.print(f"  [bold]Tests:[/]   [dim]{tr.total} generated (not executed)[/]")

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
    table.add_column("Requirement", max_width=35)
    table.add_column("Status", width=6, justify="center")
    table.add_column("Code", width=6, justify="center")
    table.add_column("Tests", width=8, justify="center")
    table.add_column("Dafny", width=6, justify="center")
    table.add_column("Checker", width=8, justify="center")
    table.add_column("Policy", width=8, justify="center")
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

        # Tests (from last iteration)
        tests_str = "[dim]—[/]"
        checker_str = "[dim]—[/]"
        policy_str = "[dim]—[/]"
        if state.iterations:
            last_iter = state.iterations[-1]
            if last_iter.test_result:
                tr = last_iter.test_result
                if tr.executed:
                    t_color = "green" if tr.failed == 0 else "red"
                    tests_str = f"[{t_color}]{tr.passed}/{tr.total}[/]"
                else:
                    tests_str = f"[dim]{tr.total}t[/]"
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
        req = record.requirement[:33]
        if len(record.requirement) > 33:
            req += ".."

        table.add_row(str(idx), req, status_str, code_str, tests_str, dafny_str, checker_str, policy_str, time_str)

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


def _show_pytest(session: Session, arg: str) -> None:
    """Show verbose pytest execution output for an iteration of the last run."""
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
    show_pytest_detail(state.iterations[iter_idx], iter_idx + 1)


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
# Scroll
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Scroll — curses-based in-terminal pager, no tmux required
# ---------------------------------------------------------------------------

def _scroll_mode(session: Session) -> None:
    """Full-screen arrow-key pager over session output. No tmux required.

    Keys:
      ↑ / k          scroll up one line
      ↓ / j          scroll down one line
      PgUp / u       scroll up half a page
      PgDn / d       scroll down half a page
      Home / g       jump to top
      End  / G       jump to bottom
      q / Esc        return to REPL
    """
    import curses
    import io
    from rich.console import Console as _Console
    from rich.syntax import Syntax as _Syntax

    # ------------------------------------------------------------------
    # 1. Render session to plain-text lines
    # ------------------------------------------------------------------
    buf_io = io.StringIO()
    buf_console = _Console(
        file=buf_io,
        width=min(console.width, 200),
        highlight=False,
        markup=True,
        no_color=True,
    )

    if not session.history:
        buf_console.print("  No runs yet — type a requirement to start.\n")
    else:
        for i, record in enumerate(session.history, start=1):
            state = record.state
            sep = "─" * 80
            buf_console.print(f"\n{sep}")
            buf_console.print(
                f"  Run {i:>3}  │  {record.requirement[:72]}\n"
                f"  {state.status.value.upper()}"
                f"  ·  {record.standard}"
                f"  ·  {record.language}"
                f"  ·  {record.elapsed_seconds:.1f}s"
            )
            buf_console.print(sep)

            if state.final_code:
                lang_map = {"Python": "python", "C": "c", "SPARK_Ada": "ada"}
                lang = lang_map.get(record.language, "c")
                buf_console.print(
                    f"\n  ── Source Code ({len(state.final_code.splitlines())} lines) ──\n"
                )
                buf_console.print(
                    _Syntax(state.final_code, lang, theme="ansi_dark",
                            line_numbers=True, padding=1)
                )

            if state.final_proof:
                buf_console.print(
                    f"\n  ── Dafny Specification"
                    f" ({len(state.final_proof.splitlines())} lines) ──\n"
                )
                buf_console.print(
                    _Syntax(state.final_proof, "csharp", theme="ansi_dark",
                            line_numbers=True, padding=1)
                )

            if state.iterations:
                last_iter = state.iterations[-1]
                if last_iter.checker_report:
                    cr = last_iter.checker_report
                    buf_console.print(
                        f"\n  ── Checker: {cr.verdict.value.upper()}"
                        f"  ({len(cr.issues)} issues,"
                        f" {len(cr.test_cases)} tests) ──"
                    )
                    for issue in cr.issues:
                        buf_console.print(
                            f"     [{issue.severity.value.upper()}]"
                            f" {issue.description}"
                        )
                if last_iter.test_result and last_iter.test_result.executed:
                    tr = last_iter.test_result
                    buf_console.print(
                        f"  ── Tests: {tr.passed}/{tr.total} passed ──"
                    )
                if last_iter.verification_result:
                    vr = last_iter.verification_result
                    buf_console.print(
                        f"  ── Dafny: {'VERIFIED' if vr.verified else 'FAILED'} ──"
                    )
                if last_iter.policy_verdict:
                    pv = last_iter.policy_verdict
                    pc = "COMPLIANT" if pv.compliant else "NON-COMPLIANT"
                    buf_console.print(
                        f"  ── Policy: {pc}"
                        f"  Risk: {pv.risk_level.value.upper()} ──"
                    )
                    for v in last_iter.policy_verdict.violations:
                        buf_console.print(f"     [{v.rule_id}] {v.description}")

        buf_console.print("\n")

    raw_text = buf_io.getvalue()
    # Strip any residual ANSI escape codes so curses sees clean text
    import re
    ansi_escape = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    lines: list[str] = [
        ansi_escape.sub("", line)
        for line in raw_text.splitlines()
    ]

    # ------------------------------------------------------------------
    # 2. Hand off to curses pager
    # ------------------------------------------------------------------
    try:
        curses.wrapper(_curses_pager, lines)
    except curses.error:
        # Terminal too small or curses unavailable — fall back to less/python pager
        _fallback_pager(raw_text)


def _curses_pager(stdscr, lines: list[str]) -> None:
    """Curses inner loop — called by curses.wrapper."""
    import curses

    curses.curs_set(0)          # hide cursor
    stdscr.keypad(True)         # enable arrow / PgUp / PgDn key constants
    curses.use_default_colors() # transparent background

    top = 0   # index of the topmost visible line

    while True:
        rows, cols = stdscr.getmaxyx()
        content_rows = rows - 1   # last row reserved for the status bar

        # Clamp top to valid range
        max_top = max(0, len(lines) - content_rows)
        top = max(0, min(top, max_top))

        # Draw visible lines
        stdscr.erase()
        for screen_row, line_idx in enumerate(range(top, top + content_rows)):
            if line_idx >= len(lines):
                break
            # Truncate to terminal width to avoid curses wrap errors
            text = lines[line_idx][:cols - 1]
            try:
                stdscr.addstr(screen_row, 0, text)
            except curses.error:
                pass  # writing to the last cell of last row raises — ignore

        # Status bar
        pct = int(top / max_top * 100) if max_top else 100
        at_end = top >= max_top
        end_marker = " [END]" if at_end else ""
        status = (
            f"  HPEMA scroll  "
            f"line {top + 1}/{len(lines)}  "
            f"{pct}%{end_marker}"
            f"  ── ↑↓ / jk  PgUp/PgDn  g/G  q:quit"
        )
        status = status[:cols - 1].ljust(cols - 1)
        try:
            stdscr.attron(curses.A_REVERSE)
            stdscr.addstr(rows - 1, 0, status)
            stdscr.attroff(curses.A_REVERSE)
        except curses.error:
            pass

        stdscr.refresh()

        # Input
        key = stdscr.getch()

        if key in (ord("q"), ord("Q"), 27):          # q / Q / Esc
            break
        elif key in (curses.KEY_UP, ord("k")):
            top -= 1
        elif key in (curses.KEY_DOWN, ord("j")):
            top += 1
        elif key in (curses.KEY_PPAGE, ord("u")):     # PgUp / u
            top -= content_rows // 2
        elif key in (curses.KEY_NPAGE, ord("d")):     # PgDn / d
            top += content_rows // 2
        elif key in (curses.KEY_HOME, ord("g")):
            top = 0
        elif key in (curses.KEY_END, ord("G")):
            top = max_top


def _fallback_pager(text: str) -> None:
    """less → more → built-in line pager. Used when curses is unavailable."""
    import os, shutil, subprocess, tempfile

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", prefix="hpema_scroll_",
        delete=False, encoding="utf-8",
    )
    try:
        tmp.write(text)
        tmp.flush()
        tmp.close()

        pager_env = os.environ.get("PAGER", "").strip()
        if pager_env:
            cmd = pager_env.split() + [tmp.name]
        elif shutil.which("less"):
            cmd = ["less", "-RXF", "--", tmp.name]
        elif shutil.which("more"):
            cmd = ["more", tmp.name]
        else:
            _python_pager(text)
            return
        try:
            subprocess.run(cmd, check=False)
        except FileNotFoundError:
            _python_pager(text)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _python_pager(text: str, page_size: int = 40) -> None:
    """Last-resort line-by-line pager for environments with nothing else."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        print("\n".join(lines[i : i + page_size]))
        i += page_size
        if i < len(lines):
            try:
                ans = input("\n-- more -- (Enter to continue, q to quit) ")
            except (EOFError, KeyboardInterrupt):
                break
            if ans.strip().lower() == "q":
                break

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
# Build-mode dispatch helpers
# ---------------------------------------------------------------------------

def _do_inline_chat(line: str, session: Session) -> None:
    """Stream a chat response inline — used for CONVERSE intent and explicit chat mode."""
    import asyncio as _asyncio
    from cli.display import _DynamicSpinner
    from cli.runner import chat_stream
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel

    console.print()
    collected: list[str] = []

    class _ChatPanel:
        def __init__(self) -> None:
            self._spinner = _DynamicSpinner("actor")

        def __rich__(self):
            text = "".join(collected)
            if not text:
                return self._spinner.__rich__()
            return Panel(Markdown(text), title="[bold bright_blue]HPEMA[/]",
                         border_style="bright_blue", padding=(1, 2))

    live = Live(_ChatPanel(), refresh_per_second=12, console=console)
    live.start()
    try:
        response = _asyncio.run(chat_stream(line, on_token=collected.append))
        live.stop()
        session.chat_history.append({"role": "user", "content": line})
        session.chat_history.append({"role": "assistant", "content": response})
        console.print(Panel(Markdown(response), title="[bold bright_blue]HPEMA[/]",
                            border_style="bright_blue", padding=(1, 2)))
    except KeyboardInterrupt:
        live.stop()
        console.print("\n  [yellow]Interrupted.[/]")
    except Exception as e:
        live.stop()
        show_error(str(e))


def _do_rerun_checker(session: Session) -> None:
    """Re-run the checker + tests on the last pipeline code."""
    from cli.runner import run_checker_only

    if not session.last_state or not session.last_state.final_code:
        console.print("  [dim]No code from a previous run. Generate code first.[/]")
        return

    code = session.last_state.final_code
    console.print()
    console.print(f"  [dim]Re-running Checker on last code  |  Standard: {session.standard}  |  Language: {session.language}[/]")
    display = DisplayManager()
    session.agents_running = True
    try:
        run_checker_only(
            code=code,
            standard=session.standard,
            language=session.language,
            display=display,
            run_tests=session.run_tests,
        )
    except KeyboardInterrupt:
        display.cleanup()
        console.print("\n  [yellow]Interrupted.[/]")
    except Exception as e:
        display.cleanup()
        show_error(str(e))
    finally:
        session.agents_running = False


def _do_rerun_dafny(session: Session) -> None:
    """Re-run the Dafny architect + verifier on the last pipeline code."""
    from cli.runner import run_dafny_only

    if not session.last_state or not session.last_state.final_code:
        console.print("  [dim]No code from a previous run. Generate code first.[/]")
        return

    code = session.last_state.final_code
    console.print()
    console.print(f"  [dim]Re-running Dafny on last code  |  Language: {session.language}[/]")
    display = DisplayManager()
    session.agents_running = True
    try:
        run_dafny_only(code=code, language=session.language, display=display)
    except KeyboardInterrupt:
        display.cleanup()
        console.print("\n  [yellow]Interrupted.[/]")
    except Exception as e:
        display.cleanup()
        show_error(str(e))
    finally:
        session.agents_running = False


def _do_rerun_policy(session: Session) -> None:
    """Re-run the policy agent on the last pipeline code."""
    from cli.runner import run_policy_only

    if not session.last_state or not session.last_state.final_code:
        console.print("  [dim]No code from a previous run. Generate code first.[/]")
        return

    code = session.last_state.final_code
    console.print()
    console.print(f"  [dim]Re-running Policy on last code  |  Standard: {session.standard}[/]")
    display = DisplayManager()
    session.agents_running = True
    try:
        run_policy_only(code=code, standard=session.standard, display=display)
    except KeyboardInterrupt:
        display.cleanup()
        console.print("\n  [yellow]Interrupted.[/]")
    except Exception as e:
        display.cleanup()
        show_error(str(e))
    finally:
        session.agents_running = False


def _do_full_pipeline(line: str, session: Session, pt_session: object | None) -> None:
    """Run the full pipeline for a GENERATE intent."""
    console.print()
    console.print(f"  [dim]Standard: {session.standard}  |  Language: {session.language}  |  Max iterations: {session.max_iterations}[/]")

    session.agents_running = True
    retry_context = ""
    requirement_text = line

    try:
        while True:
            display = DisplayManager()
            run_start = time.monotonic()

            try:
                state = run_pipeline(
                    requirement=requirement_text,
                    standard=session.standard,
                    language=session.language,
                    max_iterations=session.max_iterations,
                    display=display,
                    stage=session.stage,
                    run_tests=session.run_tests,
                    retry_context=retry_context,
                )
            except KeyboardInterrupt:
                display.cleanup()
                console.print("\n  [yellow]Pipeline interrupted.[/]")
                break
            except Exception as e:
                display.cleanup()
                show_error(str(e))
                break

            elapsed = time.monotonic() - run_start

            if state is not None:
                session.history.append(RunRecord(
                    requirement=requirement_text,
                    state=state,
                    standard=session.standard,
                    language=session.language,
                    stage=session.stage.value,
                    elapsed_seconds=elapsed,
                ))

            if state is None or state.status.value != "failed":
                break

            # --- Retry prompt ---
            error_dump = display.build_error_dump()
            console.print()
            console.print(
                "  [bold yellow]All iterations failed.[/]  "
                "Feed error dump into a new run?"
            )

            if not _PT_AVAILABLE or pt_session is None:
                answer = console.input(
                    "  [bold]Retry loop?[/] [dim][Y/n][/] "
                ).strip().lower()
                do_retry = answer in ("", "y", "yes")
            else:
                try:
                    answer = pt_session.prompt(  # type: ignore[union-attr]
                        "  Retry loop? [Y/n] ",
                    ).strip().lower()
                    do_retry = answer in ("", "y", "yes")
                except (EOFError, KeyboardInterrupt):
                    do_retry = False

            if not do_retry:
                break

            retry_context = error_dump
            console.print(
                f"  [dim]Retrying with {len(error_dump)} chars of error context …[/]"
            )
            console.print()
    finally:
        session.agents_running = False


# ---------------------------------------------------------------------------
# REPL entry point
# ---------------------------------------------------------------------------

def _run_setup(session: Session | None = None) -> None:
    """Launch the interactive setup wizard (called by /setup command or on startup)."""
    from cli.setup import run_setup_wizard
    run_setup_wizard()
    # If we returned (no restart), reload config into the live session
    if session is not None:
        try:
            new_cfg = load_config()
            session.standard       = new_cfg.policies.default_standard
            session.max_iterations = new_cfg.pipeline.max_iterations
            session.model          = new_cfg.models.actor.model
            from backend.config import find_config_path as _fcp2
            _f2 = _fcp2()
            session.config_source = str(_f2) if _f2 else ""
        except Exception:
            pass
        console.print("  [dim]Session config reloaded from new settings.[/]\n")


def _startup_config_check() -> None:
    """Gate that runs before the logo/startup sequence.

    Three cases:
      A. No config found anywhere     → auto-launch /setup wizard (blocks).
      B. Config found, never verified → show a summary panel, ask to confirm
                                        or re-run /setup. Writes ~/.hpema/.config_verified
                                        on confirmation so this prompt is shown exactly once.
      C. Config found and verified    → silent pass-through; proceed normally.
    """
    from rich.panel import Panel as _Panel
    from rich.table import Table as _Table
    from rich.text import Text as _Text
    from backend.config import hpema_home, find_config_path, load_config as _load

    home     = hpema_home()
    verified = home / ".config_verified"

    # ── Determine whether ANY config is reachable (single source of truth) ──
    found_path   = find_config_path()
    config_found = found_path is not None

    # ── Case A: no config → auto-launch wizard ──────────────────────────────
    if not config_found:
        console.print()
        console.print(_Panel(
            _Text.from_markup(
                "\n"
                "  [bold yellow]No configuration found.[/]\n"
                "  [dim]Launching the setup wizard...[/]\n"
            ),
            border_style="yellow",
            padding=(0, 2),
        ))
        console.print()
        time.sleep(0.8)
        from cli.setup import run_setup_wizard
        run_setup_wizard()
        return  # wizard calls _restart(); execution stops here

    # ── Case B: config exists but not yet verified ──────────────────────────
    if not verified.exists():
        try:
            cfg = _load()
        except Exception:
            return  # broken config — let normal startup surface the error

        t = _Table(box=None, show_header=False, padding=(0, 2))
        t.add_column("k", style="dim",  width=10)
        t.add_column("v", style="bold")
        t.add_row("Config",  str(found_path))
        t.add_row("Actor",   cfg.models.actor.model)
        t.add_row("Checker", cfg.models.checker.model)
        t.add_row("Policy",  cfg.models.policy.model)
        dafny_val = cfg.verification.binary_path
        dafny_display = (
            f"[green]{dafny_val}[/]" if dafny_val not in ("dafny", "LOCAL_DAFNY_INSTALL")
            else "[yellow]dafny  (relies on PATH)[/]"
        )
        t.add_row("Dafny",   dafny_display)

        console.print()
        console.print(_Panel(
            t,
            title="[bold]Configuration found — is this correct?[/]",
            border_style="bright_blue",
            padding=(1, 2),
        ))
        console.print()

        from cli.setup import _confirm
        if _confirm("Continue with this configuration?", default=True):
            verified.touch()
        else:
            from cli.setup import run_setup_wizard
            run_setup_wizard()

    # Case C: verified → fall through silently


def start_repl() -> None:
    """Launch the interactive REPL."""
    # Rule 0: config gate — auto-wizard on first run, confirmation on first boot.
    # Runs before the logo so the screen isn't cluttered before setup completes.
    console.clear()
    _startup_config_check()

    # Rule 1: clear again (setup wizard may have left output), then begin startup
    console.clear()

    session = Session()

    # Detect config source — use the same discovery logic as the loader
    from backend.config import find_config_path as _fcp
    _found = _fcp()
    env_config = str(_found) if _found else "hpema_config.yaml"
    session.config_source = env_config

    # Rule 2: logo first, 1.5s pause, then config info
    show_logo()
    time.sleep(1.5)
    config = load_config()
    show_startup_info(
        config_source=env_config,
        model=session.model,
        standard=session.standard,
        language=session.language,
        stage=session.stage,
        models_config=config.models,
    )

    # Show large warning if running in disconnected (partial pipeline) mode
    if session.is_disconnected:
        show_disconnected_warning(session.stage)

    # Welcome screen if API keys are missing even after config check
    from cli.setup import models_are_configured, show_welcome_screen
    if not models_are_configured():
        show_welcome_screen()

    # Rule 3: animated layer connectivity check — polls until all layers ready or Ctrl+S
    wait_for_layers_ready(config, session.stage)

    # Build input function — prompt_toolkit if available, plain fallback otherwise
    if _PT_AVAILABLE:
        # --- Key bindings ---
        _bindings = KeyBindings()

        @_bindings.add("s-tab")
        def _cycle_mode(event):  # noqa: F841
            session.mode = "chat" if session.mode == "build" else "build"
            # Redraw so the user sees the mode change in the bottom bar instantly
            event.app.invalidate()

        @_bindings.add("enter")
        def _submit_input(event):  # noqa: F841
            """Enter always submits (even in multiline mode)."""
            event.current_buffer.validate_and_handle()

        @_bindings.add("escape", "enter")
        def _insert_newline(event):  # noqa: F841
            """Alt/Meta+Enter inserts a newline so the input box grows."""
            event.current_buffer.insert_text("\n")

        # Minimal style — prompt_toolkit's bottom_toolbar shows the mode.
        # We also style the 'buffer' (input area) and 'rule' for the boxed look.
        _input_style = Style.from_dict({
            "bottom-toolbar":      "bg:#313244 #a6adc8 noreverse",
            "bottom-toolbar.text": "noreverse",
            "prompt":              "bold #89b4fa",
            "buffer":              "bg:#1e1e2e #cdd6f4", # Darker background for the input field
            "rule":                "#45475a",             # Subtle color for separators
        })

        def _toolbar() -> HTML:
            """Persistent bottom bar: mode left, command hints right."""
            mode_label = "BUILD" if session.mode == "build" else "CHAT"
            mode_fg = "#a6e3a1" if session.mode == "build" else "#89dceb"
            return HTML(
                f'<style bg="#313244">'
                f'<b fg="{mode_fg}"> {mode_label} </b>'
                f'  <style fg="#6c7086">Shift+Tab: mode · /help · /last · /config</style>'
                f'</style>'
            )

        _pt_session: PromptSession = PromptSession(
            completer=_HpemaCompleter(),
            history=InMemoryHistory(),
            auto_suggest=AutoSuggestFromHistory(),
            complete_while_typing=True,
            style=_input_style,
            key_bindings=_bindings,
            multiline=True,
            bottom_toolbar=_toolbar,
        )

        def _get_input() -> str:
            from rich.rule import Rule
            console.print(Rule(style="#45475a", title="[bold #89b4fa]Requirement[/]"))
            # We use prompt_toolkit's standard prompt here, but the Rule above
            # and the bottom_toolbar below create the 'boxed' feel.
            # To make the input background fill the line, we'd need a custom layout,
            # but styling 'buffer' is a good middle ground that works with PromptSession.
            text = _pt_session.prompt(
                HTML('<b><style fg="#89b4fa">hpema</style></b> › '),
            ).strip()
            return text
    else:
        def _get_input() -> str:  # type: ignore[misc]
            return console.input("[bold bright_blue]hpema >[/] ")

    # REPL loop
    while True:
        try:
            line = _get_input().strip()
        except (EOFError, KeyboardInterrupt):
            _show_goodbye(session)
            break

        if not line:
            continue

        if line.startswith("/"):
            if not _handle_command(line, session):
                break
            continue

        # --- Chat mode (explicit): direct LLM conversation, no pipeline ---
        if session.mode == "chat":
            _do_inline_chat(line, session)
            continue

        # --- Build mode: classify intent, then dispatch ---
        from cli.intent import Intent, classify as _classify

        classified = _classify(line, has_history=bool(session.history))

        if classified.intent == Intent.CONVERSE:
            # Off-topic or ambiguous — answer inline without switching modes
            _do_inline_chat(line, session)
            continue

        if classified.intent == Intent.RERUN_CHECKER:
            _do_rerun_checker(session)
            continue

        if classified.intent == Intent.RERUN_DAFNY:
            _do_rerun_dafny(session)
            continue

        if classified.intent == Intent.RERUN_POLICY:
            _do_rerun_policy(session)
            continue

        # --- GENERATE: full pipeline ---
        _do_full_pipeline(line, session, _pt_session if _PT_AVAILABLE else None)
