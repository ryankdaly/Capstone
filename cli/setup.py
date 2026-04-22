"""Interactive model/API-key setup wizard for HPEMA.

Called on first run (when no keys are detected) and via the /setup command.

Flow
----
1. Welcome screen
2. Ask: what provider are you using?
   - OpenAI, Groq, NVIDIA / NIM, ARC vLLM, or Custom
3. If API key is needed, prompt key for checker model, then tester/actor model, then policy model
   (each can be individualised or shared from a single key)
4. Choose specific model names (optional; defaults provided)
5. Write config to .env and generate / update hpema_config.local.yaml
6. Offer to restart immediately so the changes take effect

All prompts use prompt_toolkit when available for a rich interactive
experience (arrow-key selection, masked input), falling back to plain
``input()`` in non-tty environments.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

console = Console()

# ---------------------------------------------------------------------------
# prompt_toolkit availability
# ---------------------------------------------------------------------------
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.styles import Style
    from prompt_toolkit.validation import ValidationError, Validator
    _PT = True
except ImportError:
    _PT = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE     = PROJECT_ROOT / ".env"
LOCAL_CONFIG = PROJECT_ROOT / "hpema_config.local.yaml"

# ---------------------------------------------------------------------------
# Provider presets
# ---------------------------------------------------------------------------

PROVIDERS: list[dict[str, Any]] = [
    {
        "id":          "openai",
        "label":       "OpenAI  (gpt-4o, gpt-4o-mini, …)",
        "endpoint":    "https://api.openai.com/v1",
        "key_env":     "OPENAI_API_KEY",
        "key_prefix":  "sk-",
        "default_model": "gpt-4o-mini",
        "needs_key":   True,
    },
    {
        "id":          "groq",
        "label":       "Groq  (llama-3.3-70b, mixtral-8x7b, …)",
        "endpoint":    "https://api.groq.com/openai/v1",
        "key_env":     "GROQ_API_KEY",
        "key_prefix":  "gsk_",
        "default_model": "llama-3.3-70b-versatile",
        "needs_key":   True,
    },
    {
        "id":          "nvidia",
        "label":       "NVIDIA NIM / NGC  (llama-3.1-70b-instruct, …)",
        "endpoint":    "https://integrate.api.nvidia.com/v1",
        "key_env":     "NVIDIA_API_KEY",
        "key_prefix":  "nvapi-",
        "default_model": "meta/llama-3.1-70b-instruct",
        "needs_key":   True,
    },
    {
        "id":          "arc",
        "label":       "ARC vLLM  (on-cluster, no key needed)",
        "endpoint":    "https://llm-api.arc.vt.edu/api/v1",
        "key_env":     "HPEMA_API_KEY",
        "key_prefix":  "",
        "default_model": "gpt-oss-120b",
        "needs_key":   False,
    },
    {
        "id":          "local",
        "label":       "Local vLLM  (http://localhost:8000/v1)",
        "endpoint":    "http://localhost:8000/v1",
        "key_env":     "HPEMA_API_KEY",
        "key_prefix":  "",
        "default_model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "needs_key":   False,
    },
    {
        "id":          "custom",
        "label":       "Custom / other OpenAI-compatible endpoint",
        "endpoint":    "",
        "key_env":     "HPEMA_API_KEY",
        "key_prefix":  "",
        "default_model": "",
        "needs_key":   True,
    },
]

AGENT_LABELS = [
    ("actor",   "Actor  (code generator)"),
    ("checker", "Checker  (safety reviewer)"),
    ("policy",  "Policy  (compliance auditor)"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hr(title: str = "") -> None:
    console.print(Rule(title, style="bright_blue"))


def _print_step(n: int, total: int, title: str) -> None:
    console.print()
    console.print(
        f"  [bold bright_blue]Step {n}/{total}[/]  [bold]{title}[/]"
    )
    console.print()


def _masked(s: str) -> str:
    """Show first 4 chars + *** for display."""
    if not s:
        return "[dim](none)[/]"
    visible = s[:4]
    return f"[dim]{visible}[/][dim]{'*' * min(len(s) - 4, 20)}[/]"


def _plain_input(prompt: str, secret: bool = False) -> str:
    """Fallback plain input. Use getpass for secrets."""
    if secret:
        import getpass
        return getpass.getpass(f"  {prompt}: ").strip()
    try:
        return input(f"  {prompt}: ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""


# ---------------------------------------------------------------------------
# prompt_toolkit interactive pickers
# ---------------------------------------------------------------------------

def _pick_option(options: list[str], title: str = "Select an option") -> int:
    """Arrow-key selection menu. Returns 0-based index of chosen item.

    Falls back to numbered prompt when prompt_toolkit is unavailable or
    when stdin is not a tty.
    """
    if not _PT or not sys.stdin.isatty():
        console.print(f"\n  [bold]{title}[/]")
        for i, opt in enumerate(options, 1):
            console.print(f"  [cyan]{i}[/]  {opt}")
        while True:
            raw = _plain_input(f"Enter number (1–{len(options)})")
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(options):
                    return idx
            except ValueError:
                pass
            console.print("  [red]Invalid — enter a number from the list.[/]")

    # ---------- prompt_toolkit path ----------
    choice = {"idx": 0}
    kb = KeyBindings()

    @kb.add("up")
    def _up(event):
        choice["idx"] = (choice["idx"] - 1) % len(options)
        event.app.invalidate()

    @kb.add("down")
    def _down(event):
        choice["idx"] = (choice["idx"] + 1) % len(options)
        event.app.invalidate()

    @kb.add("enter")
    @kb.add("c-m")
    def _select(event):
        event.app.exit(result=choice["idx"])

    @kb.add("c-c")
    def _cancel(event):
        event.app.exit(result=None)

    _style = Style.from_dict({
        "bottom-toolbar": "bg:#313244 #6c7086 noreverse",
        "prompt":         "bold #89b4fa",
    })

    def _toolbar():
        return HTML(
            '<style bg="#313244"> '
            '<style fg="#6c7086">↑↓ navigate · Enter select · Ctrl+C cancel</style>'
            ' </style>'
        )

    def _prompt_text():
        lines = [f"\n  [bold bright_blue]{title}[/]\n"]
        for i, opt in enumerate(options):
            if i == choice["idx"]:
                lines.append(f"  [bold #a6e3a1]▶  {opt}[/]")
            else:
                lines.append(f"  [dim]   {opt}[/]")
        return "\n".join(lines) + "\n\n"

    # Use a session in repaint loop
    from prompt_toolkit import Application
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import Window
    from prompt_toolkit.layout.controls import FormattedTextControl

    def _get_text():
        lines = []
        lines.append(("bold #89b4fa", f"\n  {title}\n\n"))
        for i, opt in enumerate(options):
            if i == choice["idx"]:
                lines.append(("bold #a6e3a1", f"  ▶  {opt}\n"))
            else:
                lines.append(("class:dim #6c7086", f"     {opt}\n"))
        lines.append(("", "\n"))
        return lines

    window = Window(content=FormattedTextControl(_get_text, focusable=True))
    layout = Layout(window)
    app = Application(
        layout=layout,
        key_bindings=kb,
        style=_style,
        full_screen=False,
        mouse_support=False,
    )
    result = app.run()
    if result is None:
        raise KeyboardInterrupt
    return result


def _prompt_secret(prompt: str, env_var: str = "") -> str:
    """Prompt for a secret/API key with masking."""
    if not _PT or not sys.stdin.isatty():
        return _plain_input(prompt, secret=True)

    from prompt_toolkit import PromptSession as _PS
    from prompt_toolkit.formatted_text import HTML as _HTML

    _style = Style.from_dict({
        "prompt": "bold #89b4fa",
    })

    hint = f" ({env_var})" if env_var else ""
    session = _PS(style=_style)
    try:
        val = session.prompt(
            _HTML(f'<b><style fg="#89b4fa">  {prompt}{hint}: </style></b>'),
            is_password=True,
        )
        return val.strip() if val else ""
    except (EOFError, KeyboardInterrupt):
        return ""


def _prompt_text_input(prompt: str, default: str = "") -> str:
    """Prompt for a text value with an optional default."""
    if not _PT or not sys.stdin.isatty():
        raw = _plain_input(f"{prompt} [{default}]" if default else prompt)
        return raw if raw else default

    from prompt_toolkit import PromptSession as _PS
    from prompt_toolkit.formatted_text import HTML as _HTML

    _style = Style.from_dict({"prompt": "bold #89b4fa"})
    session = _PS(style=_style)
    default_hint = f" [dim](default: {default})[/dim]" if default else ""
    try:
        val = session.prompt(
            _HTML(
                f'<b><style fg="#89b4fa">  {prompt}</style></b>'
                + (f'<style fg="#6c7086"> (default: {default})</style>' if default else "")
                + ': '
            ),
            default=default,
        )
        return val.strip() if val else default
    except (EOFError, KeyboardInterrupt):
        return default


def _confirm(prompt: str, default: bool = True) -> bool:
    """Yes/No confirmation with arrow-key selection."""
    options = ["Yes", "No"]
    choice = {"val": 0 if default else 1}

    if not _PT or not sys.stdin.isatty():
        raw = _plain_input(f"{prompt} [{'Y/n' if default else 'y/N'}]").lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        return default

    kb = KeyBindings()

    @kb.add("left")
    @kb.add("right")
    @kb.add("tab")
    def _swap(event):
        choice["val"] = 1 - choice["val"]
        event.app.invalidate()

    @kb.add("enter")
    @kb.add("c-m")
    def _confirm_cb(event):
        event.app.exit(result=choice["val"])

    @kb.add("c-c")
    def _cancel(event):
        event.app.exit(result=None)

    _style = Style.from_dict({"prompt": "bold #89b4fa"})

    def _toolbar():
        yes_fg = "#a6e3a1" if choice["val"] == 0 else "#6c7086"
        no_fg  = "#f38ba8"  if choice["val"] == 1 else "#6c7086"
        return HTML(
            f'<style bg="#313244"> {prompt}  '
            f'<style fg="{yes_fg}">[ Yes ]</style>'
            f'  <style fg="{no_fg}">[ No ]</style>'
            f'  <style fg="#6c7086">← → to select · Enter to confirm</style>'
            f' </style>'
        )

    session = PromptSession(
        key_bindings=kb,
        style=_style,
    )
    try:
        result = session.prompt(
            HTML(f'<b><style fg="#89b4fa">  {prompt}</style></b> '),
            bottom_toolbar=_toolbar,
            default="",
        )
        return choice["val"] == 0
    except (EOFError, KeyboardInterrupt):
        return default


# ---------------------------------------------------------------------------
# Welcome screen (models not configured)
# ---------------------------------------------------------------------------

def show_welcome_screen() -> None:
    """Rich welcome panel shown when no API keys are detected on startup."""
    console.print()
    console.print(Panel(
        Text.from_markup(
            "\n"
            "  [bold #89b4fa]Welcome to HPEMA[/]\n"
            "  [dim]Hierarchical Policy-Enforced Multi-Agent Code Generation[/]\n\n"
            "  No model API keys were detected.\n"
            "  HPEMA needs access to at least one LLM to function.\n\n"
            "  [bold]To get started:[/]\n"
            "    [cyan]1.[/] Run [bold cyan]/setup[/] to configure your API keys interactively\n"
            "    [cyan]2.[/] Or set the [bold]OPENAI_API_KEY[/] / [bold]HPEMA_API_KEY[/] environment\n"
            "       variables and restart\n\n"
            "  [dim]Type [bold]/setup[/] now, or [bold]/help[/] for all commands.[/]"
            "\n"
        ),
        title="[bold]⚡ HPEMA Setup Required[/]",
        border_style="yellow",
        padding=(1, 3),
    ))
    console.print()


# ---------------------------------------------------------------------------
# Key detection
# ---------------------------------------------------------------------------

def _detect_active_keys() -> dict[str, str]:
    """Return mapping of env-var-name → masked value for keys that are set."""
    candidates = [
        "OPENAI_API_KEY",
        "GROQ_API_KEY",
        "NVIDIA_API_KEY",
        "HPEMA_API_KEY",
    ]
    found: dict[str, str] = {}
    for var in candidates:
        val = os.environ.get(var, "")
        if val and val not in ("unused", ""):
            found[var] = val
    return found


def models_are_configured() -> bool:
    """Return True when at least one real API key or a local endpoint is active."""
    keys = _detect_active_keys()
    if keys:
        return True
    # Check if config points to a local endpoint (no key needed)
    try:
        from backend.config import load_config
        cfg = load_config()
        ep = cfg.models.actor.endpoint
        if any(h in ep for h in ("localhost", "127.0.0.1", "0.0.0.0")):
            return True
        # ARC endpoint doesn't need external keys either
        if "arc.vt.edu" in ep:
            return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# .env writer
# ---------------------------------------------------------------------------

def _write_env(updates: dict[str, str]) -> None:
    """Merge *updates* into PROJECT_ROOT/.env, preserving existing lines."""
    existing: dict[str, str] = {}
    lines: list[str] = []

    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("export "):
                stripped = stripped[7:]
            m = re.match(r'^([A-Z_][A-Z0-9_]*)=(.*)', stripped)
            if m:
                existing[m.group(1)] = line  # keep original line for preserving comments
            lines.append(line)
    else:
        lines = [
            "# HPEMA environment — auto-generated by /setup",
            "# DO NOT commit this file.",
            "",
        ]

    for key, val in updates.items():
        export_line = f"export {key}={val}"
        if key in existing:
            # Replace in-place
            lines = [
                export_line if (
                    l.strip().replace("export ", "").startswith(f"{key}=")
                ) else l
                for l in lines
            ]
        else:
            lines.append(export_line)

    ENV_FILE.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# hpema_config.local.yaml writer
# ---------------------------------------------------------------------------

_CONFIG_TEMPLATE = """\
# HPEMA local configuration — auto-generated by /setup
# Edit as needed. Use: HPEMA_CONFIG=hpema_config.local.yaml python -m cli.main

models:
  actor:
    endpoint: "{actor_endpoint}"
    model: "{actor_model}"
    api_key_env: "{actor_key_env}"
  checker:
    endpoint: "{checker_endpoint}"
    model: "{checker_model}"
    api_key_env: "{checker_key_env}"
  policy:
    endpoint: "{policy_endpoint}"
    model: "{policy_model}"
    api_key_env: "{policy_key_env}"
  dafny_architect:
    endpoint: "{actor_endpoint}"
    model: "{actor_model}"
    api_key_env: "{actor_key_env}"

policies:
  standards_dir: "data/standards"
  default_standard: "DO_178C"
  embedding_model: "all-MiniLM-L6-v2"

verification:
  prover: "dafny"
  timeout_seconds: 120
  binary_path: "/opt/homebrew/bin/dafny"
  solver_path: "/opt/homebrew/bin/z3"

pipeline:
  max_iterations: 3
  require_human_approval: false
  audit_log_dir: "logs/audit"
  stage: "policy"
"""


def _write_local_config(cfg: dict[str, dict]) -> None:
    """Write hpema_config.local.yaml from the wizard results."""
    content = _CONFIG_TEMPLATE.format(
        actor_endpoint   = cfg["actor"]["endpoint"],
        actor_model      = cfg["actor"]["model"],
        actor_key_env    = cfg["actor"]["key_env"],
        checker_endpoint = cfg["checker"]["endpoint"],
        checker_model    = cfg["checker"]["model"],
        checker_key_env  = cfg["checker"]["key_env"],
        policy_endpoint  = cfg["policy"]["endpoint"],
        policy_model     = cfg["policy"]["model"],
        policy_key_env   = cfg["policy"]["key_env"],
    )
    LOCAL_CONFIG.write_text(content)


# ---------------------------------------------------------------------------
# Main wizard
# ---------------------------------------------------------------------------

def run_setup_wizard() -> bool:
    """Run the interactive setup wizard.

    Returns True if setup completed successfully, False if the user cancelled.
    """
    console.clear()

    # ── Header ──────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        Text.from_markup(
            "\n"
            "  [bold #89b4fa]HPEMA Model Setup Wizard[/]\n"
            "  [dim]Configure API keys and model endpoints for each pipeline agent[/]\n"
            "\n"
            "  [dim]You can re-run this at any time with [bold]/setup[/]\n"
            "  Changes are written to [bold].env[/] and [bold]hpema_config.local.yaml[/][/]\n"
        ),
        border_style="bright_blue",
        padding=(1, 3),
    ))
    console.print()

    total_steps = 5

    # ── Step 1: Provider ────────────────────────────────────────────────────
    _print_step(1, total_steps, "Choose your LLM provider")

    provider_options = [p["label"] for p in PROVIDERS]
    try:
        provider_idx = _pick_option(provider_options, "LLM Provider")
    except KeyboardInterrupt:
        console.print("\n  [yellow]Setup cancelled.[/]\n")
        return False

    provider = PROVIDERS[provider_idx]
    console.print(f"\n  [green]✓[/] Selected: [bold]{provider['label']}[/]\n")

    # ── Step 2: Custom endpoint (if chosen) ─────────────────────────────────
    if provider["id"] == "custom":
        _print_step(2, total_steps, "Custom endpoint configuration")
        endpoint = _prompt_text_input("API endpoint URL", default="http://localhost:8000/v1")
        key_env  = _prompt_text_input("Environment variable name for API key", default="HPEMA_API_KEY")
        provider = {**provider, "endpoint": endpoint, "key_env": key_env}
    else:
        _print_step(2, total_steps, "Endpoint confirmation")
        console.print(f"  Endpoint: [bold]{provider['endpoint']}[/]")

    # ── Step 3: API key(s) ──────────────────────────────────────────────────
    env_updates: dict[str, str] = {}
    shared_key  = ""

    if provider["needs_key"]:
        _print_step(3, total_steps, f"API Key  ({provider['key_env']})")

        # Check if already set
        existing_val = os.environ.get(provider["key_env"], "")
        if existing_val and existing_val not in ("unused", ""):
            console.print(
                f"  [green]✓[/] [bold]{provider['key_env']}[/] is already set: "
                f"{_masked(existing_val)}"
            )
            keep = _confirm("Keep existing key?", default=True)
            if keep:
                shared_key = existing_val
            else:
                shared_key = _prompt_secret(f"New {provider['key_env']}", env_var=provider["key_env"])
                if shared_key:
                    env_updates[provider["key_env"]] = shared_key
        else:
            console.print(
                f"  Enter your [bold]{provider['label']}[/] API key.\n"
                f"  [dim]It will be saved to [bold].env[/] as "
                f"[bold]{provider['key_env']}[/][/]\n"
            )
            shared_key = _prompt_secret(provider["key_env"])
            if not shared_key:
                console.print("  [yellow]No key entered — skipping.[/]")
            else:
                env_updates[provider["key_env"]] = shared_key

        # Per-agent keys?
        console.print()
        use_same = _confirm(
            "Use the same key for all three agents (Actor, Checker, Policy)?",
            default=True,
        )

        agent_keys: dict[str, str] = {}
        if not use_same:
            _print_step(3, total_steps, "Per-agent API keys")
            for agent_id, agent_label in AGENT_LABELS:
                console.print(f"\n  [bold]{agent_label}[/]")
                var_name = f"HPEMA_{agent_id.upper()}_API_KEY"
                val = _prompt_secret(f"{var_name}", env_var=var_name)
                if val:
                    env_updates[var_name] = val
                    agent_keys[agent_id] = var_name
                else:
                    agent_keys[agent_id] = provider["key_env"]  # fall back to shared
        else:
            for agent_id, _ in AGENT_LABELS:
                agent_keys[agent_id] = provider["key_env"]
    else:
        # No key needed (local / ARC)
        _print_step(3, total_steps, "API Key")
        console.print(f"  [dim]No API key required for [bold]{provider['label']}[/].[/]")
        env_updates[provider["key_env"]] = "unused"
        agent_keys = {agent_id: provider["key_env"] for agent_id, _ in AGENT_LABELS}

    # ── Step 4: Model names ─────────────────────────────────────────────────
    _print_step(4, total_steps, "Model names for each agent")

    console.print(
        "  You can use the same model for all agents, or use different models.\n"
        f"  [dim]Default: [bold]{provider['default_model']}[/][/]\n"
    )

    agent_models: dict[str, str] = {}
    use_same_model = _confirm(
        f"Use [{provider['default_model']}] for all agents?",
        default=True,
    )

    if use_same_model:
        shared_model = _prompt_text_input(
            "Model name", default=provider["default_model"]
        )
        for agent_id, _ in AGENT_LABELS:
            agent_models[agent_id] = shared_model
    else:
        for agent_id, agent_label in AGENT_LABELS:
            console.print(f"\n  [bold]{agent_label}[/]")
            m = _prompt_text_input("Model name", default=provider["default_model"])
            agent_models[agent_id] = m

    # ── Step 5: Save ────────────────────────────────────────────────────────
    _print_step(5, total_steps, "Review & Save")

    # Build a summary table
    table = Table(box=None, show_header=True, padding=(0, 2))
    table.add_column("Agent",    style="bold", width=12)
    table.add_column("Model",    width=36)
    table.add_column("Key env",  width=28, style="dim")
    table.add_column("Endpoint", width=44, style="dim")

    cfg_out: dict[str, dict] = {}
    for agent_id, agent_label in AGENT_LABELS:
        short_label = agent_label.split("(")[0].strip()
        cfg_out[agent_id] = {
            "endpoint": provider["endpoint"],
            "model":    agent_models[agent_id],
            "key_env":  agent_keys.get(agent_id, provider["key_env"]),
        }
        table.add_row(
            short_label,
            agent_models[agent_id],
            agent_keys.get(agent_id, provider["key_env"]),
            provider["endpoint"],
        )

    console.print(table)
    console.print()

    confirmed = _confirm("Save this configuration?", default=True)
    if not confirmed:
        console.print("\n  [yellow]Setup cancelled — no changes made.[/]\n")
        return False

    # Write .env
    if env_updates:
        _write_env(env_updates)
        for var, val in env_updates.items():
            os.environ[var] = val  # apply to current process immediately
        console.print(f"\n  [green]✓[/] API keys written to [bold]{ENV_FILE.name}[/]")

    # Write hpema_config.local.yaml
    _write_local_config(cfg_out)
    console.print(f"  [green]✓[/] Config written to [bold]{LOCAL_CONFIG.name}[/]")

    # Set HPEMA_CONFIG in current process
    os.environ["HPEMA_CONFIG"] = "hpema_config.local.yaml"
    console.print("  [green]✓[/] HPEMA_CONFIG set to [bold]hpema_config.local.yaml[/]")

    # ── Offer restart ────────────────────────────────────────────────────────
    console.print()
    restart = _confirm(
        "Restart HPEMA now to apply the new configuration?",
        default=True,
    )
    if restart:
        console.print("\n  [bold]Restarting HPEMA...[/]\n")
        env_file = str(ENV_FILE)
        python_bin = sys.executable  # use the same Python interpreter (venv-aware)
        if ENV_FILE.exists():
            restart_cmd = (
                f"source {env_file} && "
                f"HPEMA_CONFIG=hpema_config.local.yaml {python_bin} -m cli.main"
            )
        else:
            restart_cmd = f"HPEMA_CONFIG=hpema_config.local.yaml {python_bin} -m cli.main"
        os.execv("/bin/bash", ["/bin/bash", "-c", restart_cmd])
    else:
        console.print(
            "\n  [dim]Config saved. Restart HPEMA manually for changes to fully take effect.\n"
            "  New keys are already active in this session.[/]\n"
        )
        return True

    return True  # unreachable after execv, but keeps type-checker happy
