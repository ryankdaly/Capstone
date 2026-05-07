"""Interactive model/API-key setup wizard for HPEMA.

Called on first run (when no keys are detected) and via the /setup command.

All LLM providers in HPEMA use the OpenAI-compatible REST API — the same
Python client (openai.AsyncOpenAI) talks to every endpoint, with only the
base_url and api_key differing per provider. This wizard just pre-fills those
two values so the user doesn't have to write YAML by hand.

Flow
----
1. Welcome screen
2. Pick provider (all are OpenAI-compatible endpoints)
3. Enter / confirm API key
4. Optionally customise model names per agent
5. Auto-detect Dafny; warn if missing
6. Write ~/.hpema/hpema_config.yaml  +  ~/.hpema/.env
7. Offer restart
"""

from __future__ import annotations

import os
import re
import shutil
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

# ---------------------------------------------------------------------------
# Path resolution — works both from source checkout and pip-installed package
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Import canonical helpers from backend — single source of truth for paths.
# Lazy import pattern used here because setup.py is loaded early and we don't
# want to trigger the full backend import chain at module level.
def _hpema_home() -> Path:
    from backend.config import hpema_home
    return hpema_home()


def _env_file() -> Path:
    """Always ~/.hpema/.env — the canonical writable location for API keys."""
    return _hpema_home() / ".env"


def _next_config_path(name: str = "") -> Path:
    """Return a non-conflicting config path inside ~/.hpema/.

    If name is blank: tries hpema_config.yaml, then hpema_config_2.yaml, etc.
    If name is given: hpema_config_{name}.yaml, auto-increments suffix on collision.
    Never overwrites an existing file.
    """
    home = _hpema_home()
    if name:
        safe = re.sub(r"[^\w\-]", "_", name.strip())
        candidate = home / f"hpema_config_{safe}.yaml"
        if not candidate.exists():
            return candidate
        n = 2
        while (home / f"hpema_config_{safe}_{n}.yaml").exists():
            n += 1
        return home / f"hpema_config_{safe}_{n}.yaml"
    else:
        candidate = home / "hpema_config.yaml"
        if not candidate.exists():
            return candidate
        n = 2
        while (home / f"hpema_config_{n}.yaml").exists():
            n += 1
        return home / f"hpema_config_{n}.yaml"

# ---------------------------------------------------------------------------
# Dafny / Z3 auto-detection
# ---------------------------------------------------------------------------

def _detect_dafny() -> str | None:
    """Return the Dafny binary path, or None if not found."""
    # 1. System PATH (covers homebrew, apt, dotnet global tool on PATH)
    found = shutil.which("dafny")
    if found:
        return found
    # 2. ~/.hpema/dafny_path written by `hpema setup --install-dafny`
    marker = _hpema_home() / "dafny_path"
    if marker.exists():
        p = marker.read_text().strip()
        if Path(p).exists():
            return p
    # 3. Common fixed locations (platform-aware)
    candidates: list[Path] = [
        Path("/opt/homebrew/bin/dafny"),                        # macOS Homebrew (Apple Silicon)
        Path("/usr/local/bin/dafny"),                            # macOS Homebrew (Intel)
        Path.home() / ".dotnet" / "tools" / "dafny",            # dotnet global (Unix)
        Path.home() / ".dotnet" / "tools" / "dafny.exe",        # dotnet global (Windows)
        Path("/usr/bin/dafny"),                                  # Linux distro package
        PROJECT_ROOT / "local" / "installs" / "dafny" / "dafny",     # in-repo binary (Unix)
        PROJECT_ROOT / "local" / "installs" / "dafny" / "dafny.exe", # in-repo binary (Windows)
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def _detect_z3() -> str | None:
    """Return the Z3 binary path, or None if not found."""
    found = shutil.which("z3")
    if found:
        return found
    candidates: list[Path] = [
        Path("/opt/homebrew/bin/z3"),
        Path("/usr/local/bin/z3"),
        Path("/usr/bin/z3"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


# ---------------------------------------------------------------------------
# Provider presets — ALL use the OpenAI-compatible REST API
# ---------------------------------------------------------------------------

PROVIDERS: list[dict[str, Any]] = [
    {
        "id":             "nvidia",
        "label":          "NVIDIA NIM  (mistral-large, llama-3.1-70b, …)  [recommended]",
        "endpoint":       "https://integrate.api.nvidia.com/v1",
        "key_env":        "NVIDIA_API_KEY",
        "key_prefix":     "nvapi-",
        "default_model":  "mistralai/mistral-large-3-675b-instruct-2512",
        "default_family": "mistral",
        "needs_key":      True,
    },
    {
        "id":             "openai",
        "label":          "OpenAI  (gpt-4.1-mini, gpt-4.1, …)",
        "endpoint":       "https://api.openai.com/v1",
        "key_env":        "OPENAI_API_KEY",
        "key_prefix":     "sk-",
        "default_model":  "gpt-4.1-mini",
        "default_family": "openai-gpt4o",
        "needs_key":      True,
    },
    {
        "id":             "deepseek",
        "label":          "DeepSeek  (deepseek-v4-flash, deepseek-v4-pro, deepseek-r1, …)",
        "endpoint":       "https://api.deepseek.com",
        "key_env":        "DEEPSEEK_API_KEY",
        "key_prefix":     "sk-",
        "default_model":  "deepseek-v4-flash",
        "default_family": "deepseek-v3",
        "needs_key":      True,
    },
    {
        "id":             "groq",
        "label":          "Groq  (llama-3.3-70b, mixtral-8x7b, …)",
        "endpoint":       "https://api.groq.com/openai/v1",
        "key_env":        "GROQ_API_KEY",
        "key_prefix":     "gsk_",
        "default_model":  "llama-3.3-70b-versatile",
        "default_family": "generic",
        "needs_key":      True,
    },
    {
        "id":             "arc",
        "label":          "ARC vLLM  (on-cluster, no key needed)",
        "endpoint":       "https://llm-api.arc.vt.edu/api/v1",
        "key_env":        "HPEMA_API_KEY",
        "key_prefix":     "",
        "default_model":  "gpt-oss-120b",
        "default_family": "generic",
        "needs_key":      False,
    },
    {
        "id":             "local",
        "label":          "Local vLLM  (http://localhost:8000/v1, no key needed)",
        "endpoint":       "http://localhost:8000/v1",
        "key_env":        "HPEMA_API_KEY",
        "key_prefix":     "",
        "default_model":  "meta-llama/Meta-Llama-3-8B-Instruct",
        "default_family": "generic",
        "needs_key":      False,
    },
    {
        "id":             "custom",
        "label":          "Custom / other OpenAI-compatible endpoint",
        "endpoint":       "",
        "key_env":        "HPEMA_API_KEY",
        "key_prefix":     "",
        "default_model":  "",
        "default_family": "generic",
        "needs_key":      True,
    },
]

ALL_FAMILIES: list[str] = [
    "generic", "mistral", "mistral-reasoning",
    "qwen3", "gemma3", "gemma4",
    "kimi-k2", "anthropic",
    "openai-gpt4o", "openai-reasoning",
    "stepfun", "minimax",
    "deepseek-r1", "deepseek-v3", "deepseek-v4",
]

# Agents shown in the model-name step. dafny_architect is auto-set to mirror
# actor (same provider + model) — power users can edit the YAML afterwards.
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
    console.print(f"  [bold bright_blue]Step {n}/{total}[/]  [bold]{title}[/]")
    console.print()


def _masked(s: str) -> str:
    if not s:
        return "[dim](none)[/]"
    visible = s[:4]
    return f"[dim]{visible}[/][dim]{'*' * min(len(s) - 4, 20)}[/]"


def _plain_input(prompt: str, secret: bool = False) -> str:
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
    """Arrow-key selection menu. Returns 0-based index. Falls back to numbered prompt."""
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

    def _get_text():
        lines = [("bold #89b4fa", f"\n  {title}\n\n")]
        for i, opt in enumerate(options):
            if i == choice["idx"]:
                lines.append(("bold #a6e3a1", f"  ▶  {opt}\n"))
            else:
                lines.append(("#6c7086", f"     {opt}\n"))
        lines.append(("", "\n"))
        return lines

    from prompt_toolkit import Application
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import Window
    from prompt_toolkit.layout.controls import FormattedTextControl

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
    if not _PT or not sys.stdin.isatty():
        return _plain_input(prompt, secret=True)

    from prompt_toolkit import PromptSession as _PS

    _style = Style.from_dict({"prompt": "bold #89b4fa"})
    hint = f" ({env_var})" if env_var else ""
    session = _PS(style=_style)
    try:
        val = session.prompt(
            HTML(f'<b><style fg="#89b4fa">  {prompt}{hint}: </style></b>'),
            is_password=True,
        )
        return val.strip() if val else ""
    except (EOFError, KeyboardInterrupt):
        return ""


def _prompt_text_input(prompt: str, default: str = "") -> str:
    if not _PT or not sys.stdin.isatty():
        raw = _plain_input(f"{prompt} [{default}]" if default else prompt)
        return raw if raw else default

    from prompt_toolkit import PromptSession as _PS

    _style = Style.from_dict({"prompt": "bold #89b4fa"})
    session = _PS(style=_style)
    try:
        val = session.prompt(
            HTML(
                f'<b><style fg="#89b4fa">  {prompt}</style></b>'
                + (f'<style fg="#6c7086"> (default: {default})</style>' if default else "")
                + ": "
            ),
            default=default,
        )
        return val.strip() if val else default
    except (EOFError, KeyboardInterrupt):
        return default


def _confirm(prompt: str, default: bool = True) -> bool:
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

    session = PromptSession(key_bindings=kb, style=_style)
    try:
        session.prompt(
            HTML(f'<b><style fg="#89b4fa">  {prompt}</style></b> '),
            bottom_toolbar=_toolbar,
            default="",
        )
        return choice["val"] == 0
    except (EOFError, KeyboardInterrupt):
        return default


# ---------------------------------------------------------------------------
# Welcome screen
# ---------------------------------------------------------------------------

def show_welcome_screen() -> None:
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
            "    [cyan]2.[/] Or set [bold]NVIDIA_API_KEY[/] / [bold]OPENAI_API_KEY[/] and restart\n\n"
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
    candidates = ["NVIDIA_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "GROQ_API_KEY", "HPEMA_API_KEY"]
    return {
        var: val
        for var in candidates
        if (val := os.environ.get(var, "")) and val not in ("unused", "")
    }


def models_are_configured() -> bool:
    """True when at least one real API key or a local/no-key endpoint is active."""
    if _detect_active_keys():
        return True
    try:
        from backend.config import load_config
        ep = load_config().models.actor.endpoint
        return any(h in ep for h in ("localhost", "127.0.0.1", "0.0.0.0", "arc.vt.edu"))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

def _write_env(updates: dict[str, str]) -> None:
    """Merge updates into ~/.hpema/.env, preserving existing lines."""
    env_path = _env_file()
    existing: dict[str, str] = {}
    lines: list[str] = []

    if env_path.exists():
        for line in env_path.read_text().splitlines():
            stripped = line.strip().removeprefix("export ")
            m = re.match(r'^([A-Z_][A-Z0-9_]*)=(.*)', stripped)
            if m:
                existing[m.group(1)] = line
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
            lines = [
                export_line if l.strip().removeprefix("export ").startswith(f"{key}=") else l
                for l in lines
            ]
        else:
            lines.append(export_line)

    env_path.write_text("\n".join(lines) + "\n")


def _build_config_yaml(
    cfg: dict[str, dict],
    dafny_path: str | None,
    z3_path: str | None,
    family: str = "generic",
) -> str:
    """Render the hpema_config YAML from wizard results and detected binaries."""
    # Use forward slashes in all YAML path values — YAML double-quoted strings
    # interpret backslashes as escape sequences, which breaks Windows paths.
    # Python's pathlib accepts forward slashes on Windows too.
    def _yp(p: str | Path | None) -> str:
        return Path(p).as_posix() if p else ""

    dafny_bin = _yp(dafny_path) if dafny_path else "dafny"
    solver_line = (
        f'  solver_path: "{_yp(z3_path)}"' if z3_path
        else "  solver_path: null  # set to z3 path if Dafny can't find it"
    )
    # Always use absolute ~/.hpema/ paths so the config works from any cwd.
    home = _hpema_home()
    chroma_dir = (home / "chromadb").as_posix()
    standards_dir = (home / "standards").as_posix()

    return f"""\
# HPEMA configuration — auto-generated by /setup
# Re-run:  hpema then /setup
# All endpoints use the OpenAI-compatible REST API.

models:
  actor:
    endpoint: "{cfg['actor']['endpoint']}"
    model: "{cfg['actor']['model']}"
    api_key_env: "{cfg['actor']['key_env']}"
    family: "{cfg['actor'].get('family', family)}"
  checker:
    endpoint: "{cfg['checker']['endpoint']}"
    model: "{cfg['checker']['model']}"
    api_key_env: "{cfg['checker']['key_env']}"
    family: "{cfg['checker'].get('family', family)}"
  policy:
    endpoint: "{cfg['policy']['endpoint']}"
    model: "{cfg['policy']['model']}"
    api_key_env: "{cfg['policy']['key_env']}"
    family: "{cfg['policy'].get('family', family)}"
  dafny_architect:
    # Mirrors actor by default. Edit model/endpoint here for a dedicated verifier.
    endpoint: "{cfg['actor']['endpoint']}"
    model: "{cfg['actor']['model']}"
    api_key_env: "{cfg['actor']['key_env']}"
    family: "{cfg['actor'].get('family', family)}"

policies:
  standards_dir: "{standards_dir}"
  chromadb_dir: "{chroma_dir}"
  default_standard: "DO_178C"
  embedding_model: "all-MiniLM-L6-v2"

verification:
  prover: "dafny"
  timeout_seconds: 120
  binary_path: "{dafny_bin}"
{solver_line}

pipeline:
  max_iterations: 3
  require_human_approval: false
  audit_log_dir: "logs/audit"
  stage: "policy"
"""


def _write_config(
    cfg_path: Path,
    cfg: dict[str, dict],
    dafny_path: str | None,
    z3_path: str | None,
    family: str = "generic",
) -> None:
    cfg_path.write_text(_build_config_yaml(cfg, dafny_path, z3_path, family))


# ---------------------------------------------------------------------------
# Restart helper
# ---------------------------------------------------------------------------

def _restart(config_path: str | None = None) -> None:
    """Re-exec HPEMA using the installed `hpema` binary or python -m cli.main."""
    console.print("\n  [bold]Restarting HPEMA...[/]\n")

    hpema_bin = shutil.which("hpema")
    if hpema_bin:
        os.execv(hpema_bin, [hpema_bin])
        return  # unreachable

    env_path = _env_file()
    if sys.platform == "win32":
        import subprocess
        env = os.environ.copy()
        if config_path:
            env["HPEMA_CONFIG"] = config_path
        subprocess.run([sys.executable, "-m", "cli.main"], env=env)
        sys.exit(0)
    else:
        if env_path.exists() and config_path:
            os.execv(
                "/bin/bash",
                ["/bin/bash", "-c",
                 f"source {env_path} && HPEMA_CONFIG={config_path} {sys.executable} -m cli.main"],
            )
        elif env_path.exists():
            os.execv("/bin/bash", ["/bin/bash", "-c",
                     f"source {env_path} && {sys.executable} -m cli.main"])
        else:
            os.execv(sys.executable, [sys.executable, "-m", "cli.main"])


# ---------------------------------------------------------------------------
# Helpers: prefill from existing config, key validation
# ---------------------------------------------------------------------------

def _provider_idx_from_endpoint(endpoint: str) -> int:
    """Return index into PROVIDERS matching this endpoint, or last (custom) index."""
    for i, p in enumerate(PROVIDERS):
        if p["id"] != "custom" and p.get("endpoint") == endpoint:
            return i
    return len(PROVIDERS) - 1  # custom


def _prefill_from_yaml(path: Path) -> dict:
    """Load an existing config YAML and return prefill defaults for the wizard."""
    import yaml as _yaml
    prefill: dict = {
        "provider_idx": 0,
        "endpoint": PROVIDERS[0]["endpoint"],
        "key_env": PROVIDERS[0]["key_env"],
        "models": {},
        "family": "generic",
        "needs_key": True,
    }
    try:
        raw = _yaml.safe_load(path.read_text()) or {}
        actor = raw.get("models", {}).get("actor", {})
        endpoint = actor.get("endpoint", prefill["endpoint"])
        prefill["endpoint"] = endpoint
        prefill["key_env"] = actor.get("api_key_env", prefill["key_env"])
        prefill["provider_idx"] = _provider_idx_from_endpoint(endpoint)
        prefill["family"] = actor.get("family", "generic")
        for role in ("actor", "checker", "policy"):
            m = raw.get("models", {}).get(role, {})
            prefill["models"][role] = m.get("model", "")
        prefill["needs_key"] = prefill["key_env"] not in ("", "unused")
    except Exception:
        pass
    return prefill


def _validate_and_fix_keys(cfg_out: dict[str, dict]) -> None:
    """Check all configured API key env vars are set. Prompt to enter any missing ones."""
    needed = sorted({
        v["key_env"]
        for v in cfg_out.values()
        if v.get("key_env") and v["key_env"] != "unused"
    })
    if not needed:
        return

    console.print()
    console.rule("[dim]API Key Check[/]", style="dim")
    console.print()

    env_updates: dict[str, str] = {}
    all_ok = True

    for key_env in needed:
        val = os.environ.get(key_env, "")
        if val and val != "unused":
            console.print(f"  [green]✓[/] [bold]{key_env}[/] is set  {_masked(val)}")
        else:
            all_ok = False
            console.print(
                f"\n  [bold yellow]⚠  {key_env} is not set.[/]\n"
                f"  [dim]HPEMA cannot call the model without this key.[/]"
            )
            new_val = _prompt_secret(f"Enter {key_env} now (or press Enter to skip)", env_var=key_env)
            if new_val:
                os.environ[key_env] = new_val
                env_updates[key_env] = new_val
                console.print(f"  [green]✓[/] {key_env} set for this session")
            else:
                console.print(
                    f"  [yellow]Skipped.[/] [dim]Set [bold]{key_env}[/] in your shell "
                    f"or re-run [bold]/setup[/] before using HPEMA.[/]"
                )

    if env_updates:
        _write_env(env_updates)
        console.print(f"\n  [green]✓[/] Key(s) saved to [bold]{_env_file()}[/]")

    if all_ok:
        console.print("\n  [green]✓[/] All API keys are configured.\n")
    else:
        console.print()


# ---------------------------------------------------------------------------
# Config picker with side-panel YAML preview
# ---------------------------------------------------------------------------

def _pick_config_with_preview(configs: list[Path], title: str = "Select configuration") -> int:
    """Arrow-key config list with a live YAML preview panel on the right.

    Returns 0-based index into configs. Falls back to plain _pick_option if
    prompt_toolkit is unavailable or the terminal is not a tty.
    """
    if not _PT or not sys.stdin.isatty():
        return _pick_option([p.name for p in configs], title)

    import yaml as _yaml
    from prompt_toolkit import Application
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import HSplit, VSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.layout.dimension import Dimension

    choice = {"idx": 0}
    kb = KeyBindings()

    @kb.add("up")
    def _up(event):
        choice["idx"] = (choice["idx"] - 1) % len(configs)
        event.app.invalidate()

    @kb.add("down")
    def _down(event):
        choice["idx"] = (choice["idx"] + 1) % len(configs)
        event.app.invalidate()

    @kb.add("enter")
    @kb.add("c-m")
    def _select(event):
        event.app.exit(result=choice["idx"])

    @kb.add("c-c")
    def _cancel(event):
        event.app.exit(result=None)

    _style = Style.from_dict({
        "list-title": "bold #89b4fa",
        "list-item":  "#6c7086",
        "list-sel":   "bold #a6e3a1",
        "preview-title": "bold #cba6f7",
        "preview-body":  "#cdd6f4",
    })

    def _preview_lines() -> list[tuple[str, str]]:
        path = configs[choice["idx"]]
        lines: list[tuple[str, str]] = [("class:preview-title", f" {path.name}\n")]
        lines.append(("class:preview-body", " " + "─" * 38 + "\n"))
        try:
            raw = _yaml.safe_load(path.read_text()) or {}
            models = raw.get("models", {})
            for role in ("actor", "checker", "policy", "dafny_architect"):
                m = models.get(role, {})
                if m:
                    lines.append(("class:preview-body", f" [{role}]\n"))
                    for k in ("endpoint", "model", "family", "api_key_env"):
                        v = m.get(k, "")
                        if v:
                            # Truncate long values
                            disp = v if len(v) <= 36 else v[:33] + "…"
                            lines.append(("class:preview-body", f"   {k}: {disp}\n"))
            verif = raw.get("verification", {})
            if verif:
                lines.append(("class:preview-body", " [verification]\n"))
                for k in ("prover", "binary_path", "timeout_seconds"):
                    v = verif.get(k, "")
                    if v:
                        lines.append(("class:preview-body", f"   {k}: {v}\n"))
        except Exception as exc:
            lines.append(("class:preview-body", f" (parse error: {exc})\n"))
        return lines

    def _list_lines() -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = [("class:list-title", f"\n  {title}\n\n")]
        for i, p in enumerate(configs):
            if i == choice["idx"]:
                out.append(("class:list-sel", f"  ▶  {p.name}\n"))
            else:
                out.append(("class:list-item", f"     {p.name}\n"))
        out.append(("", "\n  [↑↓ navigate  ·  Enter select  ·  Ctrl-C cancel]\n"))
        return out

    list_window    = Window(
        content=FormattedTextControl(_list_lines, focusable=True),
        width=Dimension(min=30, max=50),
    )
    preview_window = Window(
        content=FormattedTextControl(_preview_lines, focusable=False),
        width=Dimension(min=40),
    )
    divider        = Window(width=1, char="│", style="class:list-item")

    layout = Layout(HSplit([
        VSplit([list_window, divider, preview_window]),
    ]))

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


# ---------------------------------------------------------------------------
# Main wizard
# ---------------------------------------------------------------------------

def run_setup_wizard(edit_path: Path | None = None) -> bool:
    """Run the interactive setup wizard.

    edit_path: if set, runs in edit mode — loads existing config as defaults
               and saves back to the same file. If None, creates a new config.
    Returns True on success, False on cancel.
    """
    console.clear()
    home = _hpema_home()

    # ── Edit existing vs create new ────────────────────────────────────────
    if edit_path is None:
        existing_configs = sorted(home.glob("hpema_config*.yaml"))

        if existing_configs:
            console.print()
            console.print(Panel(
                Text.from_markup(
                    "\n"
                    "  [bold #89b4fa]HPEMA Setup[/]\n"
                    f"  [dim]{len(existing_configs)} existing config(s) found in [bold]{home}[/][/]\n"
                ),
                border_style="bright_blue",
                padding=(1, 3),
            ))
            console.print()
            try:
                action_idx = _pick_option(
                    [
                        "Use existing configuration (restart with it)",
                        "Edit existing configuration",
                        "Create new configuration",
                    ],
                    "What would you like to do?",
                )
            except KeyboardInterrupt:
                console.print("\n  [yellow]Setup cancelled.[/]\n")
                return False

            if action_idx == 0:
                # ── Use existing config — pick, set env, offer restart ─────
                configs = sorted(home.glob("hpema_config*.yaml"))
                try:
                    cidx = _pick_config_with_preview(configs, "Select configuration to use")
                except KeyboardInterrupt:
                    console.print("\n  [yellow]Setup cancelled.[/]\n")
                    return False
                chosen = configs[cidx]
                os.environ["HPEMA_CONFIG"] = str(chosen)
                _write_env({"HPEMA_CONFIG": str(chosen)})
                console.print(
                    f"\n  [green]✓[/] Active config set to [bold]{chosen.name}[/]\n"
                    f"  [dim]Full path: {chosen}[/]\n"
                )
                if _confirm("Restart HPEMA now to apply?", default=True):
                    _restart(str(chosen))
                else:
                    console.print(
                        "\n  [dim]Config selection saved. Restart HPEMA for changes to fully take effect.[/]\n"
                    )
                return True

            if action_idx == 1:
                configs = sorted(home.glob("hpema_config*.yaml"))
                try:
                    cidx = _pick_config_with_preview(configs, "Select configuration to edit")
                except KeyboardInterrupt:
                    console.print("\n  [yellow]Setup cancelled.[/]\n")
                    return False
                edit_path = configs[cidx]
                # Recurse with edit_path set
                return run_setup_wizard(edit_path=edit_path)

    # ── Determine save target and prefill ─────────────────────────────────
    is_edit = edit_path is not None
    prefill = _prefill_from_yaml(edit_path) if is_edit else {}

    if is_edit:
        target_config = edit_path
        console.print()
        console.print(Panel(
            Text.from_markup(
                "\n"
                f"  [bold #89b4fa]Edit Configuration[/]\n"
                f"  [dim]Editing: [bold]{edit_path.name}[/]\n"
                f"  Full path: {edit_path}[/]\n\n"
                "  [dim]Press Enter to keep current values. Changes are saved on confirm.[/]\n"
            ),
            border_style="bright_blue",
            padding=(1, 3),
        ))
        console.print()
    else:
        # ── Config name for new config ─────────────────────────────────────
        console.print()
        console.print(Panel(
            Text.from_markup(
                "\n"
                "  [bold #89b4fa]HPEMA Model Setup Wizard[/]\n"
                "  [dim]Configure API keys and model endpoints for each pipeline agent[/]\n\n"
                "  [dim]Every provider below uses the [bold]OpenAI-compatible REST API[/].\n"
                "  HPEMA uses the same Python client for all of them — only the\n"
                "  base URL and API key differ.[/]\n\n"
                f"  [dim]Config saved to [bold]{home}[/][/]\n"
            ),
            border_style="bright_blue",
            padding=(1, 3),
        ))
        console.print()

        config_name_input = _prompt_text_input(
            "Config name (leave blank for auto-generated, e.g. 'nvidia_mistral')",
            default="",
        )
        target_config = _next_config_path(config_name_input)
        console.print(f"  [dim]Will save to: [bold]{target_config}[/][/]\n")

    total_steps = 6

    # ── Step 1: Provider ────────────────────────────────────────────────────
    _print_step(1, total_steps, "Choose your LLM provider")
    console.print(
        "  [dim]All options are OpenAI-compatible endpoints. HPEMA uses the same\n"
        "  protocol for every provider — only the URL and key change.[/]\n"
    )

    # In edit mode, pre-select the provider that matches the existing endpoint
    default_provider_idx = prefill.get("provider_idx", 0)
    # Rotate list so current provider appears first
    ordered_providers = (
        [PROVIDERS[default_provider_idx]]
        + [p for i, p in enumerate(PROVIDERS) if i != default_provider_idx]
    )
    provider_options = [p["label"] for p in ordered_providers]
    try:
        provider_sel = _pick_option(provider_options, "LLM Provider")
    except KeyboardInterrupt:
        console.print("\n  [yellow]Setup cancelled.[/]\n")
        return False

    provider = ordered_providers[provider_sel]
    console.print(f"\n  [green]✓[/] Selected: [bold]{provider['label']}[/]\n")

    # ── Step 2: Endpoint ────────────────────────────────────────────────────
    if provider["id"] == "custom":
        _print_step(2, total_steps, "Custom endpoint configuration")
        endpoint = _prompt_text_input(
            "OpenAI-compatible API endpoint URL",
            default=prefill.get("endpoint", "http://localhost:8000/v1"),
        )
        key_env  = _prompt_text_input(
            "Environment variable name for API key",
            default=prefill.get("key_env", "HPEMA_API_KEY"),
        )
        provider = {**provider, "endpoint": endpoint, "key_env": key_env}
    else:
        _print_step(2, total_steps, "Endpoint confirmation")
        console.print(f"  Endpoint : [bold]{provider['endpoint']}[/]")
        console.print(f"  Protocol : [dim]OpenAI-compatible REST (application/json)[/]")

    # ── Step 3: API key(s) ──────────────────────────────────────────────────
    env_updates: dict[str, str] = {}
    shared_key = ""

    if provider["needs_key"]:
        _print_step(3, total_steps, f"API Key  ({provider['key_env']})")

        existing_val = os.environ.get(provider["key_env"], "")
        if existing_val and existing_val not in ("unused", ""):
            console.print(
                f"  [green]✓[/] [bold]{provider['key_env']}[/] already set: {_masked(existing_val)}"
            )
            if _confirm("Keep existing key?", default=True):
                shared_key = existing_val
            else:
                shared_key = _prompt_secret(f"New {provider['key_env']}", env_var=provider["key_env"])
                if shared_key:
                    env_updates[provider["key_env"]] = shared_key
        else:
            console.print(
                f"  Enter your [bold]{provider['label']}[/] API key.\n"
                f"  [dim]Saved to [bold]{_env_file()}[/] as [bold]{provider['key_env']}[/][/]\n"
            )
            shared_key = _prompt_secret(provider["key_env"])
            if not shared_key:
                console.print("  [yellow]No key entered — skipping.[/]")
            else:
                env_updates[provider["key_env"]] = shared_key

        console.print()
        use_same = _confirm(
            "Use same key for all agents (Actor, Checker, Policy)?",
            default=True,
        )

        agent_keys: dict[str, str] = {}
        if not use_same:
            _print_step(3, total_steps, "Per-agent API keys")
            for agent_id, agent_label in AGENT_LABELS:
                console.print(f"\n  [bold]{agent_label}[/]")
                var_name = f"HPEMA_{agent_id.upper()}_API_KEY"
                val = _prompt_secret(var_name, env_var=var_name)
                if val:
                    env_updates[var_name] = val
                    agent_keys[agent_id] = var_name
                else:
                    agent_keys[agent_id] = provider["key_env"]
        else:
            agent_keys = {aid: provider["key_env"] for aid, _ in AGENT_LABELS}
    else:
        _print_step(3, total_steps, "API Key")
        console.print(f"  [dim]No API key required for [bold]{provider['label']}[/].[/]")
        env_updates[provider["key_env"]] = "unused"
        agent_keys = {aid: provider["key_env"] for aid, _ in AGENT_LABELS}

    # ── Step 4: Model names ─────────────────────────────────────────────────
    _print_step(4, total_steps, "Model names for each agent")
    # In edit mode use existing model as the per-agent default; otherwise provider default
    prefill_models = prefill.get("models", {})
    fallback_model = provider["default_model"]
    console.print(
        f"  [dim]Actor and Checker process all code. Policy runs RAG compliance checks.\n"
        "  Dafny Architect mirrors Actor unless you edit the config manually.[/]\n"
    )

    agent_models: dict[str, str] = {}
    # Check if all existing models are the same (offer "same for all" shortcut)
    existing_actor = prefill_models.get("actor", fallback_model) or fallback_model
    all_same = len(set(prefill_models.values())) <= 1 if prefill_models else True
    use_same_model = _confirm(
        f"Use the same model for all agents? (current: {existing_actor})",
        default=all_same,
    )

    if use_same_model:
        shared_model = _prompt_text_input("Model name", default=existing_actor)
        agent_models = {aid: shared_model for aid, _ in AGENT_LABELS}
    else:
        for agent_id, agent_label in AGENT_LABELS:
            console.print(f"\n  [bold]{agent_label}[/]")
            agent_default = prefill_models.get(agent_id, fallback_model) or fallback_model
            agent_models[agent_id] = _prompt_text_input("Model name", default=agent_default)

    # ── Step 5: Model family ────────────────────────────────────────────────
    _print_step(5, total_steps, "Model family profile")
    # In edit mode prefer existing family; new mode prefers provider default
    default_family = prefill.get("family") or provider.get("default_family", "generic")
    console.print(
        f"  [dim]Family controls thinking-mode kwargs, system-role support, and JSON schema.\n"
        f"  Current: [bold]{default_family}[/]. Change if switching model family.[/]\n"
    )
    try:
        default_idx = ALL_FAMILIES.index(default_family)
    except ValueError:
        default_idx = 0
    # Rotate list so current family is first
    ordered_families = [ALL_FAMILIES[default_idx]] + [f for i, f in enumerate(ALL_FAMILIES) if i != default_idx]
    try:
        family_idx = _pick_option(ordered_families, "Model family")
    except KeyboardInterrupt:
        family_idx = 0
    chosen_family = ordered_families[family_idx]
    console.print(f"\n  [green]✓[/] Family: [bold]{chosen_family}[/]\n")

    # ── Dafny detection ─────────────────────────────────────────────────────
    dafny_path = _detect_dafny()
    z3_path    = _detect_z3()

    # ── Step 6: Review & Save ───────────────────────────────────────────────
    _print_step(6, total_steps, "Review & Save")

    table = Table(box=None, show_header=True, padding=(0, 2))
    table.add_column("Agent",    style="bold", width=12)
    table.add_column("Model",    width=36)
    table.add_column("Family",   width=18, style="dim")
    table.add_column("Key env",  width=28, style="dim")

    cfg_out: dict[str, dict] = {}
    for agent_id, agent_label in AGENT_LABELS:
        short = agent_label.split("(")[0].strip()
        cfg_out[agent_id] = {
            "endpoint": provider["endpoint"],
            "model":    agent_models[agent_id],
            "key_env":  agent_keys.get(agent_id, provider["key_env"]),
            "family":   chosen_family,
        }
        table.add_row(
            short,
            agent_models[agent_id],
            chosen_family,
            agent_keys.get(agent_id, provider["key_env"]),
        )

    console.print(table)
    console.print(f"\n  [dim]Config will be saved to: [bold]{target_config}[/][/]")
    console.print()

    # Dafny status
    if dafny_path:
        console.print(f"  [green]✓[/] Dafny detected: [dim]{dafny_path}[/]")
        if z3_path:
            console.print(f"  [green]✓[/] Z3 detected: [dim]{z3_path}[/]")
    else:
        console.print(
            "\n  [yellow]⚠  Dafny not detected.[/]\n"
            "  [dim]The verification stage will be unavailable until Dafny is installed.\n"
            "  Quick install (macOS):  brew install dotnet dafny[/]\n"
        )

    console.print()
    action_label = "Save changes?" if is_edit else "Save this configuration?"
    confirmed = _confirm(action_label, default=True)
    if not confirmed:
        console.print("\n  [yellow]Setup cancelled — no changes made.[/]\n")
        return False

    # Write files
    env_path = _env_file()
    if env_updates:
        _write_env(env_updates)
        for var, val in env_updates.items():
            os.environ[var] = val
        console.print(f"\n  [green]✓[/] API keys written to [bold]{env_path}[/]")

    _write_config(target_config, cfg_out, dafny_path, z3_path, chosen_family)
    console.print(f"  [green]✓[/] Config {'updated' if is_edit else 'written'} → [bold]{target_config}[/]")

    os.environ["HPEMA_CONFIG"] = str(target_config)

    if not dafny_path:
        console.print(
            "\n  [bold yellow]Note:[/] config written with [dim]binary_path: dafny[/] — "
            "relies on Dafny being on PATH.\n"
            "  Once installed, re-run [bold]/setup[/] to auto-detect the path."
        )

    # ── API key validation — catch missing keys before restart ─────────────
    _validate_and_fix_keys(cfg_out)

    # Offer restart
    if _confirm("Restart HPEMA now to apply?", default=True):
        _restart(str(target_config))
    else:
        console.print(
            "\n  [dim]Config saved. Restart HPEMA for changes to fully take effect.\n"
            "  New keys are already active in this session.[/]\n"
        )

    return True
