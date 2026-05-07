"""Central configuration — loads from hpema_config.yaml + env var overrides."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "hpema_config.yaml"


# ---------------------------------------------------------------------------
# Portable path helpers — used by both config loader and setup wizard
# ---------------------------------------------------------------------------

def hpema_home() -> Path:
    """Writable user data directory. Created on demand.

    Platform defaults (overridable with the HPEMA_HOME env var):
      Windows  → %APPDATA%\\hpema   (e.g. C:\\Users\\you\\AppData\\Roaming\\hpema)
      Unix     → ~/.hpema

    Using %APPDATA% on Windows follows Windows app-data conventions and avoids
    dot-prefixed directories in the user profile root, which are hidden and
    unfamiliar to Windows users.
    """
    override = os.environ.get("HPEMA_HOME")
    if override:
        p = Path(override)
    elif sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        p = Path(appdata) / "hpema" if appdata else Path.home() / "AppData" / "Roaming" / "hpema"
    else:
        p = Path.home() / ".hpema"
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_data_path(configured: str, fallback_subdir: str) -> Path:
    """Resolve a data directory path.

    Resolution order:
    1. Absolute path in config → use as-is.
    2. Relative path → try importlib.resources (bundled package data in wheel).
    3. Fallback → ~/.hpema/{fallback_subdir} (always writable, always accessible).

    Relative paths from manually-written configs (e.g. 'data/standards') resolve
    against the package data first so wheel installs work without any cwd assumption.
    """
    p = Path(configured)
    if p.is_absolute():
        return p

    # Try bundled package data (works for both wheel and editable installs)
    try:
        import contextlib
        import importlib.resources as _ilr
        pkg_ref = _ilr.files("hpema_data") / fallback_subdir
        with contextlib.suppress(Exception):
            with _ilr.as_file(pkg_ref) as real_path:  # type: ignore[attr-defined]
                if Path(real_path).exists():
                    return Path(real_path)
    except Exception:
        pass

    return hpema_home() / fallback_subdir


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class ModelEndpointConfig(BaseModel):
    endpoint: str = "https://llm-api.arc.vt.edu/api/v1"
    model: str = "gpt-oss-120b"
    api_key_env: str = "HPEMA_API_KEY"
    family: str = Field(
        default="generic",
        description=(
            "Model family profile key. Controls system-role handling, constrained "
            "decoding, and thinking kwargs. Known families: mistral, mistral-reasoning, "
            "kimi-k2, anthropic, openai-gpt4o, openai-reasoning, qwen3, stepfun, "
            "minimax, gemma3, gemma4, deepseek-r1, deepseek-v3. "
            "Use 'generic' for untested models."
        ),
    )
    enable_thinking: bool = Field(
        default=False,
        description=(
            "Activate thinking/reasoning for models that support it. "
            "Ignored for families with thinking_mode='none' or 'always'. "
            "For toggle-on families (qwen3), False explicitly disables thinking."
        ),
    )
    extra_body: dict = Field(
        default_factory=dict,
        description=(
            "Additional JSON body fields forwarded verbatim to the API. "
            "Merged on top of profile-derived kwargs — use for one-off overrides "
            "not captured by the family profile."
        ),
    )

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")


class ModelsConfig(BaseModel):
    actor: ModelEndpointConfig = Field(default_factory=ModelEndpointConfig)
    checker: ModelEndpointConfig = Field(default_factory=ModelEndpointConfig)
    policy: ModelEndpointConfig = Field(default_factory=ModelEndpointConfig)
    dafny_architect: ModelEndpointConfig = Field(default_factory=ModelEndpointConfig)


class PoliciesConfig(BaseModel):
    standards_dir: str = "data/standards"
    chromadb_dir: str = "data/chromadb"
    default_standard: str = "DO_178C"
    embedding_model: str = "all-MiniLM-L6-v2"


class VerificationConfig(BaseModel):
    prover: str = "dafny"
    timeout_seconds: int = 120
    binary_path: str = "dafny"
    solver_path: str | None = None  # e.g. /opt/homebrew/bin/z3; None = let Dafny find Z3 on PATH


class PipelineConfig(BaseModel):
    max_iterations: int = 3
    require_human_approval: bool = True
    audit_log_dir: str = "logs/audit"
    stage: str = "policy"  # "actor", "checker", "policy" — controls how far the pipeline runs


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

class HpemaConfig(BaseModel):
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    policies: PoliciesConfig = Field(default_factory=PoliciesConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)


def find_config_path() -> Path | None:
    """Locate the config file without loading it.

    Resolution order — same for every install mode (dev, editable, wheel):

    1. ``HPEMA_CONFIG`` env var (absolute path only).
       Developers who want a custom config set this explicitly in their shell.
       Relative paths are ignored — they were a source of cross-env bleed.

    2. ``~/.hpema/hpema_config.yaml``
       Primary location written by the /setup wizard.

    3. Any other ``~/.hpema/hpema_config*.yaml`` (alphabetical).
       Named configs created by /setup ("create new").

    Nothing in the repo directory is ever auto-loaded. Devs use HPEMA_CONFIG.
    """
    # 1. Explicit absolute override
    env_val = os.environ.get("HPEMA_CONFIG", "")
    if env_val:
        p = Path(env_val)
        if p.is_absolute() and p.exists():
            return p
        # relative or missing → fall through (don't silently fail an explicit override)

    # 2 + 3. HPEMA home dir — always
    home = hpema_home()
    primary = home / "hpema_config.yaml"
    if primary.exists():
        return primary
    for p in sorted(home.glob("hpema_config*.yaml")):
        if p.exists():
            return p

    return None


def load_config(path: Path | None = None) -> HpemaConfig:
    """Load config from YAML. Pass an explicit ``path`` to bypass auto-discovery."""
    def _load(p: Path) -> HpemaConfig:
        raw: dict[str, Any] = yaml.safe_load(p.read_text()) or {}
        return HpemaConfig(**raw)

    if path is not None:
        return _load(path) if path.exists() else HpemaConfig()

    discovered = find_config_path()
    if discovered:
        return _load(discovered)

    return HpemaConfig()


# Module-level singleton — import this everywhere.
settings = load_config()
