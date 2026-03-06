"""Central configuration — loads from hpema_config.yaml + env var overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "hpema_config.yaml"


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class ModelEndpointConfig(BaseModel):
    endpoint: str = "https://llm-api.arc.vt.edu/api/v1"
    model: str = "gpt-oss-120b"
    api_key_env: str = "HPEMA_API_KEY"

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")


class ModelsConfig(BaseModel):
    actor: ModelEndpointConfig = Field(default_factory=ModelEndpointConfig)
    checker: ModelEndpointConfig = Field(default_factory=ModelEndpointConfig)
    policy: ModelEndpointConfig = Field(default_factory=ModelEndpointConfig)


class PoliciesConfig(BaseModel):
    standards_dir: str = "data/standards"
    default_standard: str = "DO_178C"
    embedding_model: str = "all-MiniLM-L6-v2"


class VerificationConfig(BaseModel):
    prover: str = "dafny"
    timeout_seconds: int = 120
    binary_path: str = "dafny"


class PipelineConfig(BaseModel):
    max_iterations: int = 3
    require_human_approval: bool = True
    audit_log_dir: str = "logs/audit"


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

class HpemaConfig(BaseModel):
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    policies: PoliciesConfig = Field(default_factory=PoliciesConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)


def load_config(path: Path | None = None) -> HpemaConfig:
    """Load config from YAML, falling back to defaults if file is missing."""
    path = path or DEFAULT_CONFIG_PATH
    if path.exists():
        raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        return HpemaConfig(**raw)
    return HpemaConfig()


# Module-level singleton — import this everywhere.
settings = load_config()
