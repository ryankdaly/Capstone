"""Model registry — maps agent roles to LLM endpoints.

Reads from the central HpemaConfig. Boeing swaps models by editing
hpema_config.yaml or setting env vars (HPEMA_ACTOR_ENDPOINT, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.config import HpemaConfig, ModelEndpointConfig, settings


@dataclass(frozen=True)
class ResolvedModel:
    """Fully resolved model connection info, ready for the async client."""

    endpoint: str
    model: str
    api_key: str
    family: str = "generic"
    enable_thinking: bool = False
    extra_body: dict = field(default_factory=dict)


class ModelRegistry:
    """Resolves agent role → model endpoint from config."""

    def __init__(self, config: HpemaConfig | None = None) -> None:
        self._config = config or settings

    def _resolve(self, cfg: ModelEndpointConfig) -> ResolvedModel:
        return ResolvedModel(
            endpoint=cfg.endpoint,
            model=cfg.model,
            api_key=cfg.api_key,
            family=cfg.family,
            enable_thinking=cfg.enable_thinking,
            extra_body=cfg.extra_body,
        )

    @property
    def actor(self) -> ResolvedModel:
        return self._resolve(self._config.models.actor)

    @property
    def checker(self) -> ResolvedModel:
        return self._resolve(self._config.models.checker)

    @property
    def policy(self) -> ResolvedModel:
        return self._resolve(self._config.models.policy)

    @property
    def dafny_architect(self) -> ResolvedModel:
        return self._resolve(self._config.models.dafny_architect)

    def get(self, role: str) -> ResolvedModel:
        """Look up by string name (useful for dynamic dispatch)."""
        mapping = {
            "actor": self.actor,
            "checker": self.checker,
            "policy": self.policy,
            "dafny_architect": self.dafny_architect,
        }
        if role not in mapping:
            raise ValueError(f"Unknown agent role: {role!r}. Valid: {list(mapping)}")
        return mapping[role]
