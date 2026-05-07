"""Model catalogue for the evaluation harness.

Staged execution order (per eval plan):
  Stage 1: OSS NVIDIA NIM  — mistral-large, qwen3-235b, gemma-3-27b
  Stage 2: DeepSeek        — deepseek-v4-pro, deepseek-v4-flash
  Stage 3: OSS + DeepSeek in popular benchmarks (EvalPlus, CyberSecEval, SecurityEval)
  Stage 4: gpt-5.4-mini    — custom then popular benchmarks

Stage 4 models have enabled=False. Pass --include-staged to run them.

API keys come from environment variables — never hardcode them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelSpec:
    id: str                          # short name used in filenames / CLI args
    model_id: str                    # model string sent to the API
    endpoint: str                    # base URL of the OpenAI-compatible endpoint
    api_key_env: str                 # env var that holds the API key
    family: str = "generic"          # HPEMA family profile key (profiles.py)
    temperature: float = 0.7
    enable_thinking: bool = False    # set True to activate thinking for toggle-off families
    extra_params: dict = field(default_factory=dict)       # top-level API params (e.g. reasoning_effort)
    extra_body_override: dict = field(default_factory=dict)  # extra_body payload additions
    enabled: bool = True             # False = staged; skipped by --all unless --include-staged
    stage_note: str = ""             # explains why this model is staged

    # Pricing (USD per 1M tokens). 0.0 for free/self-hosted endpoints.
    # Cache hit pricing applies when the provider returns cached token counts.
    input_price_per_m: float = 0.0
    cached_input_price_per_m: float = 0.0
    output_price_per_m: float = 0.0

    def compute_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
    ) -> float:
        """Compute estimated USD cost for one API call."""
        non_cached = max(0, input_tokens - cached_input_tokens)
        return (
            non_cached * self.input_price_per_m / 1_000_000
            + cached_input_tokens * self.cached_input_price_per_m / 1_000_000
            + output_tokens * self.output_price_per_m / 1_000_000
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODELS: dict[str, ModelSpec] = {

    # ── Stage 1: OSS NVIDIA NIM (free-tier) ─────────────────────────────────
    "mistral-large": ModelSpec(
        id="mistral-large",
        model_id="mistralai/mistral-large-3-675b-instruct-2512",
        endpoint="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_API_KEY",
        family="mistral",
        input_price_per_m=0.0,
        cached_input_price_per_m=0.0,
        output_price_per_m=0.0,
    ),
    "qwen3-coder": ModelSpec(
        id="qwen3-coder",
        model_id="qwen/qwen3-coder-480b-a35b-instruct",
        endpoint="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_API_KEY",
        family="qwen3",
        enable_thinking=False,
        temperature=0.7,
        extra_params={"top_p": 0.8},
        input_price_per_m=0.0,
        cached_input_price_per_m=0.0,
        output_price_per_m=0.0,
    ),
    "gemma-3-27b": ModelSpec(
        id="gemma-3-27b",
        model_id="google/gemma-3-27b-it",
        endpoint="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_API_KEY",
        family="gemma3",
        input_price_per_m=0.0,
        cached_input_price_per_m=0.0,
        output_price_per_m=0.0,
    ),

    # ── Stage 2: DeepSeek (paid, api.deepseek.com) ──────────────────────────
    "deepseek-v4-pro": ModelSpec(
        id="deepseek-v4-pro",
        model_id="deepseek-v4-pro",
        endpoint="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        family="deepseek-v4",
        enable_thinking=True,        # activates reasoning_effort=high + thinking body
        extra_params={"reasoning_effort": "high"},
        extra_body_override={"thinking": {"type": "enabled"}},
        input_price_per_m=0.435,
        cached_input_price_per_m=0.003625,
        output_price_per_m=0.87,
    ),
    "deepseek-v4-flash": ModelSpec(
        id="deepseek-v4-flash",
        model_id="deepseek-v4-flash",
        endpoint="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        family="deepseek-v3",        # non-reasoning flash — same profile as V3
        input_price_per_m=0.14,
        cached_input_price_per_m=0.0028,
        output_price_per_m=0.28,
    ),

    # ── Stage 4: GPT-5.4-mini (staged — run last) ───────────────────────────
    "gpt-5.4-mini": ModelSpec(
        id="gpt-5.4-mini",
        model_id="gpt-5.4-mini-2026-03-17",
        endpoint="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        family="openai-gpt4o",
        enabled=False,
        stage_note="Stage 4 — run after OSS + DeepSeek custom eval and popular benchmarks complete",
        input_price_per_m=0.75,
        cached_input_price_per_m=0.075,
        output_price_per_m=4.50,
    ),
}


# Convenience groupings matching the staged execution order
STAGE_1 = [m for m in MODELS.values() if m.api_key_env == "NVIDIA_API_KEY"]
STAGE_2 = [m for m in MODELS.values() if m.api_key_env == "DEEPSEEK_API_KEY"]
STAGE_4 = [m for m in MODELS.values() if not m.enabled]
ALL_ENABLED = [m for m in MODELS.values() if m.enabled]
ALL_MODELS = list(MODELS.values())


def get_model(name: str) -> ModelSpec:
    if name not in MODELS:
        raise ValueError(
            f"Unknown model {name!r}. Available: {list(MODELS)}"
        )
    return MODELS[name]
