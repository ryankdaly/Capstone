"""Model family profiles — hard-coded kwarg mappings for known model families.

Each profile captures everything the LLMClient needs to call a model correctly:
  - Whether the model supports a system role message
  - Whether it supports response_format=json_schema (constrained decoding)
  - How thinking/reasoning is controlled (and the exact kwargs to send)

Adding a new family: add one entry to PROFILES and set family: <name> in the
model config.  No other code changes required.

Thinking modes
--------------
  "none"        model does not think; no reasoning kwargs sent
  "always"      model always thinks natively (R1, MiniMax, o-series);
                no control kwargs; pipeline's <think> stripping handles output
  "toggle-off"  thinking is OFF by default; enable_thinking=True activates it
  "toggle-on"   thinking is ON by default (Qwen3); enable_thinking=False
                deactivates it; enable_thinking=True sends explicit on-kwargs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ThinkingMode = Literal["none", "always", "toggle-off", "toggle-on"]


@dataclass(frozen=True)
class ModelFamilyProfile:
    # ── Capability flags ─────────────────────────────────────────────────────
    supports_system_role: bool = True
    """False → system content is prepended to the first user message."""

    supports_json_schema: bool = False
    """True → response_format=json_schema (constrained decoding) is used.
    False → prompt-only JSON enforcement via _json_prompt_suffix()."""

    # ── Thinking / reasoning ─────────────────────────────────────────────────
    thinking_mode: ThinkingMode = "none"

    thinking_on_extra_body: dict = field(default_factory=dict)
    """extra_body dict to merge when thinking is being activated."""

    thinking_off_extra_body: dict = field(default_factory=dict)
    """extra_body dict to merge when thinking is explicitly deactivated
    (only meaningful for toggle-on families like Qwen3)."""

    thinking_on_extra_params: dict = field(default_factory=dict)
    """Top-level API params (outside extra_body) when thinking is enabled.
    Example: {"reasoning_effort": "high"} for Mistral Magistral."""

    thinking_off_extra_params: dict = field(default_factory=dict)
    """Top-level API params when thinking is explicitly turned off."""

    # ── Notes ────────────────────────────────────────────────────────────────
    notes: str = ""
    """Free-form notes about quirks — not used by the client."""


# ---------------------------------------------------------------------------
# Family profiles
# ---------------------------------------------------------------------------

PROFILES: dict[str, ModelFamilyProfile] = {

    # ── Mistral (non-reasoning) ──────────────────────────────────────────────
    # Large 3, Small (pre-4), NeMo, Devstral
    # - Full json_schema constrained decoding supported on NVIDIA NIM
    # - No chat_template_kwargs accepted at request level on NIM
    # Model IDs: mistralai/mistral-large-3-675b-instruct-2512
    #            mistralai/mistral-small-*  mistralai/mistral-nemo-*
    "mistral": ModelFamilyProfile(
        supports_system_role=True,
        supports_json_schema=True,
        thinking_mode="none",
        notes="Mistral-Large-3, Mistral-NeMo, Devstral. No thinking kwargs.",
    ),

    # ── Mistral Magistral / Small 4 with reasoning ───────────────────────────
    # Uses top-level reasoning_effort param, NOT chat_template_kwargs.
    # Model IDs: mistralai/magistral-*  mistralai/mistral-small-4-*
    "mistral-reasoning": ModelFamilyProfile(
        supports_system_role=True,
        supports_json_schema=True,
        thinking_mode="toggle-off",
        thinking_on_extra_params={"reasoning_effort": "high"},
        thinking_off_extra_params={"reasoning_effort": "none"},
        notes=(
            "Magistral / Mistral-Small-4 reasoning variant. "
            "reasoning_effort is a top-level param, not in extra_body."
        ),
    ),

    # ── Kimi K2 (Moonshot AI) ───────────────────────────────────────────────
    # K2-Instruct: no thinking by default. K2-Thinking: thinking default on.
    # Thinking toggle uses extra_body: {"thinking": {"type": "enabled"/"disabled"}}
    # Tool calling + thinking: tool_choice must be "auto" or "none".
    # Model IDs: moonshotai/kimi-k2-instruct
    #            moonshotai/kimi-k2-thinking   (thinking default on)
    "kimi-k2": ModelFamilyProfile(
        supports_system_role=True,
        supports_json_schema=False,
        thinking_mode="toggle-off",
        thinking_on_extra_body={"thinking": {"type": "enabled"}},
        thinking_off_extra_body={"thinking": {"type": "disabled"}},
        notes=(
            "Kimi K2. Thinking format differs from Qwen/StepFun — uses "
            "extra_body.thinking.type, not chat_template_kwargs. "
            "Recommended temperature: 1.0 (thinking) / 0.6 (no thinking)."
        ),
    ),

    # ── Anthropic Claude (via OpenAI-compat / Bedrock) ───────────────────────
    # Native Anthropic API supports structured outputs and extended thinking.
    # OpenAI-compat layer strips reasoning content and doesn't support json_schema.
    # Use native AnthropicClient for production structured outputs.
    # Model IDs: claude-opus-4-*  claude-sonnet-4-*  claude-haiku-4-*
    "anthropic": ModelFamilyProfile(
        supports_system_role=True,
        supports_json_schema=False,
        thinking_mode="none",
        notes=(
            "Via OpenAI-compat: no json_schema, no thinking kwargs. "
            "For production, use native Anthropic client. "
            "Extended thinking only works via native API."
        ),
    ),

    # ── OpenAI GPT-4o series ─────────────────────────────────────────────────
    # GPT-4o (2024-08-06+) and GPT-4.1 fully support json_schema.
    # No thinking kwargs — not a reasoning model.
    # Model IDs: gpt-4o  gpt-4o-mini  gpt-4.1  gpt-4-turbo
    "openai-gpt4o": ModelFamilyProfile(
        supports_system_role=True,
        supports_json_schema=True,
        thinking_mode="none",
        notes="GPT-4o, GPT-4.1. Full json_schema constrained decoding.",
    ),

    # ── OpenAI o-series (native reasoners) ──────────────────────────────────
    # o1, o3, o3-mini, o4-mini always reason natively — no control kwargs.
    # o3-mini supports json_schema; o1 does not.
    # Set supports_json_schema=False to be safe across the series.
    # Model IDs: o1  o1-mini  o3  o3-mini  o4-mini
    "openai-reasoning": ModelFamilyProfile(
        supports_system_role=True,
        supports_json_schema=False,
        thinking_mode="always",
        notes=(
            "o1/o3/o3-mini/o4-mini. Always reason natively; no thinking kwargs. "
            "o3-mini supports json_schema but o1 does not — set False for safety."
        ),
    ),

    # ── Qwen 3 / Qwen 3.5 ───────────────────────────────────────────────────
    # Thinking is ON by default for all Qwen3 models.
    # Disable with chat_template_kwargs.enable_thinking=False.
    # thinking_budget is a separate top-level extra_body key (not nested).
    # Model IDs: qwen/qwen3-235b-a22b  qwen/qwen3-30b-a3b
    #            qwen/qwen3.5-*
    "qwen3": ModelFamilyProfile(
        supports_system_role=True,
        supports_json_schema=True,
        thinking_mode="toggle-on",
        thinking_on_extra_body={
            "chat_template_kwargs": {"enable_thinking": True},
            "thinking_budget": 2000,
        },
        thinking_off_extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
        notes=(
            "Qwen3/Qwen3.5 family. Thinking is ON by default — set "
            "enable_thinking=false in model config to suppress it. "
            "thinking_budget is a top-level extra_body key, not nested in "
            "chat_template_kwargs (different from StepFun)."
        ),
    ),

    # ── StepFun step-3.5-flash ───────────────────────────────────────────────
    # Sparse MoE (~11B active / 197B total). Fast. Thinking via chat_template_kwargs.
    # thinking_budget is INSIDE chat_template_kwargs (different from Qwen3).
    # Model IDs: stepfun-ai/step-3-5-flash
    "stepfun": ModelFamilyProfile(
        supports_system_role=True,
        supports_json_schema=False,
        thinking_mode="toggle-off",
        thinking_on_extra_body={
            "chat_template_kwargs": {"enable_thinking": True, "thinking_budget": 2000},
        },
        thinking_off_extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
        notes=(
            "StepFun step-3.5-flash. thinking_budget lives INSIDE "
            "chat_template_kwargs — different nesting from Qwen3."
        ),
    ),

    # ── MiniMax M2 / M2.1 / M2.7 ────────────────────────────────────────────
    # Always uses interleaved thinking — no disable toggle found.
    # thinking tags (<think>...</think>) must be preserved in conversation history.
    # Model IDs: minimaxai/minimax-m2  minimaxai/minimax-m2.1  minimaxai/minimax-m2.7
    "minimax": ModelFamilyProfile(
        supports_system_role=True,
        supports_json_schema=False,
        thinking_mode="always",
        notes=(
            "MiniMax M2 series. Always thinks with interleaved <think> tags. "
            "No disable toggle. Preserve thinking tags in conversation history."
        ),
    ),

    # ── Gemma 3 ─────────────────────────────────────────────────────────────
    # Does NOT support system role — system content merged into user message.
    # No thinking, no constrained decoding on NIM.
    # Model IDs: google/gemma-3-27b-it  google/gemma-3-1b-it
    "gemma3": ModelFamilyProfile(
        supports_system_role=False,
        supports_json_schema=False,
        thinking_mode="none",
        notes=(
            "Gemma 3. No system role — content merged into first user message. "
            "No constrained decoding on NVIDIA NIM."
        ),
    ),

    # ── Gemma 4 ─────────────────────────────────────────────────────────────
    # Added system role support. json_schema via guided decoding on NIM.
    # Streaming + tool calling is incompatible (don't use tools + stream=True).
    # Model IDs: google/gemma-4-31b-it  google/gemma-4-26b-a4b-it
    "gemma4": ModelFamilyProfile(
        supports_system_role=True,
        supports_json_schema=True,
        thinking_mode="none",
        notes=(
            "Gemma 4. System role and json_schema supported. "
            "Do NOT combine stream=True with tool calling."
        ),
    ),

    # ── DeepSeek R1 / R1-0528 ───────────────────────────────────────────────
    # Always thinks natively via <think> tags — no toggle needed.
    # R1-0528 added system role support (original R1 lacked it).
    # No json_schema; use json_object or prompt-only.
    # Model IDs: deepseek-ai/deepseek-r1  deepseek-ai/deepseek-r1-0528
    "deepseek-r1": ModelFamilyProfile(
        supports_system_role=True,
        supports_json_schema=False,
        thinking_mode="always",
        notes=(
            "DeepSeek R1 / R1-0528. Always reasons via <think> tags. "
            "No control kwargs. R1-0528 added system prompt support. "
            "No json_schema — use prompt-only JSON enforcement."
        ),
    ),

    # ── DeepSeek V3 / V3.2 / V4 ────────────────────────────────────────────
    # Does not think. System role supported.
    # json_schema NOT supported — json_object only (needs "json" in prompt).
    # Our prompt-only fallback includes "JSON" in the injected suffix, satisfying
    # V3's requirement automatically.
    # Model IDs: deepseek-ai/deepseek-v3.2  deepseek-ai/deepseek-v4-pro
    "deepseek-v3": ModelFamilyProfile(
        supports_system_role=True,
        supports_json_schema=False,
        thinking_mode="none",
        notes=(
            "DeepSeek V3/V3.2/V4. No thinking. No json_schema — prompt-only "
            "JSON enforcement is sufficient (our suffix includes 'JSON' text). "
            "json_object mode requires 'json' in the prompt."
        ),
    ),

    # ── Generic fallback ────────────────────────────────────────────────────
    # Conservative defaults for unknown/untested models.
    # No constrained decoding, system role assumed, no thinking.
    "generic": ModelFamilyProfile(
        supports_system_role=True,
        supports_json_schema=False,
        thinking_mode="none",
        notes="Conservative fallback. Set family explicitly for best results.",
    ),
}


def get_profile(family: str) -> ModelFamilyProfile:
    """Return the profile for *family*, falling back to 'generic' if unknown."""
    return PROFILES.get(family, PROFILES["generic"])
