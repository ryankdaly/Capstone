"""Intent classifier for HPEMA's build-mode REPL.

Classifies free-text input into one of five intents so the dispatcher
in repl.py can route to the right handler without always running the
full pipeline.

Classification strategy
------------------------
1. Fast heuristics (0 ms, covers the common cases):
   - Long input (> 15 words) with no agent-action keywords → GENERATE
   - Greeting / single-word / question mark → CONVERSE
2. LLM classifier (≈1-2 s, 20 tokens) for everything else.
   Uses the actor endpoint at temperature=0 so it never hallucinates
   a different intent string.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from enum import Enum

# ---------------------------------------------------------------------------
# Intent enum
# ---------------------------------------------------------------------------

class Intent(str, Enum):
    GENERATE      = "generate"       # full pipeline with current settings
    RERUN_CHECKER = "rerun_checker"  # checker only on last code
    RERUN_DAFNY   = "rerun_dafny"    # dafny architect + verifier on last code
    RERUN_POLICY  = "rerun_policy"   # policy agent only on last code
    CONVERSE      = "converse"       # answer inline without running agents


@dataclass
class ClassifiedIntent:
    intent: Intent
    # Populated when the user pastes code inline (future extension).
    code_override: str | None = None


# ---------------------------------------------------------------------------
# Keyword patterns for the heuristic fast-path
# ---------------------------------------------------------------------------

_AGENT_ACTION_RE = re.compile(
    r"\b("
    r"re[-\s]?run|run again|check again|redo|retry|"
    r"checker|dafny|policy|verify|verif|compliance|"
    r"test again|recheck"
    r")\b|run_checker|run_dafny|run_policy",
    re.IGNORECASE,
)

_GREETING_RE = re.compile(
    r"^\s*("
    r"hi|hello|hey|sup|yo|howdy|hiya|what'?s up|"
    r"thanks|thank you|thx|ty|ok|okay|sure|"
    r"what (is|are|does)|how (do|does|can)|why|explain"
    r")[\s?!.]*$",
    re.IGNORECASE,
)

# Match "checker" in any form: word, underscore-joined, slash-joined
_CHECKER_RE = re.compile(r"\bchecker\b|run[_/]checker|re[_/]?run[_/]checker", re.IGNORECASE)
_DAFNY_RE   = re.compile(r"\bdafny\b|run[_/]dafny|re[_/]?run[_/]dafny",       re.IGNORECASE)
_POLICY_RE  = re.compile(r"\bpolicy\b|run[_/]policy|re[_/]?run[_/]policy",    re.IGNORECASE)

# Verbs that signal intent to run a specific agent by name (full-sentence form)
_CHECKER_VERBS  = re.compile(r"\b(run|re[-\s]?run|check|redo|retry)\b.{0,50}checker", re.IGNORECASE)
_DAFNY_VERBS    = re.compile(r"\b(run|re[-\s]?run|verify|redo|retry)\b.{0,50}dafny",  re.IGNORECASE)
_POLICY_VERBS   = re.compile(r"\b(run|re[-\s]?run|check|redo|retry)\b.{0,50}policy",  re.IGNORECASE)

# Also handle "checker again", "dafny again", "policy again" without leading verb
_CHECKER_AGAIN  = re.compile(r"checker.{0,20}again|again.{0,20}checker", re.IGNORECASE)
_DAFNY_AGAIN    = re.compile(r"dafny.{0,20}again|again.{0,20}dafny",     re.IGNORECASE)
_POLICY_AGAIN   = re.compile(r"policy.{0,20}again|again.{0,20}policy",   re.IGNORECASE)


# ---------------------------------------------------------------------------
# LLM classifier prompt (very small — 20 tokens output max)
# ---------------------------------------------------------------------------

_CLASSIFIER_SYSTEM = """\
You are a one-word intent classifier for a safety-critical code generation assistant.

Classify the user message into EXACTLY ONE of these words:
  generate      — user wants to generate or create new code (full pipeline)
  rerun_checker — user wants to re-run the checker/review on the last code
  rerun_dafny   — user wants to re-run the Dafny formal verifier on the last code
  rerun_policy  — user wants to re-run the policy compliance check on the last code
  converse      — user is asking a question, chatting, or doing anything else

Rules:
- If the user says "re-run", "run again", "check again", "redo", look for which agent they mention.
- If no agent is specified but they say "re-run" or "again", output converse.
- If the message is a greeting, question, or off-topic, output converse.
- Output ONLY the single word. No punctuation. No explanation.
"""

_CLASSIFIER_USER_TMPL = "Message: {text}"


async def _llm_classify(text: str) -> Intent:
    """Call the actor LLM with a minimal classify prompt. Falls back to GENERATE on error."""
    try:
        from backend.config import load_config
        from backend.services.llm.client import LLMClient
        from backend.services.llm.model_registry import ModelRegistry

        config = load_config()
        registry = ModelRegistry(config)
        client = LLMClient(registry)

        result_tokens: list[str] = []
        async for token in client.generate_stream(
            role="actor",
            system_prompt=_CLASSIFIER_SYSTEM,
            user_prompt=_CLASSIFIER_USER_TMPL.format(text=text[:500]),
            temperature=0.0,
            max_tokens=20,
        ):
            result_tokens.append(token)

        await client.aclose()
        raw = "".join(result_tokens).strip().lower().split()[0] if result_tokens else ""

        _MAP = {
            "generate":      Intent.GENERATE,
            "rerun_checker": Intent.RERUN_CHECKER,
            "rerun_dafny":   Intent.RERUN_DAFNY,
            "rerun_policy":  Intent.RERUN_POLICY,
            "converse":      Intent.CONVERSE,
        }
        return _MAP.get(raw, Intent.GENERATE)

    except (Exception, asyncio.CancelledError):
        return Intent.GENERATE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify(text: str, has_history: bool) -> ClassifiedIntent:
    """Classify *text* synchronously.

    Uses heuristics first (0 ms); falls through to an LLM call only when
    the input is genuinely ambiguous.  Re-run intents are always returned
    as-is — the dispatcher handlers print friendly "no history yet" messages
    when there is nothing to re-run on.

    Args:
        text:        Raw user input (stripped).
        has_history: Kept for API compatibility; no longer gates re-run intents.
    """
    words = text.split()
    n = len(words)

    # ── Fast path A: long input, no agent-action keywords → GENERATE ──────
    if n > 15 and not _AGENT_ACTION_RE.search(text):
        return ClassifiedIntent(intent=Intent.GENERATE)

    # ── Fast path B: greeting / conversational opener → CONVERSE ──────────
    if n <= 6 and _GREETING_RE.match(text):
        return ClassifiedIntent(intent=Intent.CONVERSE)

    # ── Fast path C: short input containing an agent name → re-run ────────
    # Covers: "run checker", "run_checker", "checker again", "could you run
    # the checker", "re-run dafny", etc.  Agent name in any short utterance
    # is a strong signal — users don't normally type "checker" to describe
    # a requirement they want to generate.
    if n <= 15:
        if _CHECKER_VERBS.search(text) or _CHECKER_AGAIN.search(text) or _CHECKER_RE.search(text):
            return ClassifiedIntent(intent=Intent.RERUN_CHECKER)
        if _DAFNY_VERBS.search(text) or _DAFNY_AGAIN.search(text) or _DAFNY_RE.search(text):
            return ClassifiedIntent(intent=Intent.RERUN_DAFNY)
        if _POLICY_VERBS.search(text) or _POLICY_AGAIN.search(text) or _POLICY_RE.search(text):
            return ClassifiedIntent(intent=Intent.RERUN_POLICY)

    # ── Fast path D: longer input with agent name + action verb → re-run ──
    if _CHECKER_VERBS.search(text) or _CHECKER_AGAIN.search(text):
        return ClassifiedIntent(intent=Intent.RERUN_CHECKER)
    if _DAFNY_VERBS.search(text) or _DAFNY_AGAIN.search(text):
        return ClassifiedIntent(intent=Intent.RERUN_DAFNY)
    if _POLICY_VERBS.search(text) or _POLICY_AGAIN.search(text):
        return ClassifiedIntent(intent=Intent.RERUN_POLICY)

    # ── Fast path E: longer input, no agent keywords → GENERATE ───────────
    if n > 8:
        return ClassifiedIntent(intent=Intent.GENERATE)

    # ── LLM classify: short/ambiguous input that doesn't mention an agent ──
    try:
        intent = asyncio.run(_llm_classify(text))
    except (Exception, asyncio.CancelledError):
        intent = Intent.GENERATE

    return ClassifiedIntent(intent=intent)
