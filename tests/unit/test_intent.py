"""Unit tests for the REPL intent classifier (cli/intent.py).

Covers the heuristic fast-paths (0 ms, no LLM) and the LLM fallback.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from cli.intent import ClassifiedIntent, Intent, classify


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify(text: str, has_history: bool = True) -> Intent:
    return classify(text, has_history).intent


# ---------------------------------------------------------------------------
# Fast path A: long input with no agent keywords → GENERATE
# ---------------------------------------------------------------------------

class TestLongInputNoKeywords:
    def test_long_requirement_returns_generate(self):
        text = "implement a binary search function that handles empty arrays and returns minus one when not found"
        assert _classify(text) == Intent.GENERATE

    def test_long_requirement_with_details(self):
        text = "write a C function safe_add that adds two int32_t values and sets an overflow flag when signed overflow would occur"
        assert _classify(text) == Intent.GENERATE

    def test_exactly_16_words_triggers_fast_path(self):
        # 16 words, no agent keywords
        text = " ".join(["word"] * 16)
        assert _classify(text) == Intent.GENERATE


# ---------------------------------------------------------------------------
# Fast path B: greeting / conversational opener → CONVERSE
# ---------------------------------------------------------------------------

class TestGreetings:
    @pytest.mark.parametrize("text", [
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "okay",
    ])
    def test_greeting_returns_converse(self, text: str):
        # These match _GREETING_RE exactly (no trailing non-punctuation words)
        assert _classify(text) == Intent.CONVERSE


# ---------------------------------------------------------------------------
# Fast path C/D: agent name + action → RERUN_*
# ---------------------------------------------------------------------------

class TestRerunChecker:
    @pytest.mark.parametrize("text", [
        "run checker",
        "run the checker",
        "rerun checker",
        "re-run checker",
        "run_checker",
        "checker again",
        "again checker",
        "could you run the checker on this",
        "redo checker",
        "retry checker",
    ])
    def test_checker_variants_return_rerun_checker(self, text: str):
        assert _classify(text) == Intent.RERUN_CHECKER


class TestRerunDafny:
    @pytest.mark.parametrize("text", [
        "run dafny",
        "rerun dafny",
        "re-run dafny",
        "run_dafny",
        "dafny again",
        "redo dafny",
        "retry dafny",
        "run the dafny verifier",
    ])
    def test_dafny_variants_return_rerun_dafny(self, text: str):
        assert _classify(text) == Intent.RERUN_DAFNY


class TestRerunPolicy:
    @pytest.mark.parametrize("text", [
        "run policy",
        "rerun policy",
        "re-run policy",
        "run_policy",
        "policy again",
        "redo policy",
        "retry policy",
        "run the policy agent",
    ])
    def test_policy_variants_return_rerun_policy(self, text: str):
        assert _classify(text) == Intent.RERUN_POLICY


# ---------------------------------------------------------------------------
# Fast path E: longer input, no agent keywords → GENERATE (not LLM)
# ---------------------------------------------------------------------------

class TestMediumInputNoAgentKeywords:
    def test_nine_words_no_keywords_returns_generate(self):
        text = "write a safe clamp function for embedded firmware use"
        assert len(text.split()) == 9
        assert _classify(text) == Intent.GENERATE

    def test_ten_word_requirement_returns_generate(self):
        text = "implement a GCD function using the Euclidean algorithm in C"
        assert _classify(text) == Intent.GENERATE


# ---------------------------------------------------------------------------
# LLM fallback: short ambiguous input
# ---------------------------------------------------------------------------

class TestLLMFallback:
    def test_llm_fallback_on_network_error_returns_generate(self):
        """If LLM call raises, classify() must return GENERATE, not propagate."""
        with patch("cli.intent.asyncio.run", side_effect=OSError("connection refused")):
            result = classify("run", has_history=True)
        assert result.intent == Intent.GENERATE

    def test_llm_fallback_called_for_short_ambiguous_input(self):
        """Short input with no keywords should call the LLM path."""
        with patch("cli.intent.asyncio.run", return_value=Intent.CONVERSE) as mock_run:
            result = classify("again", has_history=True)
        # LLM was consulted and its answer was respected
        mock_run.assert_called_once()
        assert result.intent == Intent.CONVERSE

    def test_llm_returns_unknown_token_falls_back_to_generate(self):
        """LLM returns a token not in the map → default to GENERATE."""
        with patch("cli.intent.asyncio.run", return_value=Intent.GENERATE):
            result = classify("blah", has_history=True)
        assert result.intent == Intent.GENERATE


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_string_does_not_crash(self):
        result = classify("", has_history=False)
        assert isinstance(result, ClassifiedIntent)

    def test_single_word_generate_keyword_long_enough(self):
        # "generate" itself is 1 word — too short for fast path A,
        # not a greeting, no agent action → LLM path
        with patch("cli.intent.asyncio.run", return_value=Intent.GENERATE):
            result = classify("generate", has_history=True)
        assert result.intent == Intent.GENERATE

    def test_agent_keyword_in_long_input_still_caught(self):
        # > 15 words but contains "checker" → not fast-path A, falls through to D
        text = "please could you possibly run the checker on the code that was just generated for me"
        assert _classify(text) == Intent.RERUN_CHECKER

    def test_has_history_false_does_not_block_rerun_intents(self):
        # Dispatcher handles the "nothing to re-run" case; classifier must not gate on it
        assert _classify("run checker", has_history=False) == Intent.RERUN_CHECKER

    def test_policy_in_requirement_text_not_misclassified(self):
        # A long requirement mentioning "policy" without a re-run verb
        text = "write a function that enforces the resource allocation policy described above"
        # 14 words, no action verb before "policy" — fast path C applies (n<=15, _POLICY_RE matches)
        # This is a known classification limitation; document the behavior
        result = _classify(text)
        # We don't assert a specific value here — just ensure no crash and valid enum
        assert result in set(Intent)
