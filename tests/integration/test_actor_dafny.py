"""Integration test: Actor-generated Dafny specs through DafnyRunner.

Requires:
  - HPEMA_API_KEY environment variable set (to call the LLM endpoint)
  - dafny binary on PATH

These tests call the real LLM and real Dafny binary to measure how well the
Actor agent generates parseable, verifiable formal specifications.  They are
skipped automatically when prerequisites are missing.

Results (verified=True/False, solver output) are printed so they can be used
as a baseline for B.5 prompt tuning — we do NOT assert verified=True here.
We only assert that Dafny can *parse* the spec (no syntax errors).
"""

from __future__ import annotations

import os
import shutil

import pytest

from backend.services.agents.actor import ActorAgent
from backend.services.llm.client import LLMClient
from backend.services.verification.dafny_runner import DafnyRunner

# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

dafny_installed = pytest.mark.skipif(
    shutil.which("dafny") is None,
    reason="dafny binary not on PATH — install Dafny to run these tests",
)
api_key_set = pytest.mark.skipif(
    not os.environ.get("HPEMA_API_KEY"),
    reason="HPEMA_API_KEY not set — required to call the LLM endpoint",
)

# ---------------------------------------------------------------------------
# Requirements to test.  Chosen to be simple enough that a well-prompted
# Actor should produce verifiable Dafny specs.
# ---------------------------------------------------------------------------

REQUIREMENTS: list[tuple[str, str]] = [
    (
        "absolute_value",
        "Implement a function that returns the absolute value of an integer.",
    ),
    (
        "clamp",
        "Implement a clamp function that restricts an integer value to [lo, hi].",
    ),
    (
        "safe_add",
        "Implement integer addition that detects overflow and returns 0 if it occurs.",
    ),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def actor() -> ActorAgent:
    return ActorAgent(LLMClient())


@pytest.fixture(scope="module")
def runner() -> DafnyRunner:
    return DafnyRunner(binary_path="dafny", timeout=60)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@dafny_installed
@api_key_set
@pytest.mark.asyncio
@pytest.mark.parametrize("name,requirement", REQUIREMENTS, ids=[r[0] for r in REQUIREMENTS])
async def test_actor_dafny_spec_parses(
    actor: ActorAgent,
    runner: DafnyRunner,
    name: str,
    requirement: str,
) -> None:
    """Actor must produce a non-empty dafny_spec that Dafny can parse.

    Pass/fail of formal verification is printed for B.5 baseline analysis
    but is not asserted — the Actor prompt may need tuning (B.5) before
    specs reliably verify.
    """
    candidate = await actor.run(
        requirement=requirement,
        language="C",
        standard="DO_178C",
    )

    # --- Assert: Actor must produce a spec ---
    assert candidate.dafny_spec.strip(), (
        f"[{name}] Actor produced an empty dafny_spec.\n"
        f"source_code:\n{candidate.source_code}\n"
        f"reasoning_trace:\n{candidate.reasoning_trace}"
    )

    result = await runner.verify(candidate.dafny_spec)

    # Print details for manual inspection / B.5 baseline.
    print(
        f"\n{'='*60}\n"
        f"[{name}] verified={result.verified}  "
        f"time={result.execution_time_seconds}s\n"
        f"--- dafny_spec ---\n{candidate.dafny_spec}\n"
        f"--- solver_output ---\n{result.solver_output}"
    )
    if result.failing_assertions:
        print("--- failing_assertions ---")
        for fa in result.failing_assertions:
            print(f"  {fa}")

    # --- Assert: spec must at least parse (no syntax errors) ---
    output_lower = result.solver_output.lower()
    parse_error = "parse error" in output_lower or "syntax error" in output_lower
    assert not parse_error, (
        f"[{name}] Dafny reported a parse/syntax error on the Actor-generated spec.\n"
        f"dafny_spec:\n{candidate.dafny_spec}\n"
        f"solver_output:\n{result.solver_output}"
    )
