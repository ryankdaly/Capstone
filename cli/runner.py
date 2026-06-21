"""In-process pipeline runner — drives the orchestrator directly from the CLI.

No FastAPI server needed. The CLI imports this, which creates the orchestrator
and streams events to the display manager.

Selective runners (run_checker_only, run_dafny_only, run_policy_only) bypass
the full orchestrator loop and instantiate only the required agent(s), emitting
the same StreamEvents that DisplayManager already knows how to render.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Callable
from uuid import uuid4

from backend.api.schemas.pipeline import PipelineRequest, PipelineStage, PipelineState, StreamEvent, StreamEventType
from backend.config import load_config, resolve_data_path
from backend.services.audit.logger import AuditLogger
from backend.services.llm.client import LLMClient
from backend.services.llm.model_registry import ModelRegistry
from backend.services.orchestrator import PipelineOrchestrator
from backend.services.rag.retriever import StandardsRetriever
from backend.services.verification.dafny_runner import DafnyRunner

if TYPE_CHECKING:
    from cli.display import DisplayManager


def _build_orchestrator() -> PipelineOrchestrator:
    """Wire up the orchestrator with all dependencies from config."""
    from backend.config import resolve_data_path
    config = load_config()
    registry = ModelRegistry(config)
    retriever = StandardsRetriever(
        persist_dir=str(resolve_data_path(config.policies.chromadb_dir, "chromadb")),
        auto_ingest_path=str(resolve_data_path(config.policies.standards_dir, "standards")),
    )
    return PipelineOrchestrator(
        llm_client=LLMClient(registry),
        dafny_runner=DafnyRunner(),
        retriever=retriever,
        audit_logger=AuditLogger(),
    )


async def _run_async(
    request: PipelineRequest,
    display: DisplayManager,
) -> PipelineState | None:
    orchestrator = _build_orchestrator()
    # display.skip_controller satisfies the SkipSignal Protocol the orchestrator
    # depends on. Passing it lets Ctrl+S short-circuit the current agent.
    skip_signal = getattr(display, "skip_controller", None)
    try:
        async for event in orchestrator.run(request, skip_signal=skip_signal):
            display.handle_event(event)
        return orchestrator.last_state
    finally:
        # Close httpx connection pools while the event loop is still alive.
        # Without this, Python 3.10+ logs "Event loop is closed" when the GC
        # finalizes AsyncOpenAI clients after asyncio.run() shuts down the loop.
        await orchestrator._llm.aclose()


_CHAT_TIMEOUT = 90  # seconds — fail fast rather than hanging

_CHAT_SYSTEM_PROMPT = (
    "You are HPEMA's Actor agent — a knowledgeable assistant for "
    "high-assurance and safety-critical software engineering. "
    "Answer questions helpfully and concisely. Only generate code "
    "if explicitly asked."
)


async def chat_stream(message: str, on_token: Callable[[str], None]) -> str:
    """Stream a chat response, calling on_token for each chunk.

    Returns the full accumulated response text.  Falls back to a single
    blocking call + one on_token invocation if streaming is unsupported.
    Applies a per-chunk timeout guard: if no token arrives within
    _CHAT_TIMEOUT seconds, raises TimeoutError.
    """
    config = load_config()
    registry = ModelRegistry(config)
    client = LLMClient(registry)

    chunks: list[str] = []
    try:
        async for token in client.generate_stream(
            role="actor",
            system_prompt=_CHAT_SYSTEM_PROMPT,
            user_prompt=message,
            temperature=0.4,
            max_tokens=2048,
        ):
            chunks.append(token)
            on_token(token)
    finally:
        await client.aclose()
    return "".join(chunks)


# Keep old blocking entry point for callers that don't need streaming.
def chat_with_actor(message: str, history: list | None = None) -> str:
    """Blocking non-streaming chat — kept for backwards compatibility."""
    chunks: list[str] = []

    async def _run() -> str:
        return await chat_stream(message, on_token=chunks.append)

    return asyncio.run(_run())


def run_pipeline(
    requirement: str,
    standard: str,
    language: str,
    max_iterations: int,
    display: DisplayManager,
    stage: PipelineStage = PipelineStage.POLICY,
    run_tests: bool = True,
    retry_context: str = "",
) -> PipelineState | None:
    """Run the full pipeline in-process. Blocking call.

    This is the main entry point for the CLI — no FastAPI needed.
    """
    request = PipelineRequest(
        requirement_text=requirement,
        safety_standard=standard,
        target_language=language,
        max_iterations=max_iterations,
        stage=stage,
        run_tests=run_tests,
        retry_context=retry_context,
    )

    return asyncio.run(_run_async(request, display))


# ---------------------------------------------------------------------------
# Selective runners — single-agent execution on existing code
# ---------------------------------------------------------------------------

def _make_event(event_type: StreamEventType, agent: str | None, data: dict | None = None) -> StreamEvent:
    return StreamEvent(
        event_type=event_type,
        run_id=uuid4(),
        agent=agent,
        data=data or {},
    )


def _build_llm_client() -> LLMClient:
    config = load_config()
    registry = ModelRegistry(config)
    return LLMClient(registry)


async def _run_checker_only_async(
    code: str,
    standard: str,
    language: str,
    display: DisplayManager,
    run_tests: bool = True,
) -> None:
    """Run only the checker (+ test runner) on existing code."""
    from backend.services.agents.checker import CheckerAgent
    from backend.services.testing.runner import TestRunner

    client = _build_llm_client()
    checker = CheckerAgent(client)
    test_runner = TestRunner()

    checker_report = None
    try:
        display.handle_event(_make_event(StreamEventType.AGENT_START, "checker"))
        async for item in checker.run_streaming(
            source_code=code,
            language=language,
            standard=standard,
        ):
            if isinstance(item, str):
                display.handle_event(_make_event(StreamEventType.AGENT_TOKEN, "checker", {"token": item}))
            else:
                checker_report = item

        if checker_report is not None:
            display.handle_event(_make_event(
                StreamEventType.AGENT_OUTPUT, "checker",
                {
                    "verdict": checker_report.verdict.value,
                    "issues": len(checker_report.issues),
                    "issues_detail": [iss.model_dump() for iss in checker_report.issues],
                    "test_cases": checker_report.test_cases,
                    "reasoning_trace": checker_report.reasoning_trace,
                },
            ))
    except Exception as e:
        display.handle_event(_make_event(
            StreamEventType.AGENT_ERROR, "checker", {"error": str(e)}
        ))
        await client.aclose()
        return

    # Test runner
    is_python = language.lower() == "python"
    if run_tests and is_python and checker_report is not None and checker_report.test_cases:
        try:
            display.handle_event(_make_event(StreamEventType.AGENT_START, "test_runner"))
            test_result = await test_runner.run(
                source_code=code,
                test_cases=checker_report.test_cases,
            )
            display.handle_event(_make_event(
                StreamEventType.TEST_RUN, "test_runner",
                {
                    "executed": test_result.executed,
                    "total": test_result.total,
                    "passed": test_result.passed,
                    "failed": test_result.failed,
                    "errors": test_result.errors,
                    "test_results": [t.model_dump() for t in test_result.test_results],
                    "pytest_output": test_result.pytest_output,
                },
            ))
        except Exception as e:
            display.handle_event(_make_event(
                StreamEventType.AGENT_ERROR, "test_runner", {"error": str(e)}
            ))

    await client.aclose()


async def _run_dafny_only_async(
    code: str,
    language: str,
    display: DisplayManager,
) -> None:
    """Run only the Dafny architect + verifier on existing code."""
    from backend.services.agents.dafny_architect import DafnyArchitectAgent

    client = _build_llm_client()
    architect = DafnyArchitectAgent(client)
    dafny_runner = DafnyRunner()

    dafny_result = None
    try:
        display.handle_event(_make_event(StreamEventType.AGENT_START, "dafny_architect"))
        async for item in architect.run_streaming(
            source_code=code,
            requirement="Re-verify the existing implementation.",
            language=language,
        ):
            if isinstance(item, str):
                display.handle_event(_make_event(StreamEventType.AGENT_TOKEN, "dafny_architect", {"token": item}))
            else:
                dafny_result = item

        if dafny_result is not None:
            display.handle_event(_make_event(
                StreamEventType.AGENT_OUTPUT, "dafny_architect",
                {
                    "dafny_spec": dafny_result.dafny_source,
                    "reasoning_trace": dafny_result.reasoning_trace,
                    "cycle": 1,
                },
            ))
    except Exception as e:
        display.handle_event(_make_event(
            StreamEventType.AGENT_ERROR, "dafny_architect", {"error": str(e)}
        ))
        await client.aclose()
        return

    if dafny_result is not None and dafny_result.dafny_source.strip():
        try:
            display.handle_event(_make_event(StreamEventType.AGENT_START, "dafny_verifier"))
            vr = await dafny_runner.verify(dafny_result.dafny_source)
            display.handle_event(_make_event(
                StreamEventType.AGENT_OUTPUT, "dafny_verifier",
                {
                    "verified": vr.verified,
                    "solver_output": vr.solver_output,
                    "failing_assertions": vr.failing_assertions,
                    "execution_time_seconds": vr.execution_time_seconds,
                    "cycle": 1,
                },
            ))
        except Exception as e:
            display.handle_event(_make_event(
                StreamEventType.AGENT_ERROR, "dafny_verifier", {"error": str(e)}
            ))

    await client.aclose()


async def _run_policy_only_async(
    code: str,
    standard: str,
    display: DisplayManager,
) -> None:
    """Run only the policy agent on existing code."""
    from backend.services.agents.policy import PolicyAgent

    client = _build_llm_client()
    policy = PolicyAgent(client)
    config = load_config()
    retriever = StandardsRetriever(
        persist_dir=str(resolve_data_path(config.policies.chromadb_dir, "chromadb")),
        auto_ingest_path=str(resolve_data_path(config.policies.standards_dir, "standards")),
    )

    try:
        display.handle_event(_make_event(StreamEventType.AGENT_START, "policy"))

        try:
            policy_context = await retriever.retrieve(query=standard, standard=standard)
        except Exception:
            policy_context = ""

        policy_verdict = None
        async for item in policy.run_streaming(
            source_code=code,
            dafny_spec="",
            standard=standard,
            checker_report=None,
            verification_result=None,
            policy_context=policy_context,
        ):
            if isinstance(item, str):
                display.handle_event(_make_event(StreamEventType.AGENT_TOKEN, "policy", {"token": item}))
            else:
                policy_verdict = item

        if policy_verdict is not None:
            display.handle_event(_make_event(
                StreamEventType.AGENT_OUTPUT, "policy",
                {
                    "compliant": policy_verdict.compliant,
                    "risk_level": policy_verdict.risk_level.value,
                    "violations": [v.model_dump() for v in policy_verdict.violations],
                    "recommendations": policy_verdict.recommendations,
                    "reasoning_trace": policy_verdict.reasoning_trace,
                },
            ))
    except Exception as e:
        display.handle_event(_make_event(
            StreamEventType.AGENT_ERROR, "policy", {"error": str(e)}
        ))

    await client.aclose()


# Blocking wrappers for use from the synchronous REPL loop

def run_checker_only(
    code: str,
    standard: str,
    language: str,
    display: DisplayManager,
    run_tests: bool = True,
) -> None:
    """Run checker (+ tests) on *code*. Blocking."""
    asyncio.run(_run_checker_only_async(code, standard, language, display, run_tests))


def run_dafny_only(
    code: str,
    language: str,
    display: DisplayManager,
) -> None:
    """Run Dafny architect + verifier on *code*. Blocking."""
    asyncio.run(_run_dafny_only_async(code, language, display))


def run_policy_only(
    code: str,
    standard: str,
    display: DisplayManager,
) -> None:
    """Run policy agent on *code*. Blocking."""
    asyncio.run(_run_policy_only_async(code, standard, display))
