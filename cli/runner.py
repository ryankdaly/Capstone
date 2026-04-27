"""In-process pipeline runner — drives the orchestrator directly from the CLI.

No FastAPI server needed. The CLI imports this, which creates the orchestrator
and streams events to the display manager.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from backend.api.schemas.pipeline import PipelineRequest, PipelineStage, PipelineState
from backend.config import load_config
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
    try:
        async for event in orchestrator.run(request):
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
    except Exception:
        await client.aclose()
        raise

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
