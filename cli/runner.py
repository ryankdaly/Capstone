"""In-process pipeline runner — drives the orchestrator directly from the CLI.

No FastAPI server needed. The CLI imports this, which creates the orchestrator
and streams events to the display manager.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

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
    config = load_config()
    registry = ModelRegistry(config)
    return PipelineOrchestrator(
        llm_client=LLMClient(registry),
        dafny_runner=DafnyRunner(),
        retriever=StandardsRetriever(),
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
        await orchestrator._llm_client.aclose()


def run_pipeline(
    requirement: str,
    standard: str,
    language: str,
    max_iterations: int,
    display: DisplayManager,
    stage: PipelineStage = PipelineStage.POLICY,
    run_tests: bool = True,
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
    )

    return asyncio.run(_run_async(request, display))
