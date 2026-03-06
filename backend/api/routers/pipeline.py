"""Pipeline router — SSE streaming endpoint for the full generation pipeline."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from backend.api.dependencies import get_orchestrator
from backend.api.schemas.pipeline import PipelineRequest, StreamEvent
from backend.services.orchestrator import PipelineOrchestrator

router = APIRouter()


@router.post(
    "/run",
    summary="Start the HPEMA generation pipeline (SSE stream)",
    tags=["pipeline"],
)
async def run_pipeline(
    request: PipelineRequest,
    orchestrator: PipelineOrchestrator = Depends(get_orchestrator),
):
    """Start a full pipeline run and stream events via Server-Sent Events.

    The CLI and dashboard consume this SSE stream to display real-time
    agent reasoning panels (which agent is active, chain-of-thought,
    pass/fail status).
    """

    async def event_stream():
        async for event in orchestrator.run(request):
            data = event.model_dump_json()
            yield f"event: {event.event_type.value}\ndata: {data}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
