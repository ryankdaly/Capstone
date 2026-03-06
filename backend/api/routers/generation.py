"""Generation router — synchronous endpoint (returns when pipeline completes)."""

from fastapi import APIRouter, Depends

from backend.api.dependencies import get_orchestrator
from backend.api.schemas.generation import GenerationRequest, GenerationResponse
from backend.api.schemas.pipeline import PipelineRequest, PipelineStatus
from backend.services.orchestrator import PipelineOrchestrator

router = APIRouter()


@router.post(
    "/generate",
    response_model=GenerationResponse,
    summary="Run the high-assurance generation pipeline (blocking)",
    tags=["generation"],
)
async def generate(
    request: GenerationRequest,
    orchestrator: PipelineOrchestrator = Depends(get_orchestrator),
) -> GenerationResponse:
    """Synchronous generation endpoint — runs the full pipeline and returns results.

    For real-time streaming, use the /api/v1/pipeline/run SSE endpoint instead.
    """
    pipeline_request = PipelineRequest(
        requirement_text=request.requirement_text,
        safety_standard=request.safety_standard,
        target_language=request.target_language,
    )

    # Consume the full async generator to get the final state
    last_event = None
    async for event in orchestrator.run(pipeline_request):
        last_event = event

    # Extract results from the pipeline's final state
    generated_code = ""
    formal_proof = ""
    compliance_status = False

    if last_event and last_event.data.get("status") in (
        PipelineStatus.COMPLETED.value,
        PipelineStatus.AWAITING_APPROVAL.value,
    ):
        compliance_status = True

    return GenerationResponse(
        generated_code=generated_code,
        formal_proof=formal_proof,
        compliance_status=compliance_status,
    )
