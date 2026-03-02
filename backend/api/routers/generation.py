from fastapi import APIRouter, Request

from backend.api.schemas.generation import GenerationRequest, GenerationResponse
from backend.services.generation_service import run_actor_checker_pipeline

router = APIRouter()


@router.post(
    "/generate",
    response_model=GenerationResponse,
    summary="Run the high-assurance generation pipeline",
    tags=["generation"],
)
async def generate(payload: GenerationRequest, request: Request) -> GenerationResponse:
    """Entry point for the Actor-Checker generation pipeline.

    Current behavior:
      - Validates inputs
      - Runs a placeholder Actor-Checker pipeline (deterministic stub artifacts)

    TODO:
      - Implement real Actor (code synthesis), Checker (proof/tests), and policy engine
      - Persist full audit log + artifacts for traceability
    """
    return run_actor_checker_pipeline(payload, request_id=getattr(request.state, 'request_id', None))


@router.get(
    "/capabilities",
    summary="List supported standards and target languages",
    tags=["generation"],
)
async def capabilities():
    from backend.api.schemas.generation import SafetyStandard, TargetLanguage

    return {
        "safety_standards": [s.value for s in SafetyStandard],
        "target_languages": [l.value for l in TargetLanguage],
    }
