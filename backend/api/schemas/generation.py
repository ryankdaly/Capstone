from enum import Enum

from pydantic import BaseModel, Field


class SafetyStandard(str, Enum):
    """Supported safety and assurance standards for generation."""

    NASA = "NASA"
    BOEING_SDP = "Boeing_SDP"
    DO_178C = "DO_178C"


class TargetLanguage(str, Enum):
    """Supported target implementation languages for generated artifacts."""

    C = "C"
    SPARK_ADA = "SPARK_Ada"


class GenerationRequest(BaseModel):
    """Input contract for the high-assurance generation pipeline."""

    requirement_text: str = Field(..., min_length=1, description="High-level requirement/specification text")
    safety_standard: SafetyStandard = Field(..., description="Safety/compliance standard to enforce")
    target_language: TargetLanguage = Field(..., description="Target implementation language")


class GenerationResponse(BaseModel):
    """Output contract for generated code and its associated verification artifacts."""

    generated_code: str
    formal_proof: str
    compliance_status: bool
    run_id: str
    artifact_files: list[str]

