from enum import Enum

from pydantic import BaseModel


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

    requirement_text: str
    safety_standard: SafetyStandard
    target_language: TargetLanguage


class GenerationResponse(BaseModel):
    """Output contract for generated code and its associated verification artifacts."""

    generated_code: str
    formal_proof: str
    compliance_status: bool

