from __future__ import annotations

import re
import time
import uuid
from pathlib import Path

from backend.api.schemas.generation import GenerationRequest, GenerationResponse
from backend.core.errors import AppError
from backend.core.settings import settings

def _slugify(value: str, max_len: int = 48) -> str:
    """Create a filesystem-friendly slug for artifact filenames."""
    value = value.strip().lower()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^a-z0-9_\-]", "", value)
    value = value.strip("_-")
    return value[:max_len] or "req"


def _ext_for_language(target_language: str) -> str:
    if target_language == "C":
        return ".c"
    if target_language == "SPARK_Ada":
        return ".adb"
    return ".txt"

def run_actor_checker_pipeline(req: GenerationRequest, request_id: str | None = None) -> GenerationResponse:
    """Stub Actor-Checker orchestration.

    - Actor (stub): writes a language-specific 'code' file containing ONLY the requirement text.
    - Checker (stub): writes a 'policy/proof' txt file containing ONLY the selected safety standard.
    - Compliance (stub): deterministic decision (False) with a simple reason embedded in the proof text.

    Artifacts are written under settings.artifacts_dir (default: ./artifacts).
    """
    requirement = req.requirement_text.strip()
    if not requirement:
        raise AppError(code="empty_requirement", message="requirement_text must be non-empty", status_code=400)

    if len(requirement) > settings.max_requirement_chars:
        raise AppError(
            code="requirement_too_large",
            message="requirement_text exceeds configured limit",
            status_code=413,
            details={"max_chars": settings.max_requirement_chars},
        )

    run_id = request_id or str(uuid.uuid4())
    ts_ms = int(time.time() * 1000)
    slug = _slugify(requirement)

    artifacts_root = Path(settings.artifacts_dir)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    # ---- actor (stub) ----
    code_ext = _ext_for_language(req.target_language.value)
    code_path = artifacts_root / f"{ts_ms}_{run_id}_{slug}{code_ext}"
    code_contents = requirement + "\n"
    code_path.write_text(code_contents, encoding="utf-8")

    # return a small header in the API response so it's obvious what happened.
    generated_code = (
        f"// STUB ACTOR OUTPUT\n"
        f"// target={req.target_language.value} standard={req.safety_standard.value}\n"
        f"// artifact={code_path.as_posix()}\n\n"
        f"{code_contents}"
    )

    # ---- checker (stub) ----
    policy_path = artifacts_root / f"{ts_ms}_{run_id}_{slug}_policy.txt"
    policy_contents = req.safety_standard.value + "\n"
    policy_path.write_text(policy_contents, encoding="utf-8")

    formal_proof = (
        "STUB CHECKER OUTPUT\n"
        f"standard={req.safety_standard.value}\n"
        f"artifact={policy_path.as_posix()}\n\n"
        f"{policy_contents}"
    )

    # ---- compliance (stub) ----
    compliance_status = False

    return GenerationResponse(
        generated_code=generated_code,
        formal_proof=formal_proof,
        compliance_status=compliance_status,
        run_id=run_id,
        artifact_files=[code_path.as_posix(), policy_path.as_posix()],
    )
