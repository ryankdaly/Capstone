"""In-process pipeline runner — drives the orchestrator directly from the CLI.

No FastAPI server needed. The CLI imports this, which creates the orchestrator
and streams events to the display manager.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Callable
import httpx

from backend.api.schemas.pipeline import PipelineRequest, PipelineStage, PipelineState
from backend.config import load_config
from backend.services.audit.logger import AuditLogger
from backend.services.llm.client import LLMClient
from backend.services.llm.model_registry import ModelRegistry
from backend.services.orchestrator import PipelineOrchestrator
from backend.services.rag.retriever import StandardsRetriever
from backend.services.verification.dafny_runner import DafnyRunner
from backend.api.schemas.pipeline import stage_enabled
from backend.services.audit.model_run_logger import ModelRunLogger

from openai import (
    AsyncOpenAI,
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    BadRequestError,
    APITimeoutError,
    APIConnectionError,
)

if TYPE_CHECKING:
    from cli.display import DisplayManager



class LLMPreflightError(RuntimeError):
    """Raised when the configured LLM stack is not usable for the requested run."""


def _is_local_endpoint(endpoint: str) -> bool:
    return any(host in endpoint for host in ("localhost", "127.0.0.1", "0.0.0.0"))


def _active_roles_for_stage(stage: PipelineStage) -> list[str]:
    roles = ["actor"]
    if stage_enabled(PipelineStage.CHECKER, stage):
        roles.extend(["checker", "dafny_architect"])
    if stage_enabled(PipelineStage.POLICY, stage):
        roles.append("policy")
    return roles

def _mask_key(key: str) -> str:
    if not key:
        return "<empty>"
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"



async def _probe_local_health(endpoint: str) -> None:
    base = endpoint.rstrip("/").rsplit("/v1", 1)[0]
    health_url = f"{base}/health"
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(health_url)
    except Exception as exc:
        raise LLMPreflightError(
            f"Local model server is not reachable at {health_url}. "
            f"Start the server or update your configured endpoint.\n"
            f"Details: {exc}"
        ) from exc

    if response.status_code != 200:
        raise LLMPreflightError(
            f"Local model server at {health_url} is not ready "
            f"(HTTP {response.status_code})."
        )


async def _probe_model(role_label: str, endpoint: str, model: str, api_key: str) -> None:
    client = AsyncOpenAI(
        base_url=endpoint,
        api_key=api_key or "unused",
        timeout=20.0,
    )
    try:
        await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with OK."}],
            max_tokens=1,
            temperature=0.0,
        )

    except AuthenticationError as exc:
        masked = _mask_key(api_key)
        raise LLMPreflightError(
            f"{role_label}: API key was rejected by {endpoint}.\n"
            f"Key used: {masked}\n"
            f"Check your .env and configured api_key_env."
        ) from exc
    except PermissionDeniedError as exc:
        raise LLMPreflightError(
            f"{role_label}: permission denied for model '{model}' at {endpoint}."
        ) from exc
    except NotFoundError as exc:
        raise LLMPreflightError(
            f"{role_label}: model '{model}' was not found at {endpoint}."
        ) from exc
    except BadRequestError as exc:
        raise LLMPreflightError(
            f"{role_label}: endpoint accepted the connection but rejected the request. "
            f"This usually means the model name or endpoint config is wrong.\n"
            f"Details: {exc}"
        ) from exc
    except (APIConnectionError, APITimeoutError, TimeoutError) as exc:
        raise LLMPreflightError(
            f"{role_label}: could not reach {endpoint}. "
            f"Check network access, endpoint URL, and server availability.\n"
            f"Details: {exc}"
        ) from exc
    finally:
        await client.close()


async def ensure_llm_ready(stage: PipelineStage, mode: str = "build") -> None:
    """Fail fast if the configured LLM stack is not usable."""
    config = load_config()
    registry = ModelRegistry(config)

    if mode == "chat":
        roles = ["actor"]
    else:
        roles = _active_roles_for_stage(stage)

    checked: set[tuple[str, str, str]] = set()

    for role in roles:
        resolved = registry.get(role)
        role_label = role.replace("_", " ").title()

        if not resolved.endpoint.strip():
            raise LLMPreflightError(f"{role_label}: endpoint is empty in config.")

        if not resolved.model.strip():
            raise LLMPreflightError(f"{role_label}: model is empty in config.")

        if not _is_local_endpoint(resolved.endpoint) and not resolved.api_key.strip():
            env_name = getattr(getattr(config.models, role), "api_key_env", "API key env var")
            raise LLMPreflightError(
                f"{role_label}: missing API key. "
                f"Set {env_name} in your .env or shell before running HPEMA."
            )

        key = (resolved.endpoint, resolved.model, resolved.api_key)
        if key in checked:
            continue
        checked.add(key)

        if _is_local_endpoint(resolved.endpoint):
            await _probe_local_health(resolved.endpoint)

        await _probe_model(
            role_label=role_label,
            endpoint=resolved.endpoint,
            model=resolved.model,
            api_key=resolved.api_key,
        )




def _build_orchestrator() -> PipelineOrchestrator:
    """Wire up the orchestrator with all dependencies from config."""
    from backend.config import PROJECT_ROOT
    config = load_config()
    registry = ModelRegistry(config)
    retriever = StandardsRetriever(
        persist_dir=str(PROJECT_ROOT / config.policies.chromadb_dir),
        auto_ingest_path=str(PROJECT_ROOT / config.policies.standards_dir),
    )
    return PipelineOrchestrator(
        llm_client=LLMClient(registry),
        dafny_runner=DafnyRunner(),
        retriever=retriever,
        audit_logger=AuditLogger(),
        model_run_logger=ModelRunLogger(),
    )


async def _run_async(
    request: PipelineRequest,
    display: DisplayManager,
) -> PipelineState | None:
    await ensure_llm_ready(request.stage, mode="build")
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
    await ensure_llm_ready(PipelineStage.ACTOR, mode="chat")

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
