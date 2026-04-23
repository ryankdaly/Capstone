"""FastAPI dependency injection — singleton services shared across requests."""

from __future__ import annotations

from functools import lru_cache

from backend.services.audit.logger import AuditLogger
from backend.services.llm.client import LLMClient
from backend.services.llm.model_registry import ModelRegistry
from backend.services.orchestrator import PipelineOrchestrator
from backend.services.rag.retriever import StandardsRetriever
from backend.services.verification.dafny_runner import DafnyRunner


@lru_cache
def get_model_registry() -> ModelRegistry:
    return ModelRegistry()


@lru_cache
def get_llm_client() -> LLMClient:
    return LLMClient(registry=get_model_registry())


@lru_cache
def get_dafny_runner() -> DafnyRunner:
    return DafnyRunner()


@lru_cache
def get_retriever() -> StandardsRetriever:
    return StandardsRetriever()


@lru_cache
def get_audit_logger() -> AuditLogger:
    return AuditLogger()


def get_orchestrator() -> PipelineOrchestrator:
    """Create an orchestrator with all dependencies wired."""
    return PipelineOrchestrator(
        llm_client=get_llm_client(),
        dafny_runner=get_dafny_runner(),
        retriever=get_retriever(),
        audit_logger=get_audit_logger(),
    )
