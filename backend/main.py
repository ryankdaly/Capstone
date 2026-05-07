import logging

from fastapi import FastAPI

from backend.api.routers import audit, generation, health, pipeline

logger = logging.getLogger(__name__)


# Standards ingestion happens lazily inside StandardsRetriever on first
# retrieve() call (see _maybe_auto_ingest), so no startup hook is required.
app = FastAPI(
    title="HPEMA API",
    description="Hierarchical Policy-Enforced Multi-Agent pipeline for high-assurance code generation",
    version="0.1.0",
)

app.include_router(health.router)
app.include_router(generation.router, prefix="/api/v1/generation")
app.include_router(pipeline.router, prefix="/api/v1/pipeline")
app.include_router(audit.router, prefix="/api/v1/audit")
