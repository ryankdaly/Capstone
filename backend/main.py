import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.api.routers import audit, generation, health, pipeline
from backend.services.rag.knowledge_graph import ingest_standards
from backend.services.rag.retriever import StandardsRetriever

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        retriever = StandardsRetriever()
        chunk_count = ingest_standards(retriever)
        logger.info("Startup standards ingestion complete: %d chunks", chunk_count)
    except Exception as e:
        logger.exception("Failed to initialize ChromaDB on startup: %s", e)

    yield


app = FastAPI(
    title="HPEMA API",
    description="Hierarchical Policy-Enforced Multi-Agent pipeline for high-assurance code generation",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(generation.router, prefix="/api/v1/generation")
app.include_router(pipeline.router, prefix="/api/v1/pipeline")
app.include_router(audit.router, prefix="/api/v1/audit")
