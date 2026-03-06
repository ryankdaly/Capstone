from fastapi import FastAPI

from backend.api.routers import audit, generation, health, pipeline

app = FastAPI(
    title="HPEMA API",
    description="Hierarchical Policy-Enforced Multi-Agent pipeline for high-assurance code generation",
    version="0.1.0",
)

app.include_router(health.router)
app.include_router(generation.router, prefix="/api/v1/generation")
app.include_router(pipeline.router, prefix="/api/v1/pipeline")
app.include_router(audit.router, prefix="/api/v1/audit")
