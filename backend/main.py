from fastapi import FastAPI

from backend.api.routers import generation, health

app = FastAPI(
    title="HPEMA API",
)

app.include_router(health.router)
app.include_router(generation.router, prefix="/api/v1/generation")
