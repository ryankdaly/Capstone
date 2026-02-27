from fastapi import FastAPI

from backend.api.routers import health

app = FastAPI(
    title="HPEMA API",
)

app.include_router(health.router)
