from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.api.routers import generation, health
from backend.core.settings import settings
from backend.core.errors import (
    AppError,
    app_error_handler,
    http_error_handler,
    validation_error_handler,
    unhandled_error_handler,
)

app = FastAPI(
    title="HPEMA API",
)

@app.middleware("http")
async def add_request_id_header(request: Request, call_next):
    rid = request.headers.get("x-request-id")
    if not rid:
        # Delay import to keep startup light
        import uuid
        rid = str(uuid.uuid4())

    # Make it available to handlers via request.state
    request.state.request_id = rid

    response = await call_next(request)
    response.headers["x-request-id"] = rid
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(AppError, app_error_handler)
app.add_exception_handler(StarletteHTTPException, http_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)
app.add_exception_handler(Exception, unhandled_error_handler)

app.include_router(health.router)
app.include_router(health.router, prefix="/api/v1")

app.include_router(generation.router, prefix="/api/v1/generation")
