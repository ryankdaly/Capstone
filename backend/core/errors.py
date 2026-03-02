from __future__ import annotations

from typing import Any, Dict, Optional
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import uuid
import traceback

class AppError(Exception):
    """An application-level error with a stable error code and HTTP status.

    Use this for expected failures (bad input, policy violation, model unavailable, etc.).
    """
    def __init__(
        self,
        *,
        code: str,
        message: str,
        status_code: int = 400,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}

def _request_id(request: Request) -> str:
    rid = getattr(getattr(request, "state", None), "request_id", None)
    if rid:
        return rid
    rid = request.headers.get("x-request-id")
    return rid or str(uuid.uuid4())

def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    rid = _request_id(request)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "request_id": rid,
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            },
        },
    )

def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    rid = _request_id(request)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "request_id": rid,
            "error": {
                "code": "http_error",
                "message": str(exc.detail),
                "details": {"status_code": exc.status_code},
            },
        },
    )

def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    rid = _request_id(request)
    return JSONResponse(
        status_code=422,
        content={
            "request_id": rid,
            "error": {
                "code": "validation_error",
                "message": "Request validation failed",
                "details": {"errors": exc.errors()},
            },
        },
    )

def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    from backend.core.settings import settings

    rid = _request_id(request)
    payload: Dict[str, Any] = {
        "request_id": rid,
        "error": {
            "code": "internal_error",
            "message": "Internal server error",
            "details": {},
        },
    }
    if settings.env == "dev":
        payload["error"]["details"] = {
            "exception": type(exc).__name__,
            "traceback": traceback.format_exc().splitlines()[-20:],  # cap size
        }

    return JSONResponse(status_code=500, content=payload)
