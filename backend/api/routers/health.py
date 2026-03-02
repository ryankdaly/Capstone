from fastapi import APIRouter
from backend.core.errors import AppError
from backend.core.settings import settings

router = APIRouter(tags=["health"])

def _check_postgres() -> dict:
    """
    Best-effort Postgres readiness check.

    - If no DB URL is configured: returns {"db": "not_configured"}.
    - If DB URL configured:
        - returns {"db": "ok"} if a simple SELECT 1 succeeds
        - raises AppError if required and unreachable / driver missing
    """
    if not settings.database_url:
        return {"db": "not_configured"}

    # Try psycopg (v3) first, then psycopg2.
    try:
        import psycopg  # type: ignore
        try:
            with psycopg.connect(settings.database_url, connect_timeout=2) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                    cur.fetchone()
            return {"db": "ok", "driver": "psycopg"}
        except Exception as e:
            raise AppError(
                code="db_unreachable",
                message="Postgres is configured but not reachable",
                status_code=503,
                details={"error": type(e).__name__},
            )
    except ModuleNotFoundError:
        pass

    try:
        import psycopg2  # type: ignore
        try:
            conn = psycopg2.connect(settings.database_url, connect_timeout=2)
            try:
                cur = conn.cursor()
                cur.execute("SELECT 1;")
                cur.fetchone()
                cur.close()
            finally:
                conn.close()
            return {"db": "ok", "driver": "psycopg2"}
        except Exception as e:
            raise AppError(
                code="db_unreachable",
                message="Postgres is configured but not reachable",
                status_code=503,
                details={"error": type(e).__name__},
            )
    except ModuleNotFoundError:
        # DB is configured but no driver installed.
        raise AppError(
            code="db_driver_missing",
            message="Postgres is configured but no driver is installed (install psycopg or psycopg2)",
            status_code=503,
            details={"expected": ["psycopg", "psycopg2"]},
        )

@router.get("/health", summary="Legacy health endpoint")
async def health_check():
    return {"status": "ok"}

@router.get("/health/live", summary="Liveness probe")
async def health_live():
    return {"status": "ok"}

@router.get("/health/ready", summary="Readiness probe")
async def health_ready():
    details = {}

    if settings.database_url and settings.readiness_requires_db:
        details.update(_check_postgres())
    elif settings.database_url and not settings.readiness_requires_db:
        # Non-fatal: report status but don't fail readiness.
        try:
            details.update(_check_postgres())
        except AppError as e:
            details.update({"db": "unreachable", "db_error": e.code})

    return {"status": "ok", "details": details}
