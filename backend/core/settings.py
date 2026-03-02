import os
from typing import List, Optional

def _split_csv(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]

class Settings:
    """
    Minimal settings object to avoid extra dependencies.

    Configure via environment variables:

      - HPEMA_ENV: "dev" or "prod" (controls error detail exposure).

      - HPEMA_CORS_ALLOW_ORIGINS: comma-separated list of allowed origins.
        Example: "http://localhost:3000,http://127.0.0.1:3000"

      - HPEMA_MAX_REQUIREMENT_CHARS: maximum characters allowed in requirement_text.

      - HPEMA_ARTIFACTS_DIR: directory where generated artifacts are written.
        Default: "artifacts" (relative to the working directory).

      - HPEMA_DATABASE_URL: Postgres connection string, e.g.
        "postgresql://user:pass@localhost:5432/dbname"

      - HPEMA_READINESS_REQUIRES_DB: "1" (default) or "0".
        If enabled and HPEMA_DATABASE_URL is set, /health/ready will fail if DB is unreachable.
    """
    def __init__(self) -> None:
        self.env = os.getenv("HPEMA_ENV", "dev").lower()

        cors_origins = os.getenv(
            "HPEMA_CORS_ALLOW_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000",
        )
        self.cors_allow_origins: List[str] = _split_csv(cors_origins)

        self.max_requirement_chars = int(os.getenv("HPEMA_MAX_REQUIREMENT_CHARS", "200000"))

        self.artifacts_dir = os.getenv("HPEMA_ARTIFACTS_DIR", "artifacts")

        self.database_url: Optional[str] = os.getenv("HPEMA_DATABASE_URL")
        self.readiness_requires_db: bool = os.getenv("HPEMA_READINESS_REQUIRES_DB", "1") not in ("0", "false", "False")

settings = Settings()
