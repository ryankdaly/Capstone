import subprocess

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@router.get("/", response_class=HTMLResponse)
async def root():
    try:
        result = subprocess.run(
            ["dafny", "--version"], capture_output=True, text=True, timeout=5
        )
        dafny_version = result.stdout.strip() or result.stderr.strip()
        dafny_status = "online"
        dafny_color = "#22c55e"
    except Exception as e:
        dafny_version = str(e)
        dafny_status = "unavailable"
        dafny_color = "#ef4444"

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>HPEMA</title>
    <style>
        body {{ font-family: monospace; background: #0f172a; color: #e2e8f0; max-width: 600px; margin: 80px auto; padding: 0 20px; }}
        h1 {{ color: #7dd3fc; margin-bottom: 4px; }}
        .subtitle {{ color: #64748b; margin-bottom: 40px; }}
        .card {{ background: #1e293b; border-radius: 8px; padding: 20px 24px; margin-bottom: 16px; }}
        .label {{ color: #94a3b8; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }}
        .value {{ font-size: 15px; margin-top: 4px; }}
        .badge {{ display: inline-block; padding: 2px 10px; border-radius: 99px; font-size: 12px; font-weight: bold; color: #0f172a; background: {dafny_color}; }}
        a {{ color: #7dd3fc; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <h1>HPEMA</h1>
    <p class="subtitle">Hierarchical Policy-Enforced Multi-Agent Pipeline</p>

    <div class="card">
        <div class="label">API Status</div>
        <div class="value"><span class="badge" style="background:#22c55e">online</span></div>
    </div>

    <div class="card">
        <div class="label">Dafny Verifier</div>
        <div class="value"><span class="badge">{dafny_status}</span></div>
        <div class="value" style="margin-top:8px; color:#94a3b8; font-size:13px">{dafny_version}</div>
    </div>

    <div class="card">
        <div class="label">Endpoints</div>
        <div class="value">
            <a href="/docs">Interactive API docs</a><br>
            <a href="/health">GET /health</a><br>
            <a href="/api/v1/generation/generate">POST /api/v1/generation/generate</a><br>
            <a href="/api/v1/pipeline/run">POST /api/v1/pipeline/run</a>
        </div>
    </div>
</body>
</html>
"""
