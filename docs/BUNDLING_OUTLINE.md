# HPEMA — Bundling & Release Guide

> **Audience:** Developer (+ LLM assistant) implementing the packaging pipeline from scratch.  
> **Goal:** Anyone runs `pip install hpema` → types `hpema` → it works. Full stop.  
> **Out of scope:** This document describes *what to build*, not *product features*. Do not add features while implementing.

---

## 0. How the Codebase Looks Right Now

Before touching anything, understand what exists:

| Path | Role |
|------|------|
| `cli/main.py` | Entry point. `app = typer.Typer(name="hpema")`. The `hpema` binary should call this. |
| `cli/repl.py` | Interactive REPL loop |
| `cli/runner.py` | In-process pipeline runner (no FastAPI needed for CLI) |
| `backend/` | All agents, orchestrator, LLM client, Dafny runner, RAG, audit |
| `backend/main.py` | FastAPI app — only for the API server mode, NOT used by `python -m cli.main` |
| `data/standards/` | RAG source docs (DO_178C, MISRA_C, NASA, Boeing_SDP). Must ship with the package. |
| `backend/services/llm/prompts/` | `.txt` prompt files. Must ship with the package. |
| `requirements.txt` | Flat dep list mixing core, optional RAG, dev, and dashboard deps. Needs splitting. |
| `hpema_config.yaml` | Default config. Must ship with the package. |
| `local/installs/dafny/` | Pre-downloaded Dafny binaries (Linux x64). Already on disk. |
| `Dockerfile` | Multi-stage build — installs Dafny via .NET SDK. Good reference for the Dafny step. |

**No `pyproject.toml` or `setup.py` exists yet.** That is the first thing to create.

---

## 1. Chosen Strategy

### Distribution: PyPI + GitHub Releases

| Channel | What goes there | Who uses it |
|---------|----------------|-------------|
| **PyPI** | `pip install hpema` — Python wheel, no Dafny | Most users; Dafny downloaded by `hpema setup` on first use |
| **GitHub Releases** | `.tar.gz` per platform with Dafny embedded | Offline/air-gapped installs; also attached to every tagged release automatically |

**Why not npm?** HPEMA is a Python tool. npm adds Node.js as a runtime dep for no benefit. PyPI is the right distribution channel.

**Why separate bundle from release?**  
`scripts/bundle.sh` produces artifacts locally (`.whl`, platform tarballs). You test those artifacts. Then `scripts/release.sh` tags git and uploads them. You never release untested artifacts.

---

## 2. Dependency Audit — What to Keep, What to Separate

Current `requirements.txt` mixes four categories. Split them before writing `pyproject.toml`.

### 2a. Core runtime (always required by `hpema` CLI)

```
openai>=1.0
httpx>=0.27.0
pydantic>=2.0
pyyaml>=6.0
typer>=0.12.0
rich>=13.0
prompt_toolkit>=3.0
```

These are the only packages unconditionally installed by `pip install hpema`.

### 2b. RAG / policy deps (core — always required)

```
torch==2.2.2
numpy<2.0
sentence-transformers==2.5.0
transformers==4.37.2
chromadb>=0.4.0
```

These ship as **core dependencies**, not optional extras. The policy layer is not optional — it is a first-class pipeline stage and must work out of the box after `pip install hpema`. Do not gate these behind `[rag]` or any other extra.

**Note on torch:** `torch==2.2.2` is a hard pin. Consider relaxing to `torch>=2.2,<3` for users on newer hardware, but coordinate with the team first — changing a pin is a breaking change risk.

### 2c. API server extras (only if running `backend/main.py`)

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
```

Install with: `pip install hpema[server]`

These are NOT needed for `python -m cli.main` which runs in-process with no HTTP server.

### 2d. Dev / CI only (never shipped to users)

```
pytest>=8.0
pytest-asyncio>=0.23.0
anyio>=4.0
streamlit>=1.30.0
```

Put these in `requirements-dev.txt` only. Do not appear in `pyproject.toml` at all (or put under `[project.optional-dependencies] dev = [...]`).

### Verification task

Before writing `pyproject.toml`, grep for every import in `cli/` and `backend/` and verify each maps to one of the categories above. Specifically check:

```bash
grep -rh "^import\|^from" cli/ backend/ \
  | grep -v "^from __future__\|^from typing\|^from pathlib\|^import os\|^import sys\|^import re\|^import json\|^import time\|^import uuid\|^import asyncio\|^import logging\|^import subprocess\|^import hashlib\|^import abc\|^import enum\|^import dataclasses\|^import threading" \
  | sort -u
```

Any third-party package not in the lists above must be categorized before continuing.

---

## 3. Create `pyproject.toml`

Create this file at the project root. It replaces `requirements.txt` as the canonical dependency declaration for the package.

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "hpema"
version = "0.1.0"
description = "Hierarchical Policy-Enforced Multi-Agent code generation pipeline for safety-critical software"
readme = "README.md"
license = { file = "LICENSE" }
requires-python = ">=3.11"
dependencies = [
    "openai>=1.0",
    "httpx>=0.27.0",
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "typer>=0.12.0",
    "rich>=13.0",
    "prompt_toolkit>=3.0",
    # Policy layer — always required
    "torch==2.2.2",
    "numpy<2.0",
    "sentence-transformers==2.5.0",
    "transformers==4.37.2",
    "chromadb>=0.4.0",
]

[project.optional-dependencies]
server = [
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.29.0",
]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23.0",
    "anyio>=4.0",
    "hpema[server]",
]

[project.scripts]
hpema = "cli.main:app"

[tool.hatch.build.targets.wheel]
packages = ["cli", "backend"]

[tool.hatch.build.targets.wheel.force-include]
"data/standards" = "hpema_data/standards"
"backend/services/llm/prompts" = "backend/services/llm/prompts"
"hpema_config.yaml" = "hpema_config.yaml"
```

### Why hatchling?

It is the default build backend for modern Python, zero-config for simple layouts, and `pip install hatchling` is the only build dep. Alternatives: `setuptools` (more config), `flit` (simpler but less flexible). Either works. Pick one and commit.

### The `[project.scripts]` entry

This line is what creates the `hpema` binary on the user's PATH. Hatchling generates a thin wrapper script that calls `cli.main:app`. This replaces `python -m cli.main`.

**Verify that `cli/main.py` has this at the bottom:**

```python
if __name__ == "__main__":
    app()
```

It does. The `app()` call must be reachable both from `__main__` (for `python -m cli.main`) and from the entry point wrapper (for `hpema`). No changes needed.

### Bundling `data/standards/`

The `force-include` block above copies `data/standards/` into the wheel as `hpema_data/standards/`. After install, these files live inside the package, not in `data/standards/` relative to cwd.

**Consequence:** `backend/config.py` currently resolves `standards_dir` relative to `PROJECT_ROOT` (the git checkout root). After packaging, `PROJECT_ROOT` points into the installed package directory, not a user's checkout. You must update the path resolution in `backend/config.py`:

```python
# In load_config() or wherever standards_dir is resolved to an absolute path:
import importlib.resources

def _get_bundled_standards_dir() -> Path:
    """Return path to bundled standards, whether installed or running from source."""
    # Try installed location first
    try:
        ref = importlib.resources.files("hpema_data") / "standards"
        return Path(str(ref))
    except (ModuleNotFoundError, TypeError):
        pass
    # Fall back to source checkout
    return PROJECT_ROOT / "data" / "standards"
```

Then in `HpemaConfig` or wherever `standards_dir` is resolved, call `_get_bundled_standards_dir()` when the config value is the default `"data/standards"`. If the user has overridden it to an absolute path, use that as-is.

The same applies to `chromadb_dir` (writable, so it should live in a user data dir — see section 6).

### Bundling prompt `.txt` files

The `[tool.hatch.build.targets.wheel.force-include]` entry above includes prompts as package data. Access them using `importlib.resources` similarly, or simply rely on the fact that `backend/services/llm/prompts/` is already inside the `backend` package that hatchling includes.

**Check:** `backend/services/llm/prompts/` currently loaded via `Path(__file__).parent / "prompts" / "actor.txt"` (or similar). That pattern works correctly in both source and installed contexts as long as the prompts directory is included in the wheel. Verify the current loading pattern in `backend/services/agents/` before assuming.

---

## 4. Dafny Bundling Strategy

Dafny is a .NET tool (~50 MB) + Z3 solver. It cannot go into the PyPI wheel (wheel size limits + platform specificity). Two approaches, both are implemented:

### Approach A: `hpema setup` command (downloads Dafny on first use)

Add a `setup` subcommand to `cli/main.py`:

```python
@app.command()
def setup() -> None:
    """Download and install Dafny for the current platform."""
    from cli.setup_dafny import install_dafny
    install_dafny()
```

Create `cli/setup_dafny.py`:

```python
"""Downloads the correct Dafny release for the current platform and installs it
into ~/.hpema/dafny/. Updates the active config to point binary_path there."""

import os
import platform
import shutil
import sys
import zipfile
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import Progress

DAFNY_VERSION = "4.9.0"
DAFNY_RELEASES = "https://github.com/dafny-lang/dafny/releases/download"

_PLATFORM_MAP = {
    ("Linux",  "x86_64"):  f"dafny-{DAFNY_VERSION}-x64-ubuntu-20.04.zip",
    ("Darwin", "arm64"):   f"dafny-{DAFNY_VERSION}-x64-osx-11.zip",   # Dafny has no arm64 build; Rosetta runs x64
    ("Darwin", "x86_64"):  f"dafny-{DAFNY_VERSION}-x64-osx-11.zip",
    ("Windows", "AMD64"):  f"dafny-{DAFNY_VERSION}-x64-win.zip",
}

HPEMA_HOME = Path.home() / ".hpema"

def install_dafny() -> None:
    console = Console()
    system = platform.system()
    machine = platform.machine()
    asset = _PLATFORM_MAP.get((system, machine))
    if asset is None:
        console.print(f"[red]Unsupported platform: {system} {machine}[/]")
        sys.exit(1)

    url = f"{DAFNY_RELEASES}/v{DAFNY_VERSION}/{asset}"
    dest_dir = HPEMA_HOME / "dafny"
    dest_dir.mkdir(parents=True, exist_ok=True)

    zip_path = HPEMA_HOME / asset
    console.print(f"Downloading Dafny {DAFNY_VERSION} for {system} {machine}...")

    with httpx.Client(follow_redirects=True, timeout=120) as client:
        with client.stream("GET", url) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with Progress() as progress:
                task = progress.add_task("Downloading", total=total)
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_bytes(65536):
                        f.write(chunk)
                        progress.advance(task, len(chunk))

    console.print("Extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    zip_path.unlink()

    # Find the dafny binary inside the extracted dir
    binary = next(dest_dir.rglob("dafny"), None)
    if system != "Windows" and binary:
        binary.chmod(0o755)

    # Write ~/.hpema/dafny_path so the config loader can find it
    path_file = HPEMA_HOME / "dafny_path"
    path_file.write_text(str(binary))

    console.print(f"[green]Dafny installed at {binary}[/]")
    console.print("Run [bold]hpema[/bold] to start.")
```

Then in `backend/config.py`'s `VerificationConfig`, update `binary_path` resolution:

```python
@property
def resolved_binary_path(self) -> str:
    if self.binary_path not in ("dafny", "LOCAL_DAFNY_INSTALL"):
        return self.binary_path  # user set an explicit path — trust it
    # Check ~/.hpema/dafny_path (written by `hpema setup`)
    path_file = Path.home() / ".hpema" / "dafny_path"
    if path_file.exists():
        return path_file.read_text().strip()
    # Fall through — try system PATH
    return "dafny"
```

Replace every usage of `config.verification.binary_path` in `DafnyRunner` with `config.verification.resolved_binary_path`.

### Approach B: GitHub Release tarballs with Dafny embedded

Platform tarballs (built by `scripts/bundle.sh`) embed the Dafny binary for offline install. This is handled by the bundle script in section 7.

---

## 5. User Data Directory

After packaging, users should not write into the package install directory (they usually can't). Move all writable paths to `~/.hpema/`:

| Current path | After packaging |
|-------------|----------------|
| `data/chromadb/` | `~/.hpema/chromadb/` |
| `logs/` | `~/.hpema/logs/` |
| `logs/audit/` | `~/.hpema/logs/audit/` |

Create a helper in `backend/config.py`:

```python
def _hpema_home() -> Path:
    """Writable user data directory. Created on first use."""
    p = Path(os.environ.get("HPEMA_HOME", Path.home() / ".hpema"))
    p.mkdir(parents=True, exist_ok=True)
    return p
```

Update `load_config()` to rewrite writable relative paths to use `_hpema_home()` when the process is running as an installed package (detect via `__file__` not being under cwd, or just always use `_hpema_home()`).

Specifically:

- `PoliciesConfig.chromadb_dir` default: change from `"data/chromadb"` to `str(_hpema_home() / "chromadb")`
- `PipelineConfig.audit_log_dir` default: change from `"logs/audit"` to `str(_hpema_home() / "logs/audit")`
- `_configure_logging()` in `cli/main.py`: change `Path("logs")` to `_hpema_home() / "logs"`

Source checkout behavior should be unchanged if you keep the `if env_path` branch in `load_config()` — anyone running from source with `HPEMA_CONFIG` set will still use their local paths.

---

## 6. Default Config for Installed Package

When running installed, there is no `hpema_config.yaml` in the cwd. The bundled `hpema_config.yaml` (inside the wheel) serves as the factory default, but it points at the ARC vLLM endpoint — useless for end users.

**Recommended approach:** On first run, if no config is found, copy the bundled default config to `~/.hpema/hpema_config.yaml` and prompt the user to edit it. Then always check `~/.hpema/hpema_config.yaml` before the bundled default.

Resolution order in `load_config()` after packaging:

1. Explicit `path` argument (code-level override)
2. `HPEMA_CONFIG` env var
3. `~/.hpema/hpema_config.yaml` (user's personal config)
4. Bundled package default (via `importlib.resources`) — triggers "first run" copy behavior

Implement the first-run copy in `load_config()`:

```python
user_config = _hpema_home() / "hpema_config.yaml"
if not user_config.exists():
    # Copy bundled default so user can edit it
    bundled = importlib.resources.files("hpema") / "hpema_config.yaml"
    user_config.write_text(bundled.read_text())
    console.print(f"Created default config at {user_config} — edit it to add your API key.")
```

---

## 7. Bundle Script: `scripts/bundle.sh`

Create `scripts/` directory and `scripts/bundle.sh`:

```bash
#!/usr/bin/env bash
# scripts/bundle.sh — Build distributable artifacts for HPEMA.
#
# Produces:
#   dist/hpema-VERSION-py3-none-any.whl        (PyPI wheel, no Dafny)
#   dist/hpema-VERSION-linux-x64.tar.gz        (standalone, Dafny embedded)
#   dist/hpema-VERSION-osx-x64.tar.gz          (standalone, Dafny embedded)
#   dist/hpema-VERSION-windows-x64.zip         (standalone, Dafny embedded)
#
# Usage:
#   bash scripts/bundle.sh               # build all
#   bash scripts/bundle.sh --wheel-only  # skip platform tarballs
#
# Prerequisites:
#   pip install hatchling build
#   DAFNY_VERSION env var (default: 4.9.0)
set -euo pipefail

DAFNY_VERSION="${DAFNY_VERSION:-4.9.0}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST="$ROOT/dist"

cd "$ROOT"

# ─── 1. Tests ────────────────────────────────────────────────────────────────
echo "Running tests..."
python -m pytest tests/ -x -q
echo "Tests passed."

# ─── 2. Python wheel ─────────────────────────────────────────────────────────
echo "Building wheel..."
python -m build --wheel --outdir "$DIST"
WHEEL=$(ls "$DIST"/hpema-*.whl | tail -1)
echo "Wheel: $WHEEL"

if [[ "${1:-}" == "--wheel-only" ]]; then
    echo "Done (wheel only)."
    exit 0
fi

# ─── 3. Platform tarballs with Dafny embedded ────────────────────────────────
# Dafny release assets from GitHub
DAFNY_BASE="https://github.com/dafny-lang/dafny/releases/download/v${DAFNY_VERSION}"

declare -A PLATFORM_ASSETS=(
    ["linux-x64"]="dafny-${DAFNY_VERSION}-x64-ubuntu-20.04.zip"
    ["osx-x64"]="dafny-${DAFNY_VERSION}-x64-osx-11.zip"
    ["windows-x64"]="dafny-${DAFNY_VERSION}-x64-win.zip"
)

VERSION=$(python -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); print(d['project']['version'])")

for PLATFORM in "${!PLATFORM_ASSETS[@]}"; do
    ASSET="${PLATFORM_ASSETS[$PLATFORM]}"
    DAFNY_ZIP="$DIST/$ASSET"
    BUNDLE_DIR="$DIST/hpema-${VERSION}-${PLATFORM}"

    echo "Building $PLATFORM bundle..."
    mkdir -p "$BUNDLE_DIR"

    # Download Dafny if not cached
    if [[ ! -f "$DAFNY_ZIP" ]]; then
        echo "  Downloading Dafny $DAFNY_VERSION for $PLATFORM..."
        curl -fSL "${DAFNY_BASE}/${ASSET}" -o "$DAFNY_ZIP"
    fi

    # Extract Dafny into bundle
    unzip -q "$DAFNY_ZIP" -d "$BUNDLE_DIR/dafny"

    # Copy wheel and install scripts
    cp "$WHEEL" "$BUNDLE_DIR/"
    cp "$SCRIPT_DIR/install.sh" "$BUNDLE_DIR/"   # see install scripts below
    cp "$SCRIPT_DIR/install.ps1" "$BUNDLE_DIR/"

    # Archive
    if [[ "$PLATFORM" == "windows-x64" ]]; then
        (cd "$DIST" && zip -qr "hpema-${VERSION}-${PLATFORM}.zip" "hpema-${VERSION}-${PLATFORM}/")
    else
        tar -czf "$DIST/hpema-${VERSION}-${PLATFORM}.tar.gz" -C "$DIST" "hpema-${VERSION}-${PLATFORM}/"
    fi

    rm -rf "$BUNDLE_DIR"
    echo "  Done: $DIST/hpema-${VERSION}-${PLATFORM}.tar.gz"
done

echo ""
echo "Artifacts in $DIST:"
ls -lh "$DIST"/hpema-*
```

---

## 8. End-User Install Scripts (inside bundles)

These ship inside each platform tarball. Users who download from GitHub Releases run these instead of `pip`.

### `scripts/install.sh` (Linux/macOS)

```bash
#!/usr/bin/env bash
# Installs HPEMA + Dafny from the extracted release bundle.
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "$0")" && pwd)"
HPEMA_HOME="${HPEMA_HOME:-$HOME/.hpema}"

echo "Installing HPEMA..."

# 1. Install Python wheel
pip install --quiet "$BUNDLE_DIR"/hpema-*.whl

# 2. Install Dafny
DAFNY_SRC="$BUNDLE_DIR/dafny"
DAFNY_DEST="$HPEMA_HOME/dafny"
mkdir -p "$DAFNY_DEST"
cp -r "$DAFNY_SRC"/. "$DAFNY_DEST/"
DAFNY_BIN=$(find "$DAFNY_DEST" -name "dafny" -type f | head -1)
chmod +x "$DAFNY_BIN"

# 3. Write dafny_path so HPEMA finds it
mkdir -p "$HPEMA_HOME"
echo "$DAFNY_BIN" > "$HPEMA_HOME/dafny_path"

echo ""
echo "HPEMA installed. Run: hpema"
echo "Edit your config at: $HPEMA_HOME/hpema_config.yaml (created on first run)"
```

### `scripts/install.ps1` (Windows)

```powershell
# Installs HPEMA + Dafny from the extracted release bundle (Windows)
$ErrorActionPreference = "Stop"

$BundleDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$HpemaHome = if ($env:HPEMA_HOME) { $env:HPEMA_HOME } else { "$HOME\.hpema" }

Write-Host "Installing HPEMA..."

# 1. Install Python wheel
$wheel = Get-ChildItem "$BundleDir\hpema-*.whl" | Select-Object -First 1
pip install --quiet $wheel.FullName

# 2. Install Dafny
$dafnySrc  = "$BundleDir\dafny"
$dafnyDest = "$HpemaHome\dafny"
New-Item -ItemType Directory -Force -Path $dafnyDest | Out-Null
Copy-Item -Recurse -Force "$dafnySrc\*" $dafnyDest

$dafnyBin = Get-ChildItem -Recurse "$dafnyDest" -Filter "dafny.exe" | Select-Object -First 1

# 3. Write dafny_path
New-Item -ItemType Directory -Force -Path $HpemaHome | Out-Null
$dafnyBin.FullName | Out-File -FilePath "$HpemaHome\dafny_path" -Encoding ascii

Write-Host ""
Write-Host "HPEMA installed. Run: hpema"
Write-Host "Edit config at: $HpemaHome\hpema_config.yaml (created on first run)"
```

---

## 9. Release Script: `scripts/release.sh`

```bash
#!/usr/bin/env bash
# scripts/release.sh — Tag, publish, and create a GitHub release.
#
# Usage:
#   bash scripts/release.sh 0.2.0
#
# Prerequisites:
#   pip install twine                  (for PyPI upload)
#   gh auth login                      (for GitHub release)
#   PYPI_TOKEN env var                 (for twine)
#   dist/ must contain built artifacts (run bundle.sh first)
set -euo pipefail

VERSION="${1:?Usage: release.sh VERSION}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST="$ROOT/dist"

cd "$ROOT"

# ─── Sanity checks ───────────────────────────────────────────────────────────
if [[ -z "$(ls "$DIST"/hpema-${VERSION}*.whl 2>/dev/null)" ]]; then
    echo "ERROR: No wheel found for version $VERSION in $DIST/. Run bundle.sh first."
    exit 1
fi

if git tag | grep -q "v${VERSION}"; then
    echo "ERROR: Tag v${VERSION} already exists."
    exit 1
fi

# ─── 1. Bump version in pyproject.toml ───────────────────────────────────────
# (Skip if version already set — the developer should set it before running bundle.sh)
CURRENT=$(python -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); print(d['project']['version'])")
if [[ "$CURRENT" != "$VERSION" ]]; then
    echo "ERROR: pyproject.toml version is $CURRENT, expected $VERSION."
    echo "Update [project].version in pyproject.toml and re-run bundle.sh before releasing."
    exit 1
fi

# ─── 2. Git tag ──────────────────────────────────────────────────────────────
echo "Tagging v${VERSION}..."
git tag -a "v${VERSION}" -m "Release v${VERSION}"
git push origin "v${VERSION}"

# ─── 3. Upload to PyPI ───────────────────────────────────────────────────────
echo "Uploading to PyPI..."
twine upload \
    --username __token__ \
    --password "${PYPI_TOKEN:?Set PYPI_TOKEN}" \
    "$DIST"/hpema-"${VERSION}"-*.whl \
    "$DIST"/hpema-"${VERSION}".tar.gz 2>/dev/null || true   # sdist optional

# ─── 4. GitHub release ───────────────────────────────────────────────────────
echo "Creating GitHub release..."
ASSETS=$(ls "$DIST"/hpema-"${VERSION}"-*.{tar.gz,zip,whl} 2>/dev/null | tr '\n' ' ')

gh release create "v${VERSION}" \
    --title "HPEMA v${VERSION}" \
    --notes "$(cat <<EOF
## Install

\`\`\`bash
pip install hpema==${VERSION}
hpema setup     # downloads Dafny for your platform
\`\`\`

Or download a platform bundle below (includes Dafny, no internet required):
- \`hpema-${VERSION}-linux-x64.tar.gz\`
- \`hpema-${VERSION}-osx-x64.tar.gz\`
- \`hpema-${VERSION}-windows-x64.zip\`

See [docs/LOCAL_SETUP.md](docs/LOCAL_SETUP.md) for full setup instructions.
EOF
)" \
    $ASSETS

echo ""
echo "Released v${VERSION}."
echo "PyPI: https://pypi.org/project/hpema/${VERSION}/"
echo "GitHub: $(gh release view v${VERSION} --json url -q .url)"
```

---

## 10. CI Integration (optional but recommended)

Add a GitHub Actions workflow so every push to `main` runs tests, and every push of a `v*` tag triggers the full build + release:

Create `.github/workflows/release.yml`:

```yaml
name: Release

on:
  push:
    tags: ["v*"]

jobs:
  bundle-and-release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install build tools
        run: pip install hatchling build twine

      - name: Install dev deps
        run: pip install -e ".[dev]"

      - name: Bundle
        run: bash scripts/bundle.sh

      - name: Release
        env:
          PYPI_TOKEN: ${{ secrets.PYPI_TOKEN }}
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: bash scripts/release.sh ${{ github.ref_name | replace('v', '') }}
```

**Note on Windows tarballs in CI:** The bundle script downloads Dafny for all three platforms from GitHub. This is cross-compiled (zip extraction is platform-agnostic). The Linux CI runner produces all three platform bundles correctly.

---

## 11. `.gitignore` and `.hpemaignore` Updates

Add to `.gitignore`:

```
dist/
*.egg-info/
__pycache__/
*.pyc
.hpema/
```

The `local/installs/dafny/` directory in the repo (pre-downloaded binaries) should NOT be committed long-term — it is large and the bundle script downloads fresh ones. Consider removing it and relying on the bundle script download.

---

## 12. Testing the Install Before Releasing

Always test the wheel in a clean venv before running `release.sh`:

```bash
# In a temp dir:
python -m venv /tmp/test-hpema
source /tmp/test-hpema/bin/activate
pip install /path/to/dist/hpema-VERSION-py3-none-any.whl

# Smoke test:
hpema --help
hpema setup
# (downloads Dafny, creates ~/.hpema/)

# Set up .env and test a real pipeline run
export NVIDIA_API_KEY=nvapi-...
hpema
# Type a prompt, verify it runs
```

For platform bundles:

```bash
tar -xzf dist/hpema-VERSION-linux-x64.tar.gz -C /tmp/
bash /tmp/hpema-VERSION-linux-x64/install.sh
hpema --help
```

---

## 13. Implementation Order (A → B)

Execute in this order. Do not skip ahead — each step is a dependency of the next.

1. **Dep audit** — run the grep from §2a, categorize every import. No code changes yet.
2. **Write `pyproject.toml`** — exactly as in §3. Install it: `pip install -e ".[dev]"`. Verify `hpema --help` works.
3. **Fix data paths** — update `backend/config.py` for bundled standards and user home dirs (§3, §5, §6). Verify `hpema` still runs from source after the path changes.
4. **Add `hpema setup`** — implement `cli/setup_dafny.py` and the `setup` command (§4). Test that it downloads and sets `~/.hpema/dafny_path` correctly. Verify `DafnyRunner` picks it up via `resolved_binary_path`.
5. **Write `scripts/bundle.sh`** and install scripts (§7, §8). Run it: `bash scripts/bundle.sh --wheel-only` first to test the wheel build, then full run.
6. **Smoke test the wheel** in a clean venv (§12).
7. **Write `scripts/release.sh`** (§9). Do a dry run: comment out the `git tag`, `twine upload`, and `gh release create` lines and just echo the commands.
8. **Set up GitHub Actions** if CI is desired (§10). Test by pushing to a branch (not a tag) first.
9. **First real release:** bump version in `pyproject.toml` to `0.1.0`, run `bundle.sh`, test wheel in clean venv, then `release.sh 0.1.0`.

---

## 14. Known Pitfalls

| Pitfall | Resolution |
|---------|-----------|
| `torch==2.2.2` may conflict with newer Python on some platforms | Core dep — cannot gate it. Note in README that Python 3.11 is the safest target if torch doesn't build cleanly on 3.12 for a given platform. |
| `data/standards/` not found after install | Implement `importlib.resources` lookup in §3. Test by installing wheel and running `hpema` without the source checkout present. |
| Dafny on macOS needs Z3 path | `hpema setup` should also check for Z3 and write `~/.hpema/z3_path`. See `local/installs/dafny/allow_on_mac.sh` for context. |
| Windows users lack `bash` for `install.sh` | The bundle ships both `install.sh` and `install.ps1`. README points Windows users to `.ps1`. |
| `HPEMA_CONFIG` override breaks installed path resolution | Always resolve `HPEMA_CONFIG` relative to cwd if it's a relative path. Absolute paths work as-is. |
| `chromadb` first run is slow (model download) | Expected. Document in README. The embedding model (`all-MiniLM-L6-v2`) downloads to `~/.cache/` on first use. |
| PyPI name `hpema` may be taken | Check on pypi.org before publishing. If taken, use `hpema-pipeline` or similar. |
