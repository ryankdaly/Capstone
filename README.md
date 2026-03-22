# HPEMA

Hierarchical Policy-Enforced Multi-Agent pipeline for high-assurance, safety-critical code generation.

## Prerequisites

### Option A — Docker (recommended)

**Requirement:** [Docker Desktop](https://www.docker.com/products/docker-desktop/)

```bash
# 1. Build the image
docker build -t hpema .

# 2. Verify Dafny is working inside the container
docker run --rm hpema dafny --version

# 3. Start the API
docker run --rm -e HPEMA_API_KEY=your-key -p 8000:8000 hpema
```

Open `http://localhost:8000` to see the status page confirming the API and Dafny verifier are online.

**Or use Docker Compose** (recommended for persistent logs):
```bash
HPEMA_API_KEY=your-key docker compose up --build

# Use ARC cluster config
HPEMA_API_KEY=your-key HPEMA_CONFIG=hpema_config.arc.yaml docker compose up --build
```

Audit logs and standards data are mounted as volumes (`./logs`, `./data`) so they persist across container restarts.

### Option B — Local

| Dependency | Version | Install |
|------------|---------|---------|
| Python | 3.13+ | [python.org](https://www.python.org/downloads/) |
| .NET SDK | 8.0+ | `brew install --cask dotnet-sdk` |
| Dafny | 4.x | `dotnet tool install --global Dafny` |
| Z3 | any | `brew install z3` |

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export HPEMA_API_KEY="your-key"
uvicorn backend.main:app --reload --port 8000
```

> **Note:** Dafny requires Z3 at runtime. Pass the solver path when verifying locally:
> ```bash
> dafny verify --solver-path $(brew --prefix z3)/bin/z3 <file.dfy>
> ```

## CLI Usage

```bash
# Generate code
python -m cli.main generate \
    --requirement "Implement a binary search function with bounds checking" \
    --standard DO_178C \
    --language C

# Query audit trail
python -m cli.main audit --run-id <uuid>

# Launch dashboard
python -m cli.main dashboard
```

## Supported Standards

- `DO_178C` — Airborne software
- `NASA` — NASA-STD-8739
- `BOEING_SDP` — Boeing software design practices
- `MISRA_C` — MISRA C coding rules
