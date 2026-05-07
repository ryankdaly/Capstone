# HPEMA — Local API Setup Guide

Run HPEMA on your own machine against any OpenAI-compatible API (NVIDIA NIM, OpenAI, Groq, etc.). No GPU required.

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.12 recommended (3.11+ ok) | [python.org](https://python.org) / `brew install python@3.12` |
| Git | any | system package manager |
| Dafny | 4.x | see [§ Dafny Install](#dafny-install) below |

---

## 1. Clone and Install Python Dependencies

```bash
git clone <repo-url>
cd Capstone

python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> **Note:** `torch`, `sentence-transformers`, and `chromadb` are only needed for the RAG / policy layer. If you want a lighter install (Actor + Checker only), omit those packages and set `stage: "checker"` in your config.

---

## 2. Create Your `.env`

Copy the example and fill in your keys:

```bash
cp .env.example .env
```

Open `.env` and edit:

```bash
# Required — your NVIDIA NIM key (or whichever provider you use)
export NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxx

# Suppress ChromaDB telemetry noise
export ANONYMIZED_TELEMETRY=False

# Point HPEMA at your local config (see § 3 below)
export HPEMA_CONFIG=local/configs/hpema_config.local.yaml
```

Available key env vars that configs reference:

| Env var | Provider |
|---------|----------|
| `NVIDIA_API_KEY` | NVIDIA NIM (`integrate.api.nvidia.com`) |
| `OPENAI_API_KEY` | OpenAI |
| `GROQ_API_KEY` | Groq |
| `HPEMA_API_KEY` | ARC / vLLM (any value works for local vLLM) |

---

## 3. Pick (or Write) a Config File

HPEMA loads whichever file `HPEMA_CONFIG` points to. Two ready-made local configs live in `local/configs/`:

| File | Models |
|------|--------|
| `hpema_config.local.yaml` | Mistral Large 3 (Actor/Policy) + Step-3.5-Flash (Checker/Dafny) |
| `hpema_config.local1.yaml` | Mistral Large 3 across all agents |

To use one:

```bash
export HPEMA_CONFIG=local/configs/hpema_config.local.yaml
```

### Anatomy of a config

```yaml
models:
  actor:
    endpoint: "https://integrate.api.nvidia.com/v1"   # OpenAI-compatible base URL
    model: "mistralai/mistral-large-3-675b-instruct-2512"
    api_key_env: "NVIDIA_API_KEY"                      # env var holding the key
    # extra_body:                                       # optional — model-specific params
    #   chat_template_kwargs:
    #     enable_thinking: true
    #     thinking_budget: 2000

  checker:   { ... }   # same structure
  policy:    { ... }
  dafny_architect: { ... }

policies:
  standards_dir: "data/standards"    # RAG source documents
  chromadb_dir:  "data/chromadb"     # vector DB on disk
  default_standard: "DO_178C"

verification:
  prover: "dafny"
  timeout_seconds: 120
  binary_path: "/path/to/dafny"      # ← set this (see § 4)
  solver_path: null                  # set to Z3 path on macOS if needed

pipeline:
  max_iterations: 3
  require_human_approval: false
  stage: "policy"                    # "actor" | "checker" | "policy"
```

### `stage` controls how far the pipeline runs

| Value | Agents active |
|-------|--------------|
| `"actor"` | Code generation only — fast, no verification |
| `"checker"` | Actor + Checker + Dafny |
| `"policy"` | Full pipeline — all four agents + RAG |

### Writing your own config

Copy an existing one and change `model` / `endpoint` / `api_key_env` per agent. Every agent can use a different provider. The pipeline auto-detects capability gaps (JSON schema support, thinking params, system role) and falls back gracefully.

---

## 4. Dafny Install

Dafny is only needed when `stage` is `"checker"` or `"policy"`.

### macOS (Homebrew)

```bash
brew install dotnet
brew install dafny
which dafny              # e.g. /opt/homebrew/bin/dafny
```

Set `binary_path` in your config to the output of `which dafny`.

macOS often can't find Z3 automatically — set `solver_path` too:

```bash
brew install z3
which z3                 # e.g. /opt/homebrew/bin/z3
```

Then in your config:

```yaml
verification:
  binary_path: "/opt/homebrew/bin/dafny"
  solver_path: "/opt/homebrew/bin/z3"
```

### Linux

```bash
# Install .NET 8 runtime first (required by Dafny)
wget https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
sudo apt-get update && sudo apt-get install -y dotnet-runtime-8.0

# Download Dafny release binary
DAFNY_VERSION=4.9.0
wget https://github.com/dafny-lang/dafny/releases/download/v${DAFNY_VERSION}/dafny-${DAFNY_VERSION}-x64-ubuntu-20.04.zip
unzip dafny-*.zip -d ~/dafny
```

Then set in your config:

```yaml
verification:
  binary_path: "/home/<you>/dafny/dafny"
```

### Pre-built binary (all platforms)

Download from [github.com/dafny-lang/dafny/releases](https://github.com/dafny-lang/dafny/releases).  
Pick the archive matching your OS, extract it, and point `binary_path` at the `dafny` executable inside.

### Verify the install

```bash
dafny --version
# Dafny 4.x.x
```

---

## 5. Source `.env` and Launch

```bash
source .env
python -m cli.main
```

The CLI starts, animates a connectivity check for each configured model endpoint, then drops into the interactive prompt.

```
hpema › Write a bounded FIFO queue in Python, DO-178C compliant
```

---

## 6. Common Flags and Commands

### In-CLI commands

| Command | Effect |
|---------|--------|
| `/stage actor` | Switch to code-gen only (no verification) |
| `/stage checker` | Actor + Checker + Dafny |
| `/stage policy` | Full pipeline |
| `/iterations 5` | Change max retry iterations |
| `/run-tests off` | Skip pytest execution |
| `/config` | Show active config file and model assignments |
| `/last` | Re-display last run result |
| `/help` | All commands |

### Pointing at a different config without restarting

```bash
export HPEMA_CONFIG=local/configs/hpema_config.local1.yaml
python -m cli.main
```

---

## 7. Directory Reference

```
hpema_config.yaml              Default config (ARC vLLM)
hpema_config.openai.yaml       OpenAI config template
local/configs/
  hpema_config.local.yaml      Local API config (active)
  hpema_config.local1.yaml     Alternate local config
data/
  standards/                   RAG source documents (DO-178C, MISRA-C, NASA, Boeing)
  chromadb/                    Vector DB (auto-created on first run)
logs/
  hpema.log                    Full debug log
  agent.log                    Per-agent prompt + response dump
  audit/                       JSONL audit trail per pipeline run
```

---

## Troubleshooting

**Layer status shows a model as "Retrying..." forever**
→ The endpoint is unreachable or the API key is wrong. Press `Ctrl+S` to skip and proceed — the error will surface on the first real request.

**`dafny: command not found`**  
→ `binary_path` in your config is wrong, or Dafny isn't on PATH. Set the full absolute path.

**Parse failures / empty output on first run**  
→ The pipeline auto-detects unsupported features (JSON schema, thinking params) and retries. Expect one extra round-trip per capability gap discovered on a fresh process.

**ChromaDB errors on policy stage**  
→ `data/chromadb/` will be created and ingested automatically on first use. If it corrupts, delete the directory and restart.

**`HPEMA_CONFIG` not picked up**  
→ Make sure you used `export` (not just assignment) and sourced the file in the same shell session: `source .env`.
