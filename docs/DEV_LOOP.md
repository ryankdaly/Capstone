# HPEMA — Developer Loop Guide

How to work on HPEMA, install it as a real binary, test it, and iterate — on macOS, Linux, and Windows.

---

## TL;DR (the one-minute version)

```bash
# One-time setup — puts `hpema` on your PATH globally
pipx install -e .          # macOS / Linux
# or:
uv tool install -e .       # same effect, uv flavour

# Every day
hpema                      # runs from your source (no rebuild needed)
python -m cli.main         # identical alternative

# When you want to simulate a wheel install (end-user experience)
python -m build --sdist && pipx install dist/hpema-*.tar.gz --force
```

---

## 1. Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.11+ | [python.org](https://python.org) or `brew install python@3.12` |
| pipx | any | `pip install pipx` then `pipx ensurepath` |
| uv (optional) | any | `pip install uv` |
| build | any | `pip install build` |
| Dafny | 4.x | Only needed for verification — see [LOCAL_SETUP.md](LOCAL_SETUP.md) §4 |
| Git | any | system |

> **Windows note:** use PowerShell 7+ or Windows Terminal. Git Bash also works for most commands.

---

## 2. First-Time Setup (clone → install → on PATH)

```bash
git clone <repo>
cd Capstone

# Install as a global tool — hpema lands on PATH immediately
pipx install -e ".[dev]"
```

Then just run:

```bash
hpema
```

The setup wizard runs automatically on first launch and writes your config to `~/.hpema/`.

### Alternative: venv-based (no pipx)

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -e ".[dev]"
hpema    # works while venv is active
```

---

## 3. The Development Loop

### Mode A — Editable install (recommended)

```bash
pipx install -e ".[dev]"
```

`hpema` is a real binary that reads directly from your source files. **Edit any `.py` file and the change is live immediately — no rebuild needed.**

```
edit code → hpema → see change
```

Config discovery order in source/editable mode (first match wins):

1. `HPEMA_CONFIG` env var (absolute path)
2. `local/configs/hpema_config.local.yaml` ← developer configs live here
3. `local/configs/hpema_config.local1.yaml`
4. Any other `*.yaml` in `local/configs/` (alphabetical)
5. `~/.hpema/hpema_config.yaml` ← written by `/setup` wizard
6. Project-root `hpema_config.yaml`

### Mode B — Wheel cycle (tests real packaging)

Use this to verify data files are bundled correctly and the install experience matches what end users get.

```bash
python -m build --sdist
pipx install dist/hpema-0.1.0.tar.gz --force
hpema    # acts exactly like a fresh user install
```

### Switching between modes

```bash
# Editable install
pipx install -e ".[dev]"

# Back to wheel install
python -m build --sdist && pipx install dist/hpema-*.tar.gz --force

# Uninstall
pipx uninstall hpema
```

---

## 4. Config Isolation

### HPEMA home directory

All user configs, API keys, and runtime data live here — never in the project root.

| Platform | Default path | Override |
|----------|-------------|---------|
| macOS / Linux | `~/.hpema/` | `export HPEMA_HOME=/custom/path` |
| Windows | `%APPDATA%\hpema\` | `set HPEMA_HOME=C:\custom\path` |

```
~/.hpema/
  hpema_config.yaml         ← primary config (written by /setup)
  hpema_config_nvidia.yaml  ← additional named configs
  .env                      ← API keys (bash export format)
  .config_verified          ← marker: user confirmed config on first boot
  chromadb/                 ← vector DB for policy RAG (auto-created)
  standards/                ← copy of bundled DO-178C docs (wheel install)
```

### Config files in source mode

```
Capstone/
  local/configs/
    hpema_config.local.yaml    ← your per-machine dev config (checked first)
    hpema_config.local1.yaml   ← alternate dev config
  hpema_config.yaml            ← root template / fallback
```

Developer configs in `local/configs/` take priority over `~/.hpema/` in source and editable mode. A wheel install ignores `local/configs/` entirely.

### The HPEMA_CONFIG rule

| Situation | Behaviour |
|-----------|-----------|
| Absolute path | Always used |
| Relative path, source/editable mode | Resolved against project root |
| Relative path, wheel install | Silently ignored (prevents dev-env bleed) |

```bash
# macOS / Linux — absolute override
HPEMA_CONFIG=/path/to/custom.yaml hpema

# Windows PowerShell
$env:HPEMA_CONFIG = "C:\path\to\custom.yaml"; hpema
```

---

## 5. Config Management (/setup wizard)

The wizard is launched automatically on first run and is available any time via `/setup` inside the REPL.

### When existing configs are found

```
~/.hpema/ already has configs. What would you like to do?
  ▶ Edit existing configuration   ← pick from dropdown of existing files
    Create new configuration      ← run wizard, choose a name
```

- **Edit existing**: shows a picker of all `hpema_config*.yaml` files in `~/.hpema/`. Selecting one sets `HPEMA_CONFIG` and restarts.
- **Create new**: runs the full wizard. You are asked for a name (leave blank for auto-generated). The config is saved as `hpema_config_{name}.yaml` and never overwrites an existing file.

### Wizard steps (new config)

1. Provider (NVIDIA NIM, OpenAI, Groq, ARC, local, custom)
2. Endpoint confirmation
3. API key
4. Model names per agent
5. **Model family** — controls thinking-mode kwargs and JSON schema support
6. Review & Save → optional restart

### family field

Every config generated by the wizard includes a `family:` field per agent. This determines which kwarg profile the LLM client uses. Do not leave it as `"generic"` for known providers — the pipeline will still work but thinking/reasoning mode won't activate correctly.

| Provider | Default family |
|----------|---------------|
| NVIDIA NIM | `mistral` (adjust if using Qwen3, Gemma, etc.) |
| OpenAI | `openai-gpt4o` |
| Groq | `generic` |
| ARC / local | `generic` |

All valid families: `generic`, `mistral`, `mistral-reasoning`, `qwen3`, `gemma3`, `gemma4`, `kimi-k2`, `anthropic`, `openai-gpt4o`, `openai-reasoning`, `stepfun`, `minimax`, `deepseek-r1`, `deepseek-v3`

---

## 6. GitHub Releases (CI)

Pushing to `main` triggers `.github/workflows/release.yml`:

1. Reads `version` from `pyproject.toml`
2. Builds a source distribution (`dist/hpema-X.Y.Z.tar.gz`)
3. Creates a GitHub Release tagged `vX.Y.Z` with install instructions
4. Skips if the tag already exists (idempotent)

**To cut a new release:** bump `version` in `pyproject.toml`, commit, push to `main`.

**End-user install from a release:**

```bash
# Recommended — puts hpema on PATH automatically
pipx install https://github.com/<org>/Capstone/releases/download/vX.Y.Z/hpema-X.Y.Z.tar.gz

# or with uv
uv tool install https://...hpema-X.Y.Z.tar.gz
```

---

## 7. Uninstalling

| How installed | How to remove |
|--------------|--------------|
| `pipx install` | `pipx uninstall hpema` |
| `uv tool install` | `uv tool uninstall hpema` |
| `pip install -e .` (venv) | `pip uninstall hpema` (with venv active) |
| `uv pip install -e .` | `uv pip uninstall hpema` |

To also wipe user data:

```bash
# macOS / Linux
rm -rf ~/.hpema/

# Windows PowerShell
Remove-Item -Recurse -Force "$env:APPDATA\hpema"
```

---

## 8. Running Tests

```bash
# Unit tests (no API key needed)
pytest tests/unit/ -q

# Dafny-specific unit tests
pytest tests/ -q -k "dafny" -m "not integration"

# Integration tests (require live API endpoint + key)
pytest tests/integration/ -q
```

---

## 9. Startup Behaviour

```
No config found?
  └─ Auto-launch /setup wizard → writes ~/.hpema/hpema_config.yaml → restart

Config found, first boot (no .config_verified)?
  └─ Show config summary
     "Continue? [Y/n]"
       Yes → write .config_verified → proceed
       No  → launch /setup wizard

Config found and verified?
  └─ Silent pass-through → logo → ready
```

---

## 10. Windows-Specific Notes

- **Python**: use 3.11+ from [python.org](https://python.org), not the Windows Store version (subprocess sandboxing issues)
- **PowerShell 7+** recommended: `winget install Microsoft.PowerShell`
- **pipx on PATH**: after `pip install pipx`, run `pipx ensurepath` and restart your terminal
- **Dafny**: download from [github.com/dafny-lang/dafny/releases](https://github.com/dafny-lang/dafny/releases), extract, add the folder to `PATH` — `/setup` will auto-detect it
- **API keys**: the `.env` file uses bash `export` syntax and is not auto-loaded on Windows. Set keys directly in PowerShell before launching:
  ```powershell
  $env:NVIDIA_API_KEY = "nvapi-..."
  hpema
  ```
  Or add them to your PowerShell profile (`$PROFILE`) for persistence.
- **YAML paths**: all paths in wizard-generated configs use forward slashes (`C:/Users/...`) which Python's `pathlib` handles correctly on Windows.
- **Config location**: `%APPDATA%\hpema\` (e.g. `C:\Users\you\AppData\Roaming\hpema\`)
- **`bundle.sh` / `release.sh`**: require WSL or Git Bash. Use `python -m build` directly instead.

---

## 11. Quick Reference

```bash
# Install (editable — fastest, hpema on PATH)
pipx install -e ".[dev]"

# Install (wheel — tests packaging)
python -m build --sdist && pipx install dist/hpema-*.tar.gz --force

# Uninstall
pipx uninstall hpema

# Run tests
pytest tests/unit/ -q

# Build release artifact
python -m build --sdist

# Re-run setup wizard
hpema    # then type /setup

# Use a specific config
HPEMA_CONFIG=/abs/path/to/config.yaml hpema
```
