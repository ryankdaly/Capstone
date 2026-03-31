# HPEMA Progress

**Branch:** `feat/skeleton-architecture`
**Last updated:** 2026-03-30

---

## Current State

The HPEMA pipeline runs end-to-end on a VT ARC GPU node. A user types a requirement into the interactive CLI, the Actor agent generates safety-critical C code via constrained decoding on vLLM, and the output is displayed with syntax highlighting in a Claude-Code-style terminal UI.

**What works right now:**
- Interactive REPL (`python -m cli.main`) with slash commands
- In-process orchestrator (no FastAPI needed for demo)
- Actor agent generates code via Qwen2.5-Coder-7B-Instruct on vLLM
- Constrained decoding enforces Pydantic schemas at the token level
- Pipeline stage control (`/stage actor`, `/stage checker`, `/stage policy`)
- Disconnected mode warnings when running partial pipeline
- Startup LLM health check
- Audit logging (JSONL per run)
- tmux workflow: `bash ml/slurm/start_tmux.sh` sets up vLLM + CLI

**What does NOT work yet:**
- Dafny verification (binary not installed on ARC)
- Checker + Policy agents (untested — blocked on stage integration testing)
- RAG retrieval (ChromaDB not populated)
- Multi-iteration feedback loop (needs Checker + Dafny + Policy to drive feedback)
- Streamlit dashboard
- Human-in-the-loop approval flow

---

## Architecture Summary

```
User types requirement (CLI REPL)
    -> Orchestrator creates run (UUID)
        -> Actor Agent generates code + Dafny spec (constrained decoding)
        -> [stage >= checker] Checker reviews code  }  parallel
           [stage >= checker] Dafny verifies spec   }
        -> [stage >= policy] Policy checks compliance (RAG-backed)
        -> All pass? -> Done. Any fail? -> Compose feedback -> loop (max 3)
    -> Output: code + proof + audit trail
```

See [docs/ARCHITECTURE.md](ARCHITECTURE.md) for the full plan.

---

## How to Run

### On ARC (GPU node)

```bash
# 1. Reserve a node
salloc --partition=a100_normal_q --gres=gpu:1 --time=2:00:00

# 2. cd to project, launch tmux workflow
cd /projects/meng/Capstone
bash ml/slurm/start_tmux.sh

# Pane 0: vLLM starts serving Qwen2.5-Coder-7B-Instruct
# Pane 1: CLI starts after 60s wait (for model loading)
```

### Manually restart CLI (in tmux pane 1)

```bash
export HPEMA_CONFIG=hpema_config.arc.yaml
export HPEMA_API_KEY=unused
python -m cli.main
```

### Locally (no GPU — tests only)

```bash
source .venv/bin/activate
pytest tests/ -v
```

---

## Key Config Files

| File | Purpose |
|------|---------|
| `hpema_config.yaml` | Default config (ARC shared endpoint) |
| `hpema_config.arc.yaml` | HPC node config (vLLM on localhost:8001, stage: actor) |
| `backend/services/llm/prompts/*.txt` | System prompts for each agent |

---

## Completed Milestones

| Date | Milestone |
|------|-----------|
| 2026-03-04 | Skeleton architecture: all schemas, agents, orchestrator, audit, RAG, tests |
| 2026-03-15 | CLI plan written, in-process runner, Claude-Code display rewrite |
| 2026-03-20 | Interactive REPL with slash commands, session state, LLM health check |
| 2026-03-25 | vLLM serving on ARC, constrained decoding working, Phi-4 tested |
| 2026-03-28 | Switched to Qwen2.5-Coder-7B (OOM fix), prompt tuning for conciseness |
| 2026-03-30 | Pipeline stage flag (disconnected mode), code formatting post-processor |

---

## Known Issues

1. **Model outputs literal `\n`** — post-processor in `client.py` strips them, but prompt should also be tuned
2. **24GB GPU constraint** — Qwen2.5-Coder-7B fits, but larger models (Phi-4 14B) do not on current allocation
3. **max_tokens tight** — set to 2000 with 4096 context window; bump `--max-model-len 8192` on vLLM restart
4. **No Dafny on ARC** — needs .NET SDK or pre-built binary install
