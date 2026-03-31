# HPEMA — Hierarchical Policy-Enforced Multi-Agent

High-assurance code generation pipeline for safety-critical aerospace software. Three guarantees: **constrained decoding** (token-level schema enforcement), **formal verification** (Dafny proofs), and **live policy audit** (RAG-backed standard citations).

## Quick Start (ARC GPU Node)

```bash
salloc --partition=a100_normal_q --gres=gpu:1 --time=2:00:00
cd /projects/meng/Capstone
bash ml/slurm/start_tmux.sh
```

## Documentation

| Doc | What |
|-----|------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Full system architecture and design rationale |
| [docs/PROGRESS.md](docs/PROGRESS.md) | Current state, what works, known issues |
| [docs/WORK_DISTRIBUTION.md](docs/WORK_DISTRIBUTION.md) | Task tracking by team member |
| [docs/plans/CLI.md](docs/plans/CLI.md) | CLI implementation plan (phases 1-3 complete) |

## Project Structure

```
cli/                    CLI layer (typer + rich REPL)
backend/
  api/schemas/          Pydantic data contracts between agents
  services/
    agents/             Actor, Checker, Policy agent implementations
    llm/                Async OpenAI client, model registry, prompts
    verification/       Dafny subprocess runner
    rag/                ChromaDB retriever for safety standards
    audit/              JSONL + SQLite dual-write audit logger
    orchestrator.py     Pipeline state machine
data/standards/         Sample safety standard documents for RAG
ml/slurm/              SLURM scripts and tmux launcher for ARC
hpema_config.yaml      Default configuration
hpema_config.arc.yaml  HPC node configuration (vLLM on localhost)
```
