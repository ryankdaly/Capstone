# HPEMA Work Distribution

**Last updated:** 2026-04-04

---

## Team Roles

| Role | Owner | Focus Area |
|------|-------|------------|
| **ML Lead** | (Shehryar) | Pipeline orchestration, LLM serving, CLI, prompt engineering |
| **Verification** | (Person B) | Dafny integration, formal specs, proof testing |
| **RAG + Policy** | (Person C) | Knowledge base, ChromaDB, Policy agent tuning |
| **Backend + Testing** | (Person D) | API hardening, tests, dashboard, audit display |

---

## Active Sprint (Week of 2026-03-30)

### ML Lead — CLI Demo Polish

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Pipeline stage flag (disconnected mode) | DONE | `/stage actor` skips downstream agents |
| 2 | Code formatting post-processor | DONE | Fixes literal `\n` and single-line output |
| 3 | Restart vLLM with `--max-model-len 8192` | TODO | Needs node re-reservation |
| 4 | Test full pipeline with `stage: checker` | TODO | After Dafny install |
| 5 | Create `demo.sh` canned scenario script | TODO | See [plans/CLI.md](plans/CLI.md) Phase 4.3 |
| 6 | Prompt iteration for better code quality | TODO | Test with diverse requirements |

### Verification (Nathan Chatpolarak) — Dafny Integration

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Install Dafny on ARC | DONE | Option A: `dotnet tool install --global dafny`. Option B: pre-built binary from GitHub releases |
| 2 | Write 3-5 sample `.dfy` specs | DONE | Binary search, altitude controller, array bounds. Put in `tests/fixtures/dafny/` |
| 3 | Smoke test `dafny_runner.py` standalone | DONE | `python -c "from backend.services.verification.dafny_runner import DafnyRunner; ..."` |
| 4 |  | DONE | Run pipeline with `stage: checker`, check if specs parse |
| 5 | Tune Actor prompt for better Dafny output | DONE | May need examples in the prompt |
| 6 | Update `hpema_config.arc.yaml` binary_path | DONE | Point to installed Dafny binary |

### RAG + Policy (Person C) — Knowledge Base

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Install ChromaDB + sentence-transformers | TODO | `pip install chromadb sentence-transformers` |
| 2 | Expand `data/standards/` documents | TODO | Add 20-30 more rules per standard |
| 3 | Write ingestion script | TODO | Script to load `data/standards/` into ChromaDB |
| 4 | Test retriever with sample queries | TODO | `retriever.retrieve("dynamic allocation DO-178C")` |
| 5 | Test Policy agent with RAG context | TODO | Run pipeline with `stage: policy` |
| 6 | Tune Policy prompt for better citations | TODO | Should cite specific clause IDs |

### Backend + Testing (Person D) — Hardening

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Update existing unit tests for stage flag | TODO | `test_schemas.py` needs PipelineStage tests |
| 2 | Write integration test (mocked LLM) | TODO | Full pipeline with mock responses, verify all stages |
| 3 | Add error handling for LLM timeouts | TODO | `client.py` should catch and retry transient errors |
| 4 | Streamlit dashboard basic view | TODO | Read JSONL audit logs, display run history |
| 5 | CLI `/audit` display polish | TODO | Better table formatting for traceability matrix |
| 6 | Human-in-the-loop approval flow | TODO | After pipeline complete, prompt approve/reject |

---

## Completed Tasks (Archive)

### Skeleton Architecture (2026-03-04)
- [x] All Pydantic schemas (agents.py, pipeline.py, audit.py)
- [x] LLM client with constrained decoding
- [x] Model registry from YAML config
- [x] Agent base class + Actor, Checker, Policy implementations
- [x] Orchestrator state machine
- [x] Feedback composition
- [x] Dafny runner (code ready, binary not installed)
- [x] RAG retriever (code ready, ChromaDB not populated)
- [x] Audit logger (JSONL + SQLite dual-write)
- [x] Traceability matrix generator
- [x] FastAPI routes (pipeline SSE, audit queries)
- [x] 26 unit tests passing
- [x] Sample standards documents (DO-178C, MISRA C, NASA, Boeing SDP)

### CLI Implementation (2026-03-15 — 2026-03-30)
- [x] In-process runner (`cli/runner.py`)
- [x] Claude-Code-style display (`cli/display.py`)
- [x] Interactive REPL with slash commands (`cli/repl.py`)
- [x] tmux workflow script (`ml/slurm/start_tmux.sh`)
- [x] Smoke test script (`ml/slurm/smoke_test.py`)
- [x] vLLM serving with constrained decoding on ARC
- [x] Switched model to Qwen2.5-Coder-7B-Instruct (24GB GPU fit)
- [x] Pipeline stage flag (disconnected mode)
- [x] Code formatting post-processor

---

## Dependency Graph

```
                    Install Dafny (B.1)
                         |
                         v
Sample .dfy specs (B.2) --> Smoke test runner (B.3)
                                    |
                                    v
                         Test Actor Dafny output (B.4)
                                    |
Ingest KB (C.3) ----+              |
       |            |              v
       v            +---> Test full pipeline (stage: policy)
Test Policy (C.5) --+              |
                                   v
                         Demo script (ML.5)
```

Tasks B.1 and C.1-C.3 can proceed in parallel immediately.

---

## How to Update This File

After completing a task:
1. Move the row's Status from `TODO` to `DONE`
2. Add a date in Notes
3. If new tasks emerge, add them to the appropriate person's table
4. Move completed sprints to the Archive section
