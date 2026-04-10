# HPEMA Work Distribution

**Last updated:** 2026-04-08

---

## Team Roles

| Role | Owner | Focus Area |
|------|-------|------------|
| **ML Lead** | (Shehryar) | Pipeline orchestration, LLM serving, CLI, prompt engineering |
| **Verification** | (Person B) | Dafny integration, formal specs, proof testing |
| **RAG + Policy** | (Person C) | Knowledge base, ChromaDB, Policy agent tuning |
| **Backend + Testing** | (Person D) | API hardening, tests, dashboard, audit display |

---

## Active Sprint (Week of 2026-04-08)

### ML Lead — CLI Hardening & Pipeline Engine

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Scrollable output in REPL | TODO | Long code/checker output should scroll — consider Rich `Pager` or paged `/last` view |
| 2 | QA pass — break-test the CLI end-to-end | TODO | Deliberately trigger edge cases: empty response, timeout, malformed JSON, stage skips |
| 3 | Regression test for double-input terminal bug | TODO | Verify `termios` restore works on natural layer-ready path (not just Ctrl+S skip) |
| 4 | Bypass Layer mechanism in `PipelineStage` engine | TODO | Allow individual layers to be force-skipped at runtime without changing config — e.g. `/bypass dafny` |
| 5 | Bypass flag propagation through orchestrator | TODO | `PipelineRequest` carries a `bypass_layers: set[str]` field; orchestrator skips those agent calls |
| 6 | API model fallback robustness | TODO | Ongoing — test prompt-only JSON enforcement across providers (Falcon, Nemotron, etc.) |
| 7 | `.env` key management | DONE | `start_tmux.sh` sources `.env`; keys no longer in script |

### Verification (Person B) — Dafny (Open Problem)

> **Note:** Dafny remains an open research problem for this system. Small instruction models produce syntactically plausible but semantically incorrect specs. Tasks here are exploratory — fewer deliverables, more investigation.

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Characterize failure modes of Actor-generated specs | TODO | Categorise: wrong types, missing invariants, unbounded loops, extern refs |
| 2 | Evaluate prompt scaffolding approaches | TODO | Try chain-of-thought, few-shot Dafny examples, decomposed spec generation |
| 3 | Identify a minimal verifiable subset | TODO | Find the simplest class of requirements (pure functions, no loops) where Dafny consistently passes |
| 4 | Document findings for report | TODO | Even negative results are a contribution — document what doesn't work and why |

### RAG + Policy (Person C) — Independent Layer Readiness

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Standalone RAG retrieval test harness | TODO | Script that queries ChromaDB with sample aerospace terms and checks relevance of returned chunks |
| 2 | Expand `data/standards/` coverage | TODO | Target 30+ rules per standard; prioritise DO-178C and MISRA C for demo |
| 3 | Ingestion script with idempotent upsert | TODO | Re-running ingestion should not duplicate chunks |
| 4 | Retriever quality metrics | TODO | Precision@K on a small manually-curated query set |
| 5 | Policy agent smoke test (isolated) | TODO | Call Policy agent directly with a hardcoded context — no full pipeline dependency |
| 6 | Prepare RAG for Layer 2 inflow | TODO | Policy agent must accept checker output as additional context for compliance check |
| 7 | Citation quality check | TODO | Policy verdicts should cite specific clause IDs (e.g. DO-178C §6.3.2) |

### Backend + Testing (Person D) — Aggressive Layer Testing & Human-in-the-Loop

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Actor layer stress test | TODO | 20+ diverse requirements; log pass/fail, code quality, JSON parse success rate |
| 2 | Checker layer stress test | TODO | Feed Actor outputs directly to Checker; measure verdict consistency |
| 3 | Integration test: Actor → Checker → feedback loop | TODO | At least 3 iterations; assert code improves or status changes |
| 4 | Human-in-the-loop: approval prompt | TODO | After pipeline completes, show final code and ask: Approve / Reject-Rerun / Reject-GiveUp |
| 5 | Human-in-the-loop: Approve path | TODO | Write approved code to a timestamped output file (e.g. `output/<run_id>.py`) |
| 6 | Human-in-the-loop: Reject-Rerun path | TODO | Collect user feedback, append all prior test-case failures to the next Actor prompt, re-enter pipeline |
| 7 | Human-in-the-loop: Reject-GiveUp path | TODO | Print final state summary, hand back to REPL without saving |
| 8 | Timeout + retry hardening in `client.py` | TODO | Catch transient errors (502, connection reset); retry with backoff up to 3× |
| 9 | Streamlit dashboard basic run history view | TODO | Read JSONL audit logs, display per-run status, verdict, code lines |

---

## Deprecated Sprint (Week of 2026-03-30)

### ML Lead — CLI Demo Polish

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Pipeline stage flag (disconnected mode) | DONE | `/stage actor` skips downstream agents |
| 2 | Code formatting post-processor | DONE | Fixes literal `\n` and single-line output |
| 3 | Restart vLLM with `--max-model-len 8192` | TODO | Needs node re-reservation |
| 4 | Test full pipeline with `stage: checker` | TODO | After Dafny install |
| 5 | Create `demo.sh` canned scenario script | TODO | See [plans/CLI.md](plans/CLI.md) Phase 4.3 |
| 6 | Prompt iteration for better code quality | TODO | Test with diverse requirements |
| 7 | Checker Traceability | TODO | Verbose /checker-report to show/store test cases, issues, pytest logs |
| 8 | CLI movement flexibility | TODO | CLI is vim-leaning. Need to acquire CC/Gemini CLI level cursor movement. |
| 9 | Integrate API-based LLMs | TODO | Check this against the Nvidia Free Inference Endpoints |

### Verification (Person B) — Dafny Integration

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Install Dafny on ARC | DONE | Pre-built binary, path set in config (2026-03-31) |
| 2 | Write 3-5 sample `.dfy` specs | TODO | Binary search, altitude controller, array bounds. Put in `tests/fixtures/dafny/` |
| 3 | Smoke test `dafny_runner.py` standalone | TODO | Run smoke test below |
| 4 | Test Actor-generated Dafny specs | TODO | Run pipeline with `stage: checker`, check if specs parse |
| 5 | Tune Actor prompt for better Dafny output | TODO | May need examples in the prompt |
| 6 | Update `hpema_config.arc.yaml` binary_path | DONE | Pointed to installed binary (2026-03-31) |

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
