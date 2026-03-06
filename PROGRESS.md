# HPEMA Progress Report — Skeleton Architecture

**Branch:** `feat/skeleton-architecture` (off `ml`)
**Date:** 2026-03-04
**Status:** Skeleton complete, all imports pass, 26 unit tests pass, FastAPI app serves all routes.

---

## What's Been Built

The entire HPEMA pipeline architecture is implemented as working, importable code. Every component connects — config loads, agents instantiate, orchestrator wires everything together, API serves all routes. This is the scaffold that all further work builds on.

### Architecture in Plain English

```
User types a requirement (CLI or API)
    → Orchestrator creates a run (UUID) and starts the loop
        → Actor Agent generates code + Dafny formal spec
        → Checker Agent reviews code  }  these run in PARALLEL
           Dafny Verifier proves spec }
        → Policy Agent checks compliance (using RAG to retrieve standard rules)
        → All pass? → Done. Any fail? → Compose feedback → loop (max 3 times)
    → Output: code + proof + tests + audit trail + compliance report
```

### Files Created

| File | What It Does |
|---|---|
| `hpema_config.yaml` | Single config file for the whole pipeline — swap models, standards, prover here |
| `backend/config.py` | Loads YAML config into typed Pydantic models, env var overrides |
| **Schemas (the data contracts between agents)** | |
| `backend/api/schemas/agents.py` | `CodeCandidate`, `CheckerReport`, `VerificationResult`, `PolicyVerdict`, `FeedbackMessage` |
| `backend/api/schemas/pipeline.py` | `PipelineRequest`, `PipelineState`, `IterationRecord`, `StreamEvent` |
| `backend/api/schemas/audit.py` | `AuditEntry`, `TraceabilityMatrix`, query schemas |
| **LLM Layer** | |
| `backend/services/llm/client.py` | Async OpenAI client — supports constrained decoding via `guided_json` |
| `backend/services/llm/model_registry.py` | Maps agent roles → endpoints (from config) |
| `backend/services/llm/prompts/*.txt` | System prompts for Actor, Checker, Policy agents |
| **Agents** | |
| `backend/services/agents/base.py` | Base class: loads prompt, calls LLM, parses structured output |
| `backend/services/agents/actor.py` | Generates code + Dafny spec, handles feedback from prior iterations |
| `backend/services/agents/checker.py` | Reviews code, finds bugs, generates test cases |
| `backend/services/agents/policy.py` | Checks compliance using RAG-retrieved standard clauses |
| **Core Pipeline** | |
| `backend/services/orchestrator.py` | State machine: Actor → (Checker ∥ Dafny) → Policy → loop. ~180 lines |
| `backend/services/feedback.py` | Composes prioritized feedback from all checker outputs |
| **Verification** | |
| `backend/services/verification/dafny_runner.py` | Runs `dafny verify` as async subprocess, parses results |
| `backend/services/verification/sandbox.py` | Temp directory sandboxing for safe execution |
| **RAG** | |
| `backend/services/rag/retriever.py` | ChromaDB vector search over safety standards |
| `backend/services/rag/knowledge_graph.py` | Standards ingestion utilities |
| `data/standards/*/` | Sample standard documents (DO-178C, MISRA C, NASA, Boeing SDP) |
| **Audit** | |
| `backend/services/audit/logger.py` | Dual-write: JSONL per run + SQLite for cross-run queries |
| `backend/services/audit/traceability.py` | Generates requirement→code→test→proof→policy matrix |
| **API** | |
| `backend/api/routers/pipeline.py` | SSE streaming endpoint for real-time pipeline events |
| `backend/api/routers/audit.py` | Audit trail query endpoints |
| `backend/api/dependencies.py` | FastAPI dependency injection (singleton services) |
| **CLI** | |
| `cli/main.py` | `hpema generate`, `hpema audit`, `hpema dashboard` commands (typer + rich) |
| `cli/display.py` | Rich panels for streaming agent status |
| **Dashboard** | |
| `dashboard/app.py` | Streamlit audit viewer (reads same JSONL logs) |
| **Infra** | |
| `ml/slurm/serve_models.sh` | SLURM script to launch vLLM instances on ARC |
| `tests/unit/test_*.py` | 26 unit tests (schemas, feedback, audit, config) |

---

## What Works Right Now

- `pytest tests/` → **26/26 pass**
- FastAPI app starts, all 5 routes register (`/health`, `/generate`, `/pipeline/run`, `/audit/query`, `/audit/run/{id}`)
- Config loads from YAML with env var overrides
- Audit logger writes JSONL + SQLite
- Feedback composition correctly prioritizes failures
- All schemas export JSON schemas (ready for constrained decoding)

## What Does NOT Work Yet

The pipeline skeleton is wired but **no LLM calls have been tested end-to-end** — that requires an API key and a running model endpoint. The individual pieces (config, schemas, audit, feedback) are tested; the full loop is not.

---

## Next Steps — By Role

### Back-End (API + Orchestrator)

1. **Wire the generation endpoint fully** — right now `/generate` runs the pipeline but doesn't extract the final code from the state properly. Needs to pull `final_code` and `final_proof` from the last iteration.
2. **Add pipeline state persistence** — currently runs are only stored in audit logs. Add a `runs` table in SQLite so the API can return a list of past runs and their final states.
3. **Human-in-the-loop endpoint** — add `POST /api/v1/pipeline/{run_id}/approve` and `/reject` endpoints for the approval workflow.
4. **Error handling / retries** — the orchestrator catches exceptions but doesn't retry individual agent failures. Add retry logic with exponential backoff for transient LLM errors.
5. **WebSocket alternative to SSE** — SSE works but WebSockets would allow bidirectional communication (e.g., user sends feedback mid-run). Consider for stretch.

### Front-End (CLI + Dashboard)

1. **Test CLI against running backend** — start `uvicorn`, run `hpema generate`, verify the rich panels render correctly.
2. **Flesh out the Streamlit dashboard** — add traceability matrix view, diff view between iterations, code syntax highlighting.
3. **CLI approval flow** — after pipeline completes with `awaiting_approval`, prompt the user to approve/reject with optional feedback text.
4. **Export functionality** — `hpema export --run-id <uuid> --format pdf` to generate a compliance report document.

### ML (Models + RAG + Verification)

1. **Test with ARC endpoint** — set `HPEMA_API_KEY` to your ARC key, run the pipeline against `gpt-oss-120b`. This is the **first real end-to-end test**.
2. **Tune system prompts** — the prompts in `backend/services/llm/prompts/*.txt` are starting points. Iterate on them with real model outputs to get better code generation and reviews.
3. **Install Dafny on ARC** — test `dafny_runner.py` with simple specs (the `data/standards/` directory has examples of what the specs should look like). Write 3-5 sample `.dfy` files for testing.
4. **RAG testing** — install `chromadb` and `sentence-transformers`, run `ingest_standards()`, verify the retriever returns relevant results for queries like "dynamic memory allocation DO-178C".
5. **Constrained decoding validation** — when vLLM is running, test `guided_json` with our schemas. Verify the model's output is always valid JSON matching the Pydantic model. This is the "token-level safety" demo.
6. **vLLM serving on ARC** — use `ml/slurm/serve_models.sh` to launch model instances. Start with one model first (Phi-4 is smallest/fastest), verify it serves, then scale up.
7. **Prompt engineering iteration** — once models are serving, iterate on prompts to improve:
   - Actor: quality of generated Dafny specs
   - Checker: adversarial review quality, test case generation
   - Policy: accuracy of standard citations

### Testing Needed

| Test | What It Validates | Priority |
|---|---|---|
| End-to-end with ARC endpoint | Full pipeline works with real LLM | **P0** |
| Dafny verification | Formal proofs pass/fail correctly | **P0** |
| Constrained decoding | `guided_json` produces valid schemas | **P0** |
| RAG retrieval accuracy | Correct standards retrieved for queries | **P1** |
| SSE streaming | CLI receives and displays events correctly | **P1** |
| Multi-iteration convergence | Pipeline improves code across iterations | **P1** |
| Config swap | Changing `hpema_config.yaml` changes behavior | **P2** |
| Audit trail completeness | Every event is logged, nothing dropped | **P2** |

---

## How to Run Locally

```bash
# 1. Activate the venv
source .venv/bin/activate

# 2. Set your ARC API key
export HPEMA_API_KEY="your-arc-api-key"

# 3. Run tests
pytest tests/ -v

# 4. Start the backend
uvicorn backend.main:app --reload --port 8000

# 5. Check routes
open http://localhost:8000/docs

# 6. Run the CLI (once backend is running)
python -m cli.main generate \
    --requirement "Implement a binary search function with bounds checking" \
    --standard DO_178C \
    --language C
```

## How to Run on ARC

```bash
# 1. SSH into ARC
ssh <pid>@tinkercliffs1.arc.vt.edu

# 2. Request GPU node
salloc --partition=a100_normal_q --gres=gpu:1 --time=1:00:00

# 3. Start vLLM (single model for testing)
python -m vllm.entrypoints.openai.api_server \
    --model microsoft/Phi-4-Reasoning-Plus --port 8002 &

# 4. Override config to point at local vLLM
export HPEMA_CHECKER_ENDPOINT=http://localhost:8002/v1
export HPEMA_CHECKER_MODEL=microsoft/Phi-4-Reasoning-Plus

# 5. Start backend and test
uvicorn backend.main:app --port 8000 &
python -m cli.main generate --requirement "binary search" --standard DO_178C
```

---

## Key Design Decisions (For Team Context)

1. **Why custom orchestrator instead of LangGraph?** — Our pipeline is deterministic (Actor→Checker→Policy→loop). LangGraph adds complexity for free-form agent chats we don't need. Our orchestrator is ~180 lines, fully auditable.

2. **Why Pydantic schemas everywhere?** — Two reasons: (a) type safety between agents, (b) constrained decoding. vLLM can use these schemas to force the model to output valid JSON at the token level. This is one of our three differentiators.

3. **Why YAML config?** — Boeing plug-and-play. They change one file to swap models, standards, and prover. No code changes.

4. **Why dual-write audit (JSONL + SQLite)?** — JSONL is append-only and immutable (good for compliance evidence). SQLite enables fast cross-run queries (good for the dashboard). Belt and suspenders.

5. **Why ChromaDB for RAG?** — Simple, embeds in-process, persists to disk. Right for a capstone. If Boeing needs scale, they swap to Pinecone/Weaviate — same retriever interface.
