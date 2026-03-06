# HPEMA ML Architecture Plan

## Context

The HPEMA (Hierarchical Policy-Enforced Multi-Agent) capstone project needs a complete architectural blueprint for the agentic ML pipeline. The project currently has documentation (proposal, flowchart, model selection rationale) and a FastAPI stub, but zero agent implementation. This plan defines the full architecture for the ML/agentic side: how agents are orchestrated, how models are served and aligned, how formal verification integrates, and how it all gets showcased.

---

## 0. What Makes HPEMA Novel — The Three Guarantees

Existing agentic coding tools (Copilot, Devin, SWE-Agent, CrewAI pipelines) share a fundamental weakness: they **hope** the LLM behaves. HPEMA **guarantees** it at three independent levels:

### Guarantee 1: Token-Level Structural Safety (Constrained Decoding)

Every agent in HPEMA produces output through vLLM's `guided_json` — a constrained decoding mode that enforces a JSON schema **at the token level** during generation. The model literally cannot emit tokens that would violate the output schema. This is not prompt engineering or post-hoc parsing. It is a formal grammar constraint on the decoder.

**Why this matters for Boeing:** In safety-critical systems, "the model usually follows the format" is unacceptable. Constrained decoding provides a structural guarantee that every agent output is machine-parseable and schema-compliant — zero chance of malformed inter-agent communication causing silent failures.

**Demo moment:** Show the raw constrained decoding in action — the model's token probabilities being masked in real-time to enforce the schema. Compare to an unconstrained run where the same model occasionally produces unparseable output.

### Guarantee 2: Mathematical Proof of Correctness (Dafny in the Loop)

The Actor agent generates code **and** a formal specification (Dafny annotations: `requires`, `ensures`, invariants). The Dafny theorem prover then mathematically verifies that the code satisfies its specification. If verification fails, the solver's counterexample feeds back into the next iteration.

No existing agentic coding tool does this. They test code (unit tests, fuzzing) — but testing can only show the presence of bugs, never their absence. Formal verification **proves** correctness for all inputs.

**Why this matters for Boeing:** DO-178C Level A (flight-critical software) already requires formal methods for the highest assurance levels. HPEMA automates the generation of both the code and its proof — collapsing what currently takes weeks of manual effort into minutes.

**Demo moment:** Generate an altitude hold controller, show the Dafny proof that altitude stays within bounds for ALL possible inputs, not just tested ones. Then introduce a deliberate bug — show Dafny catch it with a concrete counterexample that testing missed.

### Guarantee 3: Live Policy Enforcement with Standard Citations (RAG-Backed Audit)

The Policy Agent doesn't just "review" code — it retrieves specific clauses from safety standards (DO-178C, MISRA C, NASA-STD-8739.8) via RAG and cites them in its compliance verdict. Each violation references the exact standard section, creating audit-ready evidence.

**Why this matters for Boeing:** Standards change. Boeing has internal SDPs that evolve. With HPEMA, updating the knowledge base updates the enforcement — no retraining, no code changes. Boeing drops their SDP documents into the knowledge base directory, and the Policy Agent starts enforcing them immediately.

**Demo moment:** Run the same code generation against DO-178C, then swap to MISRA C — show how different rules trigger different violations with different citations. Then show adding a custom Boeing rule document and seeing it enforced on the next run.

### The Compound Effect

Any one of these is interesting. All three together create **defense in depth for AI-generated code** — a concept Boeing already understands from systems engineering. The output is structurally valid (constrained decoding), logically correct (formal proof), and policy-compliant (RAG audit). Each layer catches what the others miss.

---

## 1. Plug-and-Play Design Philosophy

HPEMA is designed so that Boeing (or any organization) can take the pipeline and plug in their own models, standards, and infrastructure with **zero code changes** — configuration only.

### Single Configuration File

```yaml
# hpema_config.yaml — the only file an organization needs to customize
models:
  actor:
    endpoint: "https://internal-llm.boeing.com/v1"
    model: "boeing-codegen-v2"
    api_key_env: "BOEING_LLM_KEY"          # env var name for the API key
  checker:
    endpoint: "https://internal-llm.boeing.com/v1"
    model: "boeing-reviewer-v1"
    api_key_env: "BOEING_LLM_KEY"
  policy:
    endpoint: "https://internal-llm.boeing.com/v1"
    model: "boeing-policy-v1"
    api_key_env: "BOEING_LLM_KEY"

policies:
  standards_dir: "/path/to/boeing/sdp/documents"   # drop PDFs/markdown here
  default_standard: "Boeing_SDP"
  embedding_model: "all-MiniLM-L6-v2"              # or Boeing's internal embedder

verification:
  prover: "dafny"                                   # or "lean4"
  timeout_seconds: 120
  binary_path: "/usr/local/bin/dafny"               # custom install path

pipeline:
  max_iterations: 3
  require_human_approval: true
  audit_log_dir: "/secure/audit/logs"
```

### What's Swappable

| Component | How to Swap | Config Only? |
|---|---|---|
| LLM models | Change `endpoint` + `model` in YAML | Yes |
| Safety standards | Drop documents into `standards_dir` | Yes |
| API authentication | Set env var named in `api_key_env` | Yes |
| Formal prover | Change `prover` field | Yes |
| Audit storage | Change `audit_log_dir` | Yes |
| Embedding model | Change `embedding_model` | Yes |
| Custom agent prompts | Override files in `prompts/` directory | Yes |

### Why Open-Source Models

We use open-source models (Qwen, Phi-4, Mistral) specifically to demonstrate that HPEMA works with **any** model behind an OpenAI-compatible API. Boeing sees: "this works with open models on commodity hardware — it will work with our proprietary models on our infrastructure." The architecture doesn't depend on any single model's capabilities; the three guarantees (constrained decoding, formal verification, RAG policy) work regardless of which model generates the tokens.

---

## 2. User Interface: Rich CLI + Streamlit Demo Dashboard

**Primary interface: CLI using `typer` + `rich`** (matches "CLI-first" in proposal, fits Boeing engineer personas, resembles Claude Code).

**Secondary: Streamlit dashboard** for capstone demo/audit viewing — reads from the same audit log. No heavy frontend needed.

```
hpema generate --requirement "Implement altitude hold controller" \
               --standard DO_178C --language SPARK_Ada --max-iterations 3
hpema audit    --run-id <uuid>
hpema dashboard                     # Launch Streamlit viewer
```

The CLI streams real-time agent reasoning panels (which agent is active, its chain-of-thought, pass/fail status) via Server-Sent Events from the backend.

---

## 3. Agent Orchestration — Core Pipeline

### Why Custom (not LangGraph/CrewAI)

HPEMA is a **structured pipeline**, not a free-form agent chat. The control flow is deterministic (Actor → Checker → Policy → loop); only the LLM outputs are non-deterministic. A custom orchestrator (~300 lines) is simpler, fully auditable, and gives total control over the feedback loop — critical for aerospace traceability.

### Data Flow

```
User Request (CLI / API)
       │
       ▼
┌──────────────────┐
│   Orchestrator    │  Creates PipelineRun (UUID), loads policy context via RAG
│  (state machine)  │
└──────────────────┘
       │
       ▼
┌──────────────────┐
│  Context Agent    │  (Optional) DeepSeek-V3.2 — scans repo, produces context summary
│  skip in MVP      │
└──────────────────┘
       │
       ▼
╔══════════════════════════════════════════════════╗
║         ACTOR-CHECKER LOOP (max 3 iters)         ║
║                                                  ║
║  ┌────────────────────┐                          ║
║  │   Actor Agent       │  Qwen3-Coder-Next       ║
║  │   → CodeCandidate   │  (code + Dafny spec)     ║
║  └────────┬───────────┘                          ║
║           │                                      ║
║     ┌─────┴──────┐                               ║
║     ▼            ▼         (parallel)            ║
║  ┌──────────┐ ┌──────────────┐                   ║
║  │ Checker  │ │ Dafny/Lean   │                   ║
║  │ Agent    │ │ Verifier     │                   ║
║  │ (Phi-4)  │ │ (subprocess) │                   ║
║  └────┬─────┘ └──────┬──────┘                   ║
║       └──────┬───────┘                           ║
║              ▼                                   ║
║  ┌────────────────────┐                          ║
║  │  Policy Agent       │  Mistral-Small-24B       ║
║  │  + RAG (Boeing KG)  │  → PolicyVerdict         ║
║  └────────┬───────────┘                          ║
║           │                                      ║
║     [ALL PASS?]──No──▶ compose feedback, loop     ║
║           │                                      ║
║          Yes                                     ║
╚══════════════════════════════════════════════════╝
       │
       ▼
┌──────────────────┐
│ Human Checkpoint  │  CLI: approve / reject with feedback
└──────────────────┘
       │
       ▼
  Output Artifacts: Code + Tests + Proof + Audit Log + Compliance Report
```

**Key latency optimization:** Checker Agent and Dafny Verifier run in parallel via `asyncio.gather` — they are independent. This cuts ~50% off each iteration's wall time.

### Inter-Agent Communication: Structured Pydantic Messages

Agents do NOT exchange free-text. Each produces a typed Pydantic model, enforced at the token level via vLLM's `guided_json` (constrained decoding). This is how models stay "aligned" — schema enforcement, not hope.

```python
CodeCandidate       # Actor output: source_code, dafny_spec, reasoning_trace, annotations
CheckerReport       # Checker output: test_cases, issues_found, suggested_fixes, verdict
VerificationResult  # Dafny output: verified (bool), solver_output, failing_assertions
PolicyVerdict       # Policy output: compliant (bool), violations, standard_references, risk_level
PipelineState       # Full run state: request, iterations[], status, final_artifacts
```

### Convergence Logic

- Max 3 iterations (research shows loops rarely converge after 3 if they haven't already)
- Structured feedback composition: orchestrator builds a `FeedbackMessage` from Checker, Verifier, and Policy results — prioritizes critical failures
- If max iterations reached without convergence → status = FAILED, full audit trail preserved

---

## 4. Model Serving on VT ARC

### Architecture: One vLLM Instance Per Model

```
ARC Infer Node (A100/H100 GPUs)
├── Port 8001: vLLM → Qwen3-Coder-Next (80B MoE, 2x A100-80GB)
├── Port 8002: vLLM → Phi-4-Reasoning-Plus (14B, 1x A100)
├── Port 8003: vLLM → Mistral-Small-3.2-24B (1x A100)
└── Port 8004: vLLM → DeepSeek-V3.2 (4x A100, stretch goal)

ARC TinkerCliffs (CPU nodes)
└── Dafny / Lean 4 verification jobs
```

Each vLLM server exposes an OpenAI-compatible API. The system uses the **same async OpenAI client** for all agents — only the endpoint URL and model name change.

### Live Demo Strategy (No RunPod Needed)

Since we have full ARC HPC access, the entire demo runs on-cluster — no paid providers required. This strengthens the Boeing narrative (air-gapped, on-prem deployment).

**Demo setup:** One interactive SLURM session on Infer partition:
1. Launch 3 vLLM servers (Actor, Checker, Policy) as background processes on allocated GPUs
2. Start FastAPI backend on the same node
3. Run the CLI — it connects to `localhost` ports
4. Everything runs within ARC's network. SSH into the node, run the demo.

```bash
# Example: interactive session with 4 GPUs
salloc --partition=a100_normal_q --gres=gpu:4 --time=2:00:00

# Inside the session, launch vLLM servers as background processes
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-Coder-Next --port 8001 --tensor-parallel-size 2 &
python -m vllm.entrypoints.openai.api_server --model microsoft/Phi-4-Reasoning-Plus --port 8002 &
python -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-Small-3.2-24B-Instruct --port 8003 &

# Start backend
uvicorn backend.main:app --port 8000 &

# Run CLI demo
hpema generate --requirement "Implement altitude hold controller" --standard DO_178C
```

For development/testing, use the shared ARC endpoint (`gpt-oss-120b`) for all agents with different system prompts — override via env vars when vLLM is ready:

```bash
HPEMA_ACTOR_ENDPOINT=http://localhost:8001/v1
HPEMA_ACTOR_MODEL=Qwen/Qwen3-Coder-Next
```

### Model Registry

The model registry loads from `hpema_config.yaml` (see Section 1) and can be overridden by environment variables. This is the core of the plug-and-play design — Boeing changes the YAML, not the code.

```python
# backend/services/llm/model_registry.py
# Loads from hpema_config.yaml, with env var overrides (HPEMA_ACTOR_ENDPOINT, etc.)
class ModelConfig(BaseSettings):
    actor_endpoint: str = "https://llm-api.arc.vt.edu/api/v1"
    actor_model: str = "gpt-oss-120b"         # MVP default (ARC shared endpoint)
    checker_endpoint: str = "https://llm-api.arc.vt.edu/api/v1"
    checker_model: str = "gpt-oss-120b"
    policy_endpoint: str = "https://llm-api.arc.vt.edu/api/v1"
    policy_model: str = "gpt-oss-120b"
    class Config:
        env_prefix = "HPEMA_"
```

The async OpenAI client wrapper uses this registry so every agent talks through the same interface — swap the endpoint, swap the model.

---

## 5. RAG & Knowledge Graph

**Vector store: ChromaDB** (embeds in-process, persists to disk, simple API — right for a capstone).

**Embedding model:** `all-MiniLM-L6-v2` (runs on CPU, fast).

### What Goes in the Knowledge Base

For the capstone (no real Boeing data), create a realistic mock KB from public sources:
- DO-178C sections (from FAA advisory circulars)
- NASA-STD-8739.8 coding standards (public)
- MISRA C guidelines (public subset)
- 50–100 synthetic "Boeing Design Practices" based on the above

### Integration with Policy Agent

The Policy Agent's system prompt includes retrieved context from ChromaDB before every audit call. The retriever filters by `safety_standard` metadata to return only relevant rules.

---

## 6. Formal Verification Integration

### Dafny Runner

- Actor generates code WITH Dafny annotations (`requires`, `ensures`, invariants)
- `DafnyRunner` writes spec to temp `.dfy` file, runs `dafny verify` as async subprocess with timeout
- Parses output for pass/fail + failing assertion details
- Results feed back into the orchestrator's feedback composition

### MVP Scope

Start with simple Dafny specs: binary search, array bounds, altitude controller with range constraints. The system proves the architecture works even with small examples.

---

## 7. Audit & Traceability

### Append-Only Audit Log

Every agent call, verification result, and policy decision is logged to:
1. **JSONL file** per run (`logs/audit/{run_id}.jsonl`) — immutable append-only
2. **SQLite database** (`data/audit.db`) — for querying across runs

### Traceability Matrix

Generated per run: `requirement → code artifact → test cases → formal proof → policy verdict`. This is the artifact Boeing/FAA auditors need.

---

## 8. File Structure

```
Capstone/
├── hpema_config.yaml                 # Plug-and-play configuration (models, standards, prover)
├── cli/
│   ├── main.py                       # typer app, commands
│   ├── display.py                    # rich panels for streaming
│   └── config.py                     # CLI config
│
├── backend/
│   ├── main.py                       # FastAPI app (EXISTS)
│   ├── config.py                     # NEW: Pydantic Settings, central config
│   ├── api/
│   │   ├── routers/
│   │   │   ├── generation.py         # EXISTS — wire to orchestrator
│   │   │   ├── health.py             # EXISTS
│   │   │   ├── pipeline.py           # NEW: SSE streaming, pipeline start
│   │   │   └── audit.py              # NEW: audit trail queries
│   │   ├── schemas/
│   │   │   ├── generation.py         # EXISTS — expand
│   │   │   ├── agents.py             # NEW: CodeCandidate, CheckerReport, etc.
│   │   │   ├── pipeline.py           # NEW: PipelineState, StreamEvent
│   │   │   └── audit.py              # NEW: audit query schemas
│   │   └── dependencies.py           # NEW: DI for services
│   │
│   ├── services/                     # NEW: all business logic
│   │   ├── orchestrator.py           # Pipeline state machine (~300 LOC)
│   │   ├── feedback.py               # Feedback composition
│   │   ├── agents/
│   │   │   ├── base.py               # BaseAgent ABC
│   │   │   ├── actor.py              # Actor (Qwen3)
│   │   │   ├── checker.py            # Checker (Phi-4)
│   │   │   ├── policy.py             # Policy (Mistral-Small)
│   │   │   └── context.py            # Context (DeepSeek, stretch)
│   │   ├── llm/
│   │   │   ├── client.py             # Async OpenAI wrapper
│   │   │   ├── model_registry.py     # Agent → endpoint mapping
│   │   │   └── prompts/              # System prompts per agent
│   │   ├── verification/
│   │   │   ├── dafny_runner.py       # Subprocess wrapper
│   │   │   └── sandbox.py            # Execution sandboxing
│   │   ├── rag/
│   │   │   ├── retriever.py          # ChromaDB vector search
│   │   │   └── knowledge_graph.py    # Standards ingestion
│   │   └── audit/
│   │       ├── logger.py             # Append-only JSONL + SQLite
│   │       └── traceability.py       # Req→Code→Test→Proof matrix
│   │
│   └── data/                         # Persistence
│       ├── database.py               # SQLite connection
│       ├── repositories/             # CRUD for runs, audit, artifacts
│       └── models.py                 # ORM models
│
├── ml/                               # Training & artifacts (NOT runtime)
│   ├── ARCHITECTURE.md               # THIS PLAN
│   ├── slurm/                        # SLURM batch scripts for vLLM serving
│   ├── training/                     # LoRA fine-tuning scripts (stretch)
│   ├── data/                         # Training data, KG documents
│   └── eval-outputs/                 # Evaluation results
│
├── dashboard/
│   └── app.py                        # Streamlit audit viewer (demo)
│
└── tests/
    ├── unit/                         # Mocked LLM tests per agent
    └── integration/                  # End-to-end pipeline tests
```

---

## 9. MVP vs Stretch

| Component | MVP | Stretch |
|---|---|---|
| Models | Single ARC endpoint (`gpt-oss-120b`), different system prompts | 4 dedicated vLLM instances |
| Knowledge base | 50–100 synthetic rules from public standards | Real Boeing KG, LoRA fine-tuning |
| Formal verification | Real Dafny on simple specs | Lean 4, SLURM job submission |
| Context Agent | Skipped | DeepSeek repo scanning |
| Dashboard | Skip or basic | Full Streamlit audit viewer |
| Human-in-the-loop | CLI y/n prompt | Approval workflow with feedback |

---

## 10. Implementation Order

1. **Foundation** — `config.py`, `llm/client.py`, `model_registry.py`, all Pydantic schemas in `agents.py`
2. **Agents** — `base.py`, `actor.py`, `checker.py`, `policy.py` with system prompts
3. **Orchestrator** — `orchestrator.py` state machine, feedback composition
4. **Verification** — Install Dafny, `dafny_runner.py`, test with simple specs
5. **RAG** — ChromaDB setup, ingest synthetic standards, wire into Policy Agent
6. **Audit** — `logger.py`, `traceability.py`, SQLite persistence
7. **CLI** — `typer` + `rich` commands, SSE streaming from backend
8. **Polish** — 3–5 demo scenarios, Streamlit dashboard, documentation

---

## 11. Verification / Testing Plan

- **Unit tests**: Mock LLM responses, verify each agent parses/produces correct Pydantic models
- **Integration test**: End-to-end pipeline with a simple requirement (e.g., "binary search with bounds checking"), verify all artifacts generated
- **Dafny test**: Verify a known-good `.dfy` spec passes and a known-bad one fails
- **RAG test**: Ingest test documents, verify retrieval returns relevant results for a policy query
- **Demo scenarios**: (1) Generate altitude hold controller with DO-178C compliance, (2) Generate binary search with formal proof, (3) Audit existing code against Boeing SDP
