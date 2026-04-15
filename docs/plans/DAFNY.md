# Dafny Formal Verification — Integration Plan

**Last updated:** 2026-04-09

---

## Context

Dafny integration is HPEMA's "Guarantee 2: Mathematical Proof of Correctness," but it is currently nonfunctional. The 7B Actor model (Qwen2.5-Coder-7B) cannot produce valid Dafny. Half the time it produces nothing; when it does, the specs are syntactically broken (wrong types, extern refs, missing bodies). The 5 verified examples in `data/dafny_examples/` are never shown to the model. The feedback loop is disabled in config. And `client.py` was reverted to remove the constrained-decoding fallback — so API models that reject `json_schema` response_format will crash.

### Root Causes

1. **Model incapacity**: 7B models don't know Dafny syntax reliably. This is the dominant failure mode.
2. **No few-shot examples**: The 5 verified programs in `data/dafny_examples/` are never injected into any prompt.
3. **Single-shot generation**: The Actor is asked to produce both C code and Dafny spec simultaneously — two fundamentally different tasks.
4. **Feedback loop disabled**: `hpema_config.arc.yaml` sets `stage: "actor"`, forcing `effective_max_iterations` to 1 (orchestrator.py line 103-106). Even at `stage: "checker"`, the feedback for Dafny failures is just raw assertion text — not enough guidance.
5. **No syntax pre-check**: Broken Dafny goes straight to `dafny verify`, which produces parser errors that don't match the 5 regex patterns in `_parse_failing_assertions`. Feedback is lost silently.
6. **No constrained-decoding fallback**: `client.py` was reverted — any API model that rejects `json_schema` crashes.

### Key Observation

Dafny is simplest for **pure functions without loops**. The existing examples prove this: AbsoluteValue, Clamp, SafeAdd all verify trivially. BinarySearch (with loops + invariants + decreases) is the hard case. The plan prioritizes getting pure functions working first.

---

## Phase 1: Make It Work (prompt + model + config)

Highest impact for lowest effort. Target: valid specs for pure-function requirements.

**Estimated effort:** 1–2 days.

### 1A. Inject few-shot Dafny examples into Actor prompt

**Files:** `backend/services/agents/actor.py`, `backend/services/llm/prompts/actor.txt`

In `actor.py._build_user_prompt()`, after the feedback section, load the PASS-only variants from 3 example files and append as a `## Dafny Specification Examples` section. Strip everything after the first `// FAIL:` line in each file.

Examples to include (small, loop-free, all verify):
- `data/dafny_examples/absolute_value.dfy` — simplest (pure function, no branching)
- `data/dafny_examples/clamp.dfy` — branching + range postconditions
- `data/dafny_examples/safe_add.dfy` — overflow detection, dual return values

Skip binary_search (loops are too hard for Phase 1) and altitude_hold (redundant with clamp).

Add a helper `_load_dafny_examples() -> str` that reads and strips these files, cached on first call.

### 1B. Rewrite Dafny instructions in actor.txt

The current prompt says "Dafny annotations (requires/ensures/invariants)" which leads models to produce annotation-style output (just requires/ensures without method bodies). Dafny cannot verify annotations without a method body.

Replace with concrete rules:

```
DAFNY SPECIFICATION RULES:
- Write a STANDALONE Dafny method with a full body — not just annotations
- Use `method Name(params) returns (result: Type)` syntax
- Use Dafny-native types ONLY: int, bool, array<int>, seq<T>. NEVER use C types
- Add `requires` (preconditions) and `ensures` (postconditions)
- For loops: add `invariant` AND `decreases` clauses — Dafny rejects loops without them
- The spec must verify by itself — no {:extern}, no imports, no modules
- Mirror your source_code logic inside the Dafny method body
```

### 1C. Upgrade Actor model to 32B

**Files:** `hpema_config.arc.yaml`, `ml/slurm/start_tmux.sh`

On 8×A100 (640GB total VRAM), serve `Qwen/Qwen2.5-Coder-32B-Instruct` via vLLM on port 8002 with tensor parallelism (`--tensor-parallel-size 4`). Point Actor at it. Keep 7B on port 8001 for Checker/Policy (adequate for review tasks).

```yaml
models:
  actor:
    endpoint: "http://localhost:8002/v1"
    model: "Qwen/Qwen2.5-Coder-32B-Instruct"
    api_key_env: "HPEMA_API_KEY"
  checker:
    endpoint: "http://localhost:8001/v1"
    model: "Qwen/Qwen2.5-Coder-7B-Instruct"
    api_key_env: "HPEMA_API_KEY"
  policy:
    endpoint: "http://localhost:8001/v1"
    model: "Qwen/Qwen2.5-Coder-7B-Instruct"
    api_key_env: "HPEMA_API_KEY"
```

Update `start_tmux.sh` to launch two vLLM instances:
- Pane 0: 32B model on 4 GPUs (`--tensor-parallel-size 4 --port 8002`)
- Pane 1: 7B model on 1 GPU (`--port 8001`)
- Pane 2: CLI

### 1D. Enable the feedback loop

**File:** `hpema_config.arc.yaml`

Change `stage: "actor"` → `stage: "checker"` (at minimum) or `stage: "policy"`. Without this, `effective_max_iterations` is forced to 1 and the Checker + Dafny verification step never runs.

### 1E. Restore constrained-decoding fallback in client.py

**File:** `backend/services/llm/client.py`

The file was reverted to a version with no fallback. `generate_structured` calls `generate` with `response_schema`, which always sends `response_format: json_schema`. This crashes on any API that rejects it (Nvidia, Groq, etc.).

Restore:
- `_no_constrained_decoding: set[str]` on `LLMClient` — tracks endpoints that don't support it
- `BadRequestError` catch in `generate()` that retries with schema-in-prompt fallback
- `_extract_json()` — extracts JSON from markdown fences or prose
- `_example_from_model()` — builds human-readable placeholder JSON instead of raw schema (small models echo raw schemas back)
- `_is_schema_echo()` — detects when model returns the schema itself instead of an instance
- Multi-level parse in `generate_structured`: direct → extract from fences → partial parse
- `aclose()` method — closes httpx connection pools before event loop shuts down (fixes "Event loop is closed" errors)

**This is prerequisite for any API-based model to work.**

---

## Phase 2: Make It Reliable (structural separation)

Target: ~90% syntactic validity on generated specs. Decouples Dafny from the Actor.

**Estimated effort:** 3–5 days.

### 2A. Dedicated DafnyArchitect agent

Instead of asking the Actor to produce both C code and Dafny spec, split the responsibility.

**New files:**
- `backend/services/agents/dafny_architect.py` — new agent class extending `BaseAgent`
- `backend/services/llm/prompts/dafny_architect.txt` — specialized Dafny-only prompt with baked-in few-shot examples

**Modified files:**
- `backend/config.py` — add `dafny_architect: ModelEndpointConfig` to `ModelsConfig`
- `backend/services/llm/model_registry.py` — add `dafny_architect` property + mapping entry in `get()`
- `backend/services/orchestrator.py` — insert DafnyArchitect step between Actor and Checker
- `backend/api/schemas/agents.py` — add `DafnySpec(BaseModel)` with `dafny_source: str, reasoning_trace: str`

**Pipeline flow change:**
```
Actor(requirement) → source_code
  ↓
DafnyArchitect(source_code, requirement) → dafny_spec
  ↓
[Checker(source_code) || DafnyRunner(dafny_spec)]  (parallel)
  ↓
Policy(source_code, dafny_spec, checker_report, verification_result)
```

The DafnyArchitect receives the Actor's `source_code` + the requirement + few-shot examples, and produces a standalone Dafny method. The Actor's prompt can then be simplified to remove all Dafny instructions — it focuses purely on generating correct source code.

The Architect's prompt bakes in the 3 PASS examples permanently (not dynamically loaded — they're always relevant for Dafny syntax guidance).

**Model options for DafnyArchitect:**
- Same 32B model used for Actor (simplest, no extra VRAM)
- Nvidia free inference API for a 70B+ model (e.g., `nvidia/llama-3.1-nemotron-ultra-253b-v1`) — maximum Dafny capability, no local VRAM cost
- The config supports per-layer endpoints, so this is just a YAML change

### 2B. Dafny syntax pre-check

**File:** `backend/services/verification/dafny_runner.py`

Add `_syntax_precheck(source: str) -> list[str]` that catches obvious LLM mistakes before invoking the subprocess. This saves time and produces cleaner error messages:

```python
def _syntax_precheck(self, source: str) -> list[str]:
    errors = []
    if "extern" in source.lower():
        errors.append("Spec must be self-contained (no {:extern} attributes)")
    if "method" not in source and "function" not in source:
        errors.append("Spec must contain at least one method or function")
    if not re.search(r'\b(ensures|requires)\b', source):
        errors.append("Spec must have at least one requires or ensures clause")
    c_types = re.findall(r'\b(int32_t|uint8_t|size_t|void|char\s*\*)\b', source)
    if c_types:
        errors.append(f"Found C types in Dafny spec: {c_types}. Use Dafny types (int, bool, array<int>)")
    return errors
```

Call at the top of `verify()`. If errors found, return `VerificationResult(verified=False, failing_assertions=errors)` immediately — skip the subprocess.

### 2C. Expand DafnyRunner error pattern coverage

**File:** `backend/services/verification/dafny_runner.py`

`_parse_failing_assertions()` only matches 5 semantic patterns. Parser errors from syntactically broken Dafny are silently dropped. Add:

```python
r"Error:.*semi expected.*",
r"Error:.*invalid.*",
r"Error:.*unexpected token.*",
r"Error:.*type mismatch.*",
r"Error:.*unresolved identifier.*",
```

### 2D. Human-readable Dafny error hints in feedback

**File:** `backend/services/feedback.py`

The current feedback for Dafny failures is: `FORMAL VERIFICATION FAILED. Failing assertions: <raw dafny output>`. This is nearly useless for the LLM.

Map Dafny error patterns to actionable natural-language hints:

```python
DAFNY_HINTS = {
    "postcondition might not hold":
        "Check that every return path establishes the ensures clause.",
    "precondition could not be proved":
        "The caller doesn't establish the callee's requires clause.",
    "invariant might not be maintained":
        "The loop body breaks the invariant. Strengthen or fix it.",
    "decreases might not decrease":
        "The loop variant doesn't strictly decrease each iteration.",
    "semi expected":
        "Syntax error — missing semicolons or malformed statement.",
    "type mismatch":
        "Wrong types used — ensure all types are Dafny-native (int, bool, array<int>).",
    "unresolved identifier":
        "Undefined name — ensure all variables are declared and methods are self-contained.",
}
```

In `compose_feedback()`, iterate over `failing_assertions`, match against these patterns, and append the hint:
```
FORMAL VERIFICATION FAILED.
- Error: "postcondition might not hold on line 5"
  Hint: Check that every return path establishes the ensures clause.
```

---

## Phase 3: Stretch Goals (if time permits)

### 3A. DafnyBench few-shot retrieval via RAG

**Estimated effort:** 2–3 days.

Download a subset of DafnyBench (750+ verified programs), ingest ~50 curated ones into ChromaDB alongside the existing safety standards. Retrieve the 2 most relevant verified programs as few-shot examples based on the requirement text.

**Files:**
- `backend/services/rag/retriever.py` — new `DafnyExampleRetriever` class or second ChromaDB collection `"dafny_examples"`
- New directory: `data/dafny_bench/` — curated subset covering common patterns (pure functions, loops, arrays, recursion)
- `backend/services/agents/dafny_architect.py` — in `_build_user_prompt()`, call retriever for 2 most relevant examples

### 3B. Dafny repair mini-loop

**Estimated effort:** 2–3 days.

When `DafnyRunner.verify()` returns `verified=False`, invoke a specialized repair agent (2 attempts max) before falling back to the main feedback loop. The repair agent receives the failing Dafny source + Z3 error messages and produces a corrected version.

**New files:**
- `backend/services/agents/dafny_repair.py`
- `backend/services/llm/prompts/dafny_repair.txt`

**Modified:** `orchestrator.py` — inner repair loop after Dafny verification:

```python
if not verification_result.verified:
    for repair_attempt in range(2):
        repaired = await self._dafny_repair.run(
            dafny_source=code_candidate.dafny_spec,
            errors=verification_result.failing_assertions,
            solver_output=verification_result.solver_output,
        )
        verification_result = await self._dafny.verify(repaired.dafny_source)
        if verification_result.verified:
            code_candidate.dafny_spec = repaired.dafny_source
            break
```

Point `dafny_repair` at a reasoning model (DeepSeek-R1-Distill-32B) or the same 32B used for the Architect.

### 3C. Logic-drift diff-checker

**Estimated effort:** 2 days.

When the repair agent modifies a Dafny spec, functional behavior might change. Compile both original + repaired spec to Python via `dafny build --target:py`, run differential tests to ensure outputs match for identical inputs. Only worth pursuing if the repair agent is operational and showing drift.

---

## Implementation Sequence

| # | Task | Phase | Files | Dependencies |
|---|------|-------|-------|-------------|
| 1 | Restore client.py fallback | 1E | `client.py` | None (do first) |
| 2 | Few-shot examples in Actor prompt | 1A | `actor.py`, `actor.txt` | Parallel with 1,3,4 |
| 3 | Dafny rules in actor.txt | 1B | `actor.txt` | Parallel with 1,2,4 |
| 4 | Config: 32B model + stage: checker | 1C+1D | `hpema_config.arc.yaml`, `start_tmux.sh` | Parallel with 1,2,3 |
| 5 | DafnyRunner: pre-check + parser patterns | 2B+2C | `dafny_runner.py` | After 1–4 |
| 6 | Feedback hints | 2D | `feedback.py` | After 5 |
| 7 | DafnyArchitect agent | 2A | 6 files (see 2A) | After 1–6 |
| 8 | DafnyBench RAG | 3A | `retriever.py`, data/ | After 7 |
| 9 | Repair agent | 3B | 3 files | After 7 |
| 10 | Diff-checker | 3C | new verification util | After 9 |

**Tasks 1–4:** One sitting (parallel).
**Tasks 5–6:** One sitting.
**Task 7:** Largest single change (new agent + orchestrator rewiring).
**Tasks 8–10:** Independent stretch goals.

---

## Token Budget Analysis

The Actor prompt (`actor.txt`) is ~350 tokens. Adding 3 few-shot Dafny examples adds ~300–400 tokens. With the user prompt (requirement + language + standard + feedback), total input is ~1500–2500 tokens. With `max_tokens=2000` for output, this fits within 8K context. The 32B Qwen model supports 32K context — no pressure.

If Phase 2A introduces DafnyArchitect, the Actor prompt can be simplified by removing all Dafny instructions, making it shorter and more focused.

---

## Verification Criteria

1. **Phase 1 smoke test:** Simple requirement like "Write a function that returns the absolute value of an integer":
   - Actor produces non-empty `dafny_spec`
   - `dafny verify` succeeds (or produces parseable errors, not garbage)
   - Feedback loop runs 2+ iterations if first attempt fails

2. **Phase 2 validation:** 10 diverse requirements (pure functions, branching, overflow):
   - 90%+ runs produce syntactically valid Dafny
   - 50%+ specs verify successfully for pure functions

3. **Regression:** Existing unit tests in `tests/unit/test_dafny_runner.py` must pass.

---

## Model Recommendations (for reference)

From the architecture doc and DAFNY.md research:

| Role | Recommended Model | Rationale |
|------|------------------|-----------|
| **Actor** | Qwen-2.5-Coder-32B-Instruct | Best open-weights instruction-following for Dafny syntax. Fits on 4×A100 with TP. |
| **DafnyArchitect** | Same 32B or Nvidia API (70B+) | Dedicated Dafny specialist. API avoids local VRAM pressure. |
| **DafnyRepair** (stretch) | DeepSeek-R1-Distill-32B | Exceptional chain-of-thought reasoning for debugging Z3 errors. |
| **Checker/Policy** | Qwen-2.5-Coder-7B-Instruct | Adequate for code review and compliance checking. 1 GPU. |

---

## References

- **DafnyPro (POPL 2026):** LLM-Assisted Automated Verification — Pruner + Hint-Augmentation loops
- **Clover (2024/2025):** Benchmark for "Vericoding" — simultaneous code/proof generation
- **DafnyBench:** Dataset of 750+ verified programs — few-shot prompting source
- **DafnySynth:** Fine-tuning dataset for invariant generation
- **NASA-STD-8739.8:** Explicitly recommends formal methods for Class A software
- **DO-178C §6.4 + DO-333:** Formal methods supplement — formal proofs for all possible inputs
