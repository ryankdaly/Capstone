# Plan: Claude-Code-Style CLI on HPC Node

> **Status as of 2026-03-30:** Phases 1-3 COMPLETE. Phase 4 partially done (error resilience, startup banner done; demo script pending).

## Goal

Build a polished, interactive CLI that **looks and feels like Claude Code** — a terminal-native agentic interface where a user types a requirement, watches agents think in real-time, and gets verified code out the other end. Runs entirely on a reserved HPC GPU node (L40S or A100) with vLLM serving Phi-4. Full constrained decoding, no compromises.

This is for **mentor showcase** — needs to look impressive, work reliably, and tell the HPEMA story.

---

## What Claude Code Looks Like (Our Reference)

Key UX patterns we're mimicking:

1. **Interactive prompt** — user types freely, not just CLI flags
2. **Live streaming** — tokens appear as they're generated, not after completion
3. **Activity indicators** — spinners/status lines showing which agent is active
4. **Collapsible panels** — agent reasoning shown in expandable Rich panels
5. **Tool use display** — when an agent "uses a tool" (Dafny verification, RAG lookup), it shows as a distinct action with status
6. **Color-coded verdicts** — green pass, red fail, yellow warn
7. **Persistent context** — the CLI remembers the current run and lets you drill into it

---

## Current State vs Target

| Feature | Current State | Target |
|---|---|---|
| Entry point | `python -m cli.main generate --requirement "..."` | Interactive REPL: `hpema` then type naturally |
| Streaming | SSE parsed line-by-line, rendered as static panels | Live animated spinners + completed panels |
| Agent display | Basic Rich panels with key-value dumps | Claude-Code-style panels: spinner while active, verdict + content on completion |
| Code output | Not displayed (buried in pipeline state) | Syntax-highlighted code block with language tag |
| Iteration tracking | Single "iteration complete" panel | Progress bar + iteration-by-iteration diff |
| Error handling | Generic error string | Formatted error with suggestion for next steps |
| Constrained decoding | Working via vLLM `response_format` | Same — full json_schema enforcement on HPC |

---

## Runtime Architecture

Everything runs on the same HPC GPU node. No external APIs, no network auth.

```
HPC GPU Node (L40S / A100)
├── tmux pane 0: vLLM (Phi-4, localhost:8001, --structured-outputs-config.backend outlines)
└── tmux pane 1: hpema CLI (in-process orchestrator → talks to localhost:8001)
```

The CLI imports the orchestrator directly — no FastAPI server needed for the demo. The orchestrator's LLM client talks to `localhost:8001` (vLLM). That's the entire stack.

```
CLI (REPL)  →  Orchestrator (in-process)  →  vLLM (localhost:8001)
                    ↓
              Dafny verifier (subprocess)
                    ↓
              Audit logger (JSONL + SQLite)
```

The FastAPI backend stays for the API/dashboard use case but is **not needed** for the CLI demo.

---

## Step-by-Step Implementation Plan

### Phase 1: In-Process Runner -- COMPLETE

**Goal:** CLI runs the orchestrator directly, no FastAPI in between.

#### Step 1.1: `cli/runner.py` — in-process orchestrator driver

New file. Imports the orchestrator, wires dependencies, yields events to the display layer:

```python
async def run_pipeline(
    requirement: str,
    standard: str,
    language: str,
    max_iterations: int,
    display: DisplayManager,
) -> PipelineState:
    """Run the orchestrator in-process, updating the display in real-time."""
    orchestrator = PipelineOrchestrator(
        llm_client=LLMClient(ModelRegistry(config)),
        dafny_runner=DafnyRunner(),
        retriever=StandardsRetriever(),
        audit_logger=AuditLogger(),
    )

    request = PipelineRequest(
        requirement_text=requirement,
        safety_standard=standard,
        target_language=language,
        max_iterations=max_iterations,
    )

    async for event in orchestrator.run(request):
        display.handle_event(event)

    return orchestrator.last_state
```

#### Step 1.2: Update orchestrator to expose final state

The orchestrator currently only yields `StreamEvent`s. We need access to the final `PipelineState` after the generator exhausts. Add a `self.last_state` attribute that the runner can read after iteration.

#### Step 1.3: Verify it works

Quick test on the GPU node:
```python
# test_inprocess.py — run from tmux pane 1 while vLLM serves on pane 0
import asyncio
from cli.runner import run_pipeline
# ... should produce events and a final state
```

---

### Phase 2: Claude-Code Display Rewrite -- COMPLETE

**Goal:** Make the terminal output look like Claude Code's agent panels.

#### Step 2.1: Display manager with live spinners

**Rewrite `cli/display.py`** — replace static panels with animated components:

```
> Implement binary search with bounds checking

  Standard: DO_178C  |  Language: C

  ┌ Actor (Phi-4) ─────────────────────────────────
  │ ◐ Generating code + Dafny specification...
  └────────────────────────────────────────────────

     ↓  (after Actor completes, panel updates in-place)

  ┌ Actor (Phi-4) ──────────────── 3.2s ───────────
  │ ✓ Generated 24 lines of C + Dafny spec
  │
  │ int binary_search(int arr[], int n, int target) {
  │     /* REQ: bounds-checked binary search */
  │     if (n <= 0) return -1;
  │     int lo = 0, hi = n - 1;
  │     ...
  │ }
  └────────────────────────────────────────────────

  ┌ Checker (Phi-4) ───────────────────────────────
  │ ◐ Reviewing code...
  ├ Dafny Verifier ────────────────────────────────
  │ ◐ Verifying formal specification...
  └────────────────────────────────────────────────

     ↓  (after parallel completion)

  ┌ Checker ── FAIL ── 2.8s ───────────────────────
  │ ✗ 2 issues found
  │   [CRITICAL] Signed integer overflow possible in (lo + hi) / 2
  │   [MAJOR] Missing null check for arr parameter
  │
  │ Test cases generated: 3
  └────────────────────────────────────────────────

  ┌ Dafny ── VERIFIED ── 1.1s ─────────────────────
  │ ✓ All assertions hold
  │   requires arr.Length > 0
  │   ensures result >= -1 && result < arr.Length
  └────────────────────────────────────────────────

  ┌ Policy (Phi-4) ── DO_178C ─────────────────────
  │ ◐ Checking compliance...
  └────────────────────────────────────────────────

     ↓

  ┌ Policy ── NON-COMPLIANT ── 2.1s ───────────────
  │ ✗ Risk: HIGH
  │   [DO-178C-CR-005] Signed integer overflow is undefined behavior
  │   [DO-178C-CR-004] Array pointer not validated before use
  │
  │ Recommendations:
  │   • Use lo + (hi - lo) / 2 to prevent overflow
  │   • Add null check: if (arr == NULL) return -1
  └────────────────────────────────────────────────

  ── Iteration 1/2: FAIL ──────────────────────────
  Composing feedback → starting iteration 2...

     ↓  ... iteration 2 ...

  ╔══════════════════════════════════════════════════╗
  ║  Pipeline COMPLETE — 2 iterations, 14.3s        ║
  ║  Checker: PASS | Dafny: VERIFIED | Policy: COMPLIANT ║
  ║                                                  ║
  ║  Run ID: a1b2c3d4-...                           ║
  ║  Audit log: logs/audit/a1b2c3d4-....jsonl       ║
  ╚══════════════════════════════════════════════════╝
```

Key display components to implement:
- **`DisplayManager`** — orchestrates all display updates, tracks timing
- **`AgentSpinner`** — Rich `Status` with animated spinner while agent runs
- **`AgentPanel`** — completed panel with verdict, timing, content
- **`CodeBlock`** — syntax-highlighted source code (Rich `Syntax`)
- **`VerdictBanner`** — color-coded pass/fail for checker/dafny/policy
- **`IterationDivider`** — separator between iterations with pass/fail summary
- **`PipelineSummary`** — final results box

#### Step 2.2: Syntax-highlighted code output

When the Actor produces code, render with Rich `Syntax`:
```python
from rich.syntax import Syntax
syntax = Syntax(code_candidate.source_code, "c", theme="monokai", line_numbers=True)
console.print(syntax)
```

#### Step 2.3: Agent timing

Track wall-clock time per agent. Show in the panel header: `Actor (Phi-4) ── 3.2s`.

---

### Phase 3: Interactive REPL -- COMPLETE

**Goal:** User launches `hpema`, types requirements naturally, changes settings with slash commands.

#### Step 3.1: `cli/repl.py` — REPL loop

```
$ python -m cli.main

  ╭─────────────────────────────────────────────╮
  │  HPEMA v0.1.0                               │
  │  Hierarchical Policy-Enforced Multi-Agent    │
  │                                              │
  │  Model:    Phi-4-Reasoning-Plus (vLLM)      │
  │  Standard: DO_178C                           │
  │  Language: C                                 │
  │  Prover:   Dafny                             │
  │                                              │
  │  Type a requirement to begin, or /help       │
  ╰─────────────────────────────────────────────╯

hpema >
```

REPL commands:
- Free text → treated as a requirement, starts pipeline
- `/standard DO_178C` or `/standard MISRA_C` → change active standard
- `/language C` or `/language SPARK_Ada` → change target language
- `/iterations 3` → change max iterations
- `/audit` → show audit trail for last run
- `/last` → show full output of last run (code + proof + verdict)
- `/config` → show current configuration
- `/help` → show available commands
- `/quit` → exit

#### Step 3.2: Session state

The REPL holds session state:
```python
class Session:
    standard: str = "DO_178C"
    language: str = "C"
    max_iterations: int = 2
    last_run_id: UUID | None = None
    last_state: PipelineState | None = None
```

This lets `/audit` and `/last` reference the most recent run without re-specifying the run ID.

#### Step 3.3: Wire into `cli/main.py`

Update `main.py` so the default command (no subcommand) launches the REPL. Keep the existing `generate`, `audit`, `dashboard` commands for non-interactive use:

```
python -m cli.main                → REPL mode
python -m cli.main generate ...   → one-shot mode (existing)
python -m cli.main audit ...      → audit query (existing)
```

---

### Phase 4: Demo Polish -- IN PROGRESS

#### Step 4.1: Error resilience

- vLLM not running → `"Cannot connect to LLM at localhost:8001. Start vLLM first."`
- Dafny not installed → skip verification gracefully, show: `"Dafny not found — formal verification skipped. Install: dotnet tool install --global dafny"`
- LLM returns malformed JSON (shouldn't happen with constrained decoding, but just in case) → show partial results with a warning

#### Step 4.2: Startup banner with config detection

On startup, the REPL pings the vLLM endpoint to confirm it's live and shows the actual model name:

```
  Model: Phi-4-Reasoning-Plus (vLLM @ localhost:8001) ✓ connected
```

or:

```
  Model: Phi-4-Reasoning-Plus (vLLM @ localhost:8001) ✗ not reachable
```

#### Step 4.3: Canned demo script

Create `demo.sh` for rehearsed demos — runs scenarios non-interactively:
```bash
#!/bin/bash
echo "=== HPEMA Demo ==="
echo ""
echo "Scenario 1: Altitude Hold Controller (DO-178C)"
echo "───────────────────────────────────────────────"
HPEMA_CONFIG=hpema_config.arc.yaml python -m cli.main generate \
    --requirement "Implement altitude hold controller that clamps vertical speed between -1000 and 1000 ft/min" \
    --standard DO_178C --language C

read -p "Press Enter for Scenario 2..."

echo ""
echo "Scenario 2: Binary Search (MISRA C)"
echo "────────────────────────────────────"
HPEMA_CONFIG=hpema_config.arc.yaml python -m cli.main generate \
    --requirement "Implement binary search with bounds checking for sorted integer array" \
    --standard MISRA_C --language C
```

---

## File Changes Summary

| File | Action | Description |
|---|---|---|
| `cli/runner.py` | CREATE | In-process orchestrator driver, no FastAPI needed |
| `cli/repl.py` | CREATE | Interactive REPL with slash commands and session state |
| `cli/display.py` | REWRITE | Claude-Code-style panels, spinners, syntax highlighting, timing |
| `cli/main.py` | UPDATE | Default command launches REPL; keep existing subcommands |
| `cli/config.py` | UPDATE | Add REPL session defaults |
| `backend/services/orchestrator.py` | UPDATE | Expose `last_state` after generator exhausts |
| `demo.sh` | CREATE | Canned demo scenarios for rehearsal |

No new config files needed — `hpema_config.arc.yaml` already points at `localhost:8001`.

## Implementation Order

1. **Phase 1** — `cli/runner.py` + orchestrator `last_state` — get pipeline running in-process
2. **Phase 2** — `cli/display.py` rewrite — make it look like Claude Code
3. **Phase 3** — `cli/repl.py` + update `cli/main.py` — interactive mode
4. **Phase 4** — polish (error handling, startup banner, demo script)

Phase 1 is the critical path. Phase 2 is the visual payoff. Phases 3-4 are presentation polish.

---

## HPC Node Setup (Recap)

```bash
# Reserve node
salloc --partition=a100_normal_q --gres=gpu:1 --time=2:00:00

# tmux pane 0: vLLM
python -m vllm.entrypoints.openai.api_server \
    --model microsoft/Phi-4-Reasoning-Plus \
    --port 8001 \
    --max-model-len 8192 \
    --structured-outputs-config.backend outlines \
    --dtype auto --trust-remote-code

# tmux pane 1: CLI (after vLLM is ready)
export HPEMA_CONFIG=hpema_config.arc.yaml
export HPEMA_API_KEY=unused
python -m cli.main
```

---

## Demo Day Flow

1. SSH into ARC, show the GPU node reservation (Boeing: "this runs on your hardware")
2. Show `hpema_config.arc.yaml`: "Swap this file to plug in your models"
3. Launch `hpema` REPL
4. Type: "Implement altitude hold controller that clamps vertical speed..."
5. Watch agents work in real-time — spinners, code panels, verdicts
6. Point at Dafny panel: "This code is mathematically proven correct for ALL inputs"
7. Point at Policy panel: "These are real DO-178C clause citations"
8. Type `/standard MISRA_C`, re-run → different violations, different citations
9. Type `/audit` → traceability matrix
10. Mentor takeaway: "Three guarantees. Plug-and-play. Runs on your infra."
