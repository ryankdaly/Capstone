# Directory Structure Guide
---

## Recommended Structure We Follow

```
HPEMA-Aerospace/
│
├── README.md                      # Project overview & HPEMA architecture
├── PROJECT_ONBOARDING.md          # VT ARC specific setup (module load, conda, etc.)
│
├── data/
│   ├── raw/                       # Unprocessed datasets (e.g., flight logs, design docs)
│   ├── kg/                        # Boeing Knowledge Graph & 2,700 Design Practices
│   │   ├── ontology.json
│   │   └── practices_indexed.db   # Vector store or SQLite for RAG
│   │
│   └── training_ready/            # PEFT/LoRA fine-tuning datasets
│       ├── policy_alignment/      # Data for aligning Mistral-Small to AS9100
│       └── actor_specialization/  # Data for Qwen3 Python/Java compliance
│
├── src/                           # All Python and Formal Logic source code
│   ├── agents/                    # Multi-Agent implementation
│   │   ├── actor.py               # Qwen3-based Coder agent
│   │   ├── checker.py             # Phi-4-based Verifier agent
│   │   └── senior_policy.py       # Mistral-based Auditor agent
│   │
│   ├── formal_methods/            # Verification logic
│   │   ├── dafny/                 # .dfy files & templates
│   │   └── lean4/                 # .lean files for algorithmic proofs
│   │
│   ├── orchestration/             # Framework logic (HPEMA Triad)
│   │   └── system_orchestrator.py
│   │
│   └── evaluation/
│       ├── safety_benchmarks.py   # MISRA/DO-178C compliance checks
│       └── langsmith_eval.py      # Observability integration
│
├── slurm/                         # VT ARC HPC Batch Scripts
│   ├── tinkercliffs/              # CPU-heavy jobs (Dafny verification)
│   ├── infer/                     # GPU-heavy jobs (A100/H100 for LLM inference)
│   │   ├── run_agent_pipeline.sbatch
│   │   └── train_lora_peft.sbatch
│   └── setup_env.sh               # script to automate 'module load' & conda activate
│
├── outputs/                       # Model artifacts
│   ├── lora_adapters/             # Fine-tuned weights (Policy/Actor)
│   └── logs/                      # Agent communication traces (JSONL)
│
├── results/                       # High-level metrics
│   ├── verification_reports/      # PDF/MD summaries of Dafny successes
│   └── compliance_audit.csv       # Senior Policy Agent pass/fail logs
│
├── logs/                          # SLURM output/error files
│   ├── agents/                    # Agent execution logs
│   └── training/                  # DeepSpeed/LoRA training logs
│
└── tars/                          # Archives for transfer to/from /scratch
    └── datasets.tar.gz

```

---

## VT ARC Naming Conventions

### Output Directories

Every experiment or agent run gets a unique ID to prevent overwriting on shared filesystems:
`outputs/run{ID}_{AgentType}_{ModelName}/`

Example:

* `outputs/run01_PolicyAgent_Mistral24B/`

### SLURM Logs

On ARC, always use `%j` (Job ID) and `%x` (Job Name) to differentiate logs:

```bash
#SBATCH --job-name=HPEMA_Actor_Train
#SBATCH --output=logs/training/%x_%j.out
#SBATCH --error=logs/training/%x_%j.err

```

---

## Rules for VT ARC Efficiency

### 1. The /scratch Rule

**Never run training directly from `/home`.**

1. Move your data `tar` to `/scratch/user/$PID`.
2. Untar, run your SLURM job.
3. Copy the resulting `lora_adapters` back to `/work` or `/home` before the scratch purge.

### 2. Formal Verification Isolation

Dafny and Lean 4 verification are CPU-intensive. Run these on **TinkerCliffs** (CPU nodes) to save your **Infer** (GPU) allocation for the LLMs.

### 3. Agent Traceability

Because the HPEMA framework is non-deterministic, every agentic loop MUST save its interaction history:

* **Input:** `logs/agents/run{ID}_trace.jsonl`
* **Content:** `{ "agent": "Actor", "output": "code...", "thought": "...", "checker_feedback": "..." }`

### 4. Knowledge Graph Versioning

The Boeing Design Practices (KG) are the "Policy." Any update to the Knowledge Graph requires a new version tag in `data/kg/` to ensure your Senior Policy Agent isn't auditing against outdated rules.

---

## Migration Checklist (For Team)

1. **Clone to ARC:** `git clone ...` into your `/home` or `/work` directory.
2. **Setup Symlinks:**
```bash
# Link scratch for heavy outputs
ln -s /scratch/your_vt_id/hpema_outputs ./outputs_scratch

```


3. **Path Updates:** Ensure `src/orchestration/config.py` uses relative paths or environment variables (e.g., `$DATA_DIR`) so scripts work for all teammates.
