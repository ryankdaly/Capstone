#!/usr/bin/env python3
"""HPEMA condition runner.

Invokes the full HPEMA pipeline (stage: policy, max_iterations: 3).
Extracts state.final_code from the best-scored iteration.
Saves output to data/harness/outputs/.

Staged execution mirrors run_solo.py:
  Stage 1: OSS NVIDIA NIM  (mistral-large, qwen3-235b, gemma-3-27b)
  Stage 2: DeepSeek        (deepseek-v4-pro, deepseek-v4-flash)
  Stage 4: gpt-5.4-mini    (--include-staged to unlock)

Usage:
    python scripts/harness/run_hpema.py --model mistral-large --all --reps 5
    python scripts/harness/run_hpema.py --model deepseek-v4-pro --prompt-id is_zero --rep 1
    python scripts/harness/run_hpema.py --model gpt-5.4-mini --all --include-staged
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from scripts.harness.models import MODELS, ModelSpec, get_model

PROMPTS_FILE = PROJECT_ROOT / "data" / "harness" / "prompts" / "prompts.yaml"
OUTPUTS_DIR = PROJECT_ROOT / "data" / "harness" / "outputs"
LOCAL_CONFIG = PROJECT_ROOT / "hpema_config.local.yaml"

# Point HPEMA_CONFIG to the project-local config before any backend module is
# imported. The backend's `settings` singleton is created at import time via
# `settings = load_config()`, so the env var must be set first. All backend
# imports in this file are inside functions, so this runs in time.
if LOCAL_CONFIG.exists() and not os.environ.get("HPEMA_CONFIG"):
    os.environ["HPEMA_CONFIG"] = str(LOCAL_CONFIG)


def load_prompts() -> list[dict]:
    return yaml.safe_load(PROMPTS_FILE.read_text())


def output_path(model_id: str, prompt_id: str, rep: int) -> Path:
    p = OUTPUTS_DIR / model_id / "hpema"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{prompt_id}_rep{rep:02d}.json"


def _build_config(spec: ModelSpec):
    """Build a HpemaConfig with the actor model overridden to `spec`.

    Passes enable_thinking and extra_body through to the actor config so
    the LLMClient activates the correct family profile kwargs.
    For DeepSeek-v4-pro with thinking, the profile handles reasoning_effort
    and extra_body.thinking automatically when enable_thinking=True.
    """
    from backend.config import (
        HpemaConfig,
        ModelEndpointConfig,
        ModelsConfig,
        PipelineConfig,
        load_config,
    )
    base = load_config(LOCAL_CONFIG if LOCAL_CONFIG.exists() else None)
    actor_cfg = ModelEndpointConfig(
        endpoint=spec.endpoint,
        model=spec.model_id,
        api_key_env=spec.api_key_env,
        family=spec.family,
        enable_thinking=spec.enable_thinking,
        # extra_body_override feeds into profile's thinking_on_extra_body merge
        # For non-profile overrides pass through extra_body field directly
        extra_body=spec.extra_body_override,
    )
    return HpemaConfig(
        models=ModelsConfig(
            actor=actor_cfg,
            checker=base.models.checker,
            policy=base.models.policy,
            dafny_architect=base.models.dafny_architect,
        ),
        policies=base.policies,
        verification=base.verification,
        pipeline=PipelineConfig(
            max_iterations=3,
            stage="policy",
            require_human_approval=False,
            audit_log_dir=base.pipeline.audit_log_dir,
        ),
    )


async def run_one(
    spec: ModelSpec,
    prompt: dict,
    rep: int,
    *,
    dry_run: bool = False,
) -> dict:
    out_path = output_path(spec.id, prompt["id"], rep)
    # Legacy name inside new folder: outputs/{model}/hpema/{model}_hpema_{prompt_id}_rep{n}.json
    legacy_path = out_path.parent / f"{spec.id}_hpema_{prompt['id']}_rep{rep:02d}.json"
    if not out_path.exists() and legacy_path.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_path.rename(out_path)
        print(f"  [migrated] {legacy_path.name} → {out_path.relative_to(OUTPUTS_DIR)}")
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        code = existing.get("source_code", "").strip()
        if code:
            try:
                compile(code, "<string>", "exec")
                print(f"  [skip] {out_path.name} (status={existing.get('pipeline_status')!r})")
                return existing
            except SyntaxError:
                print(f"  [retry] {out_path.name} — source code is syntactically incomplete")
        else:
            print(f"  [retry] {out_path.name} — no source code generated")

    api_key = os.environ.get(spec.api_key_env, "")
    if not api_key and not dry_run:
        raise RuntimeError(
            f"API key env var {spec.api_key_env!r} is not set for model {spec.id!r}"
        )

    if dry_run:
        print(f"  [dry-run] would run HPEMA {spec.model_id} for {prompt['id']} rep={rep}")
        return {}

    from backend.config import HpemaConfig
    from backend.services.llm.model_registry import ModelRegistry
    from backend.services.orchestrator import PipelineOrchestrator
    from backend.api.schemas.pipeline import PipelineRequest, PipelineStage, PipelineStatus
    from backend.api.schemas.generation import SafetyStandard, TargetLanguage
    from scripts.harness.tracked_client import TrackedLLMClient

    cfg = _build_config(spec)
    registry = ModelRegistry(config=cfg)
    llm = TrackedLLMClient(registry=registry)   # token-counting wrapper
    orchestrator = PipelineOrchestrator(llm_client=llm)

    standard_map = {
        "DO_178C": SafetyStandard.DO_178C,
    }
    standard = standard_map.get(prompt.get("standard", "DO_178C"), SafetyStandard.DO_178C)

    request = PipelineRequest(
        requirement_text=prompt["requirement"].strip(),
        safety_standard=standard,
        target_language=TargetLanguage.PYTHON,
        max_iterations=3,
        stage=PipelineStage.POLICY,
        run_tests=True,
    )

    start = datetime.now(timezone.utc)

    # Consume the async generator to drive the pipeline to completion
    async for _event in orchestrator.run(request):
        pass  # events go to logs; we read final state below

    end = datetime.now(timezone.utc)
    state = orchestrator.last_state

    if state is None:
        raise RuntimeError("Orchestrator returned no state — pipeline did not run")

    code = state.final_code or ""
    proof = state.final_proof or ""
    loc = len([l for l in code.splitlines() if l.strip() and not l.strip().startswith("#")])
    iterations_used = state.current_iteration

    # Token usage from TrackedLLMClient
    usage_summary = llm.get_usage_summary()
    actor_tok = llm.actor_tokens()
    actor_in = actor_tok["input"]
    actor_out = actor_tok["output"]
    actor_cached = actor_tok["cached"]
    total_in = usage_summary["total_input_tokens"]
    total_out = usage_summary["total_output_tokens"]
    total_cached = usage_summary["total_cached_input_tokens"]
    total_tokens = total_in + total_out

    # Cost estimate: actor tokens billed at actor model rate;
    # non-actor pipeline tokens (checker/policy/dafny_arch) at actor rate too
    # since they often use the same or cheaper model — conservative upper bound.
    estimated_actor_cost = spec.compute_cost(actor_in, actor_out, actor_cached)
    estimated_pipeline_cost = spec.compute_cost(total_in, total_out, total_cached)

    record = {
        "prompt_id": prompt["id"],
        "difficulty": prompt["level"],
        "model": spec.id,
        "condition": "hpema",
        "repetition": rep,
        "source_code": code,
        "dafny_spec": proof,
        "lines_of_code": loc,
        "iterations_used": iterations_used,
        "pipeline_status": state.status.value,
        "thinking_enabled": spec.enable_thinking,
        # Token / cost tracking (business metrics)
        "actor_input_tokens": actor_in,
        "actor_output_tokens": actor_out,
        "actor_cached_input_tokens": actor_cached,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_cached_input_tokens": total_cached,
        "total_tokens": total_tokens,
        "token_usage_by_role": usage_summary["by_role"],
        "estimated_actor_cost_usd": round(estimated_actor_cost, 6),
        "estimated_cost_usd": round(estimated_pipeline_cost, 6),
        # Quality metrics (populated by evaluate.py)
        "bandit_high": None,
        "bandit_medium": None,
        "bandit_low": None,
        "vulnerability_density": None,
        "VS": None,
        "policy_violations": None,
        "PA": None,
        "edge_tests_passed": None,
        "edge_tests_total": None,
        "EC": None,
        "dafny_verified": None,
        "FV": None,
        "CSS": None,
        # Business metrics (populated by aggregate.py after pairing with solo)
        "css_per_1k_tokens": None,
        "elapsed_seconds": (end - start).total_seconds(),
        "generated_at": start.isoformat(),
    }

    out_path.write_text(json.dumps(record, indent=2))
    print(f"  [ok] {out_path.name} (status={state.status.value}, iters={iterations_used})")
    return record


async def run_all(
    spec: ModelSpec,
    prompts: list[dict],
    reps: int,
    *,
    dry_run: bool = False,
    prompt_filter: str | None = None,
) -> None:
    wall_start = datetime.now(timezone.utc)
    run_times: list[tuple[str, int, float]] = []  # (prompt_id, rep, elapsed_s)
    errors: list[tuple[str, int, str]] = []

    for prompt in prompts:
        if prompt_filter and prompt["id"] != prompt_filter:
            continue
        for rep in range(1, reps + 1):
            print(f"[hpema] {spec.id} / {prompt['id']} rep={rep}")
            t0 = datetime.now(timezone.utc)
            try:
                await run_one(spec, prompt, rep, dry_run=dry_run)
            except Exception as e:
                print(f"  [error] {e}")
                errors.append((prompt["id"], rep, str(e)))
            elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
            run_times.append((prompt["id"], rep, elapsed))
            # Brief delay between pipeline runs
            await asyncio.sleep(1.0)

    # ── Summary ──────────────────────────────────────────────────────────────
    total_wall = (datetime.now(timezone.utc) - wall_start).total_seconds()
    n = len(run_times)
    if n == 0:
        return

    times = [t for _, _, t in run_times]
    avg_t = sum(times) / n
    min_t = min(times)
    max_t = max(times)
    slowest = max(run_times, key=lambda x: x[2])

    def _hms(s: float) -> str:
        h, rem = divmod(int(s), 3600)
        m, sec = divmod(rem, 60)
        return f"{h}h {m}m {sec}s" if h else f"{m}m {sec}s"

    print("\n" + "=" * 60)
    print(f"HPEMA run complete — {spec.id}")
    print(f"  Total wall time : {_hms(total_wall)}")
    print(f"  Runs attempted  : {n}  (errors: {len(errors)})")
    print(f"  Per-run elapsed : avg {avg_t:.1f}s  min {min_t:.1f}s  max {max_t:.1f}s")
    print(f"  Slowest         : {slowest[0]} rep={slowest[1]} ({slowest[2]:.1f}s)")
    if errors:
        print(f"  Errors:")
        for pid, r, msg in errors:
            print(f"    {pid} rep={r}: {msg[:80]}")
    print("=" * 60)


def _slice_prompts(prompts: list[dict], *, from_id: str | None, upto_id: str | None) -> list[dict]:
    ids = [p["id"] for p in prompts]
    if from_id and from_id not in ids:
        print(f"ERROR: --from {from_id!r} not found. Valid: {ids}")
        sys.exit(1)
    if upto_id and upto_id not in ids:
        print(f"ERROR: --upto {upto_id!r} not found. Valid: {ids}")
        sys.exit(1)
    start = ids.index(from_id) if from_id else 0
    end = ids.index(upto_id) + 1 if upto_id else len(ids)
    return prompts[start:end]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HPEMA pipeline eval condition")
    parser.add_argument("--model", required=True, choices=list(MODELS), help="Actor model")
    parser.add_argument("--prompt-id", help="Run a single prompt by ID")
    parser.add_argument("--all", action="store_true", help="Run all 20 prompts")
    parser.add_argument("--from", dest="from_id", metavar="PROMPT_ID",
                        help="Start from this prompt (inclusive, YAML order) — use with --all")
    parser.add_argument("--upto", dest="upto_id", metavar="PROMPT_ID",
                        help="Stop after this prompt (inclusive, YAML order) — use with --all")
    parser.add_argument("--reps", type=int, default=5, help="Repetitions (default 5)")
    parser.add_argument("--rep", type=int, default=1, help="Single rep number (with --prompt-id)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would run without calling API")
    parser.add_argument(
        "--include-staged", action="store_true",
        help="Allow running staged (enabled=False) models — e.g. gpt-5.4-mini",
    )
    args = parser.parse_args()

    if not args.prompt_id and not args.all:
        parser.error("Pass --prompt-id <id> or --all")

    spec = get_model(args.model)

    if not spec.enabled and not args.include_staged:
        print(
            f"ERROR: {spec.id!r} is staged (enabled=False).\n"
            f"  Reason: {spec.stage_note}\n"
            f"  Pass --include-staged to run anyway."
        )
        sys.exit(1)

    prompts = load_prompts()

    if args.prompt_id:
        matched = [p for p in prompts if p["id"] == args.prompt_id]
        if not matched:
            print(f"ERROR: prompt_id {args.prompt_id!r} not found.")
            sys.exit(1)
        asyncio.run(run_one(spec, matched[0], args.rep, dry_run=args.dry_run))
    else:
        subset = _slice_prompts(prompts, from_id=args.from_id, upto_id=args.upto_id)
        asyncio.run(run_all(spec, subset, args.reps, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
