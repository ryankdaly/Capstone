#!/usr/bin/env python3
"""Solo condition runner.

One-shot LLM call with a clean minimal prompt. No pipeline, no verification,
no feedback loop. Saves raw output to data/harness/outputs/.

Staged execution (models with enabled=False are skipped in --all runs unless
--include-staged is passed):
  Stage 1: OSS NVIDIA NIM  (mistral-large, qwen3-235b, gemma-3-27b)
  Stage 2: DeepSeek        (deepseek-v4-pro, deepseek-v4-flash)
  Stage 4: gpt-5.4-mini    (--include-staged to unlock)

Usage:
    python scripts/harness/run_solo.py --model mistral-large --all --reps 5
    python scripts/harness/run_solo.py --model deepseek-v4-pro --prompt-id is_zero --rep 1
    python scripts/harness/run_solo.py --model gpt-5.4-mini --all --include-staged
    python scripts/harness/run_solo.py --model mistral-large --all --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from openai import AsyncOpenAI

from scripts.harness.models import MODELS, ModelSpec, get_model

PROMPTS_FILE = PROJECT_ROOT / "data" / "harness" / "prompts" / "prompts.yaml"
OUTPUTS_DIR = PROJECT_ROOT / "data" / "harness" / "outputs"

SOLO_SYSTEM_PROMPT = (
    "You are a Python software engineer writing safety-critical avionics code. "
    "Write safe, correct, well-tested Python code that complies with {standard}. "
    "Return ONLY the Python implementation — no markdown fences, no explanation, "
    "no preamble. Just the Python source."
)


def load_prompts() -> list[dict]:
    return yaml.safe_load(PROMPTS_FILE.read_text())


def output_path(model_id: str, prompt_id: str, rep: int) -> Path:
    p = OUTPUTS_DIR / model_id / "solo"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{prompt_id}_rep{rep:02d}.json"


def extract_code(raw: str) -> str:
    """Strip markdown code fences if the model added them anyway."""
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", raw, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return raw.strip()


def _build_create_kwargs(spec: ModelSpec) -> dict:
    """Build chat.completions.create() kwargs from ModelSpec fields.

    extra_params go as top-level keyword args (e.g. reasoning_effort="high").
    extra_body_override is merged into the extra_body dict.
    Temperature is omitted for reasoning models that don't support it
    (DeepSeek-v4-pro with thinking enabled requires temp=1 or omit).
    """
    kwargs: dict = {}

    # DeepSeek-v4-pro with thinking requires temperature=1 or omitted.
    # For non-reasoning models, pass the configured temperature.
    if spec.enable_thinking and spec.extra_params.get("reasoning_effort"):
        kwargs["temperature"] = 1  # required by DeepSeek reasoning mode
    else:
        kwargs["temperature"] = spec.temperature

    # Top-level extra params (e.g. reasoning_effort="high")
    kwargs.update(spec.extra_params)

    # extra_body payload
    if spec.extra_body_override:
        kwargs["extra_body"] = spec.extra_body_override

    return kwargs


async def run_one(
    spec: ModelSpec,
    prompt: dict,
    rep: int,
    *,
    dry_run: bool = False,
) -> dict:
    out_path = output_path(spec.id, prompt["id"], rep)
    legacy_path = out_path.parent / f"{spec.id}_solo_{prompt['id']}_rep{rep:02d}.json"
    if not out_path.exists() and legacy_path.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_path.rename(out_path)
        print(f"  [migrated] {legacy_path.name} → {out_path.relative_to(OUTPUTS_DIR)}")
    if out_path.exists():
        print(f"  [skip] {out_path.name} already exists")
        return json.loads(out_path.read_text())

    api_key = os.environ.get(spec.api_key_env, "")
    if not api_key and not dry_run:
        raise RuntimeError(
            f"API key env var {spec.api_key_env!r} is not set for model {spec.id!r}"
        )

    system = SOLO_SYSTEM_PROMPT.format(standard=prompt.get("standard", "DO-178C"))
    user = prompt["requirement"].strip()

    if dry_run:
        kwargs = _build_create_kwargs(spec)
        print(
            f"  [dry-run] model={spec.model_id} endpoint={spec.endpoint} "
            f"prompt={prompt['id']} rep={rep} extra_kwargs={kwargs}"
        )
        return {}

    client = AsyncOpenAI(
        base_url=spec.endpoint,
        api_key=api_key or "unused",
        timeout=360.0,
    )

    create_kwargs = _build_create_kwargs(spec)

    _RETRY_DELAYS = [5, 10, 30]  # then repeats 30s indefinitely

    start = datetime.now(timezone.utc)
    attempt = 0
    while True:
        try:
            response = await client.chat.completions.create(
                model=spec.model_id,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **create_kwargs,
            )
            break
        except Exception as exc:
            delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
            attempt += 1
            print(f"  [retry #{attempt} in {delay}s] {type(exc).__name__}: {exc}")
            await asyncio.sleep(delay)
    end = datetime.now(timezone.utc)

    raw = response.choices[0].message.content or ""
    code = extract_code(raw)
    loc = len([l for l in code.splitlines() if l.strip() and not l.strip().startswith("#")])

    # Token usage from API response
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = getattr(details, "cached_tokens", 0) or 0
    total_tokens = input_tokens + output_tokens
    estimated_cost = spec.compute_cost(input_tokens, output_tokens, cached_tokens)

    record = {
        "prompt_id": prompt["id"],
        "difficulty": prompt["level"],
        "model": spec.id,
        "condition": "solo",
        "repetition": rep,
        "source_code": code,
        "lines_of_code": loc,
        "thinking_enabled": spec.enable_thinking,
        # Token / cost tracking (business metrics)
        "actor_input_tokens": input_tokens,
        "actor_output_tokens": output_tokens,
        "actor_cached_input_tokens": cached_tokens,
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "total_cached_input_tokens": cached_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(estimated_cost, 6),
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
        # Business metrics (populated by aggregate.py after pairing with HPEMA)
        "css_per_1k_tokens": None,
        "iterations_used": 1,
        "elapsed_seconds": (end - start).total_seconds(),
        "generated_at": start.isoformat(),
    }

    out_path.write_text(json.dumps(record, indent=2))
    print(f"  [ok] {out_path.name} ({loc} LOC, {(end-start).total_seconds():.1f}s)")
    return record


async def run_all(
    spec: ModelSpec,
    prompts: list[dict],
    reps: int,
    *,
    dry_run: bool = False,
    prompt_filter: str | None = None,
) -> None:
    for prompt in prompts:
        if prompt_filter and prompt["id"] != prompt_filter:
            continue
        for rep in range(1, reps + 1):
            print(f"[solo] {spec.id} / {prompt['id']} rep={rep}")
            await run_one(spec, prompt, rep, dry_run=dry_run)
            await asyncio.sleep(0.5)


def _slice_prompts(prompts: list[dict], *, from_id: str | None, upto_id: str | None) -> list[dict]:
    ids = [p["id"] for p in prompts]
    start = ids.index(from_id) if from_id else 0
    end = ids.index(upto_id) + 1 if upto_id else len(ids)
    if from_id and from_id not in ids:
        print(f"ERROR: --from {from_id!r} not found. Valid: {ids}")
        sys.exit(1)
    if upto_id and upto_id not in ids:
        print(f"ERROR: --upto {upto_id!r} not found. Valid: {ids}")
        sys.exit(1)
    return prompts[start:end]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run solo (no-pipeline) eval condition")
    parser.add_argument("--model", required=True, choices=list(MODELS), help="Model to run")
    parser.add_argument("--prompt-id", help="Run a single prompt by ID")
    parser.add_argument("--all", action="store_true", help="Run all 20 prompts")
    parser.add_argument("--from", dest="from_id", metavar="PROMPT_ID",
                        help="Start from this prompt (inclusive, YAML order) — use with --all")
    parser.add_argument("--upto", dest="upto_id", metavar="PROMPT_ID",
                        help="Stop after this prompt (inclusive, YAML order) — use with --all")
    parser.add_argument("--reps", type=int, default=5, help="Repetitions per prompt (default 5)")
    parser.add_argument("--rep", type=int, default=1, help="Single rep number (with --prompt-id)")
    parser.add_argument("--dry-run", action="store_true", help="Print calls without hitting API")
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
