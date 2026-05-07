#!/usr/bin/env python3
"""Evaluate artifact files: run Bandit, pytest edge suite, Dafny verify, Policy agent.

Reads output JSON files (from run_solo.py / run_hpema.py), runs all evaluators,
and writes VS, PA, EC, FV back into each JSON file in-place.

Usage:
    # Evaluate a single artifact:
    python scripts/harness/evaluate.py data/harness/outputs/gpt-5.4-mini_solo_is_zero_rep01.json

    # Evaluate all unevaluated artifacts:
    python scripts/harness/evaluate.py --all

    # Re-evaluate even if already evaluated:
    python scripts/harness/evaluate.py --all --force
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUTS_DIR = PROJECT_ROOT / "data" / "harness" / "outputs"
EDGE_TESTS_DIR = PROJECT_ROOT / "data" / "harness" / "edge_tests"
PROMPTS_FILE = PROJECT_ROOT / "data" / "harness" / "prompts" / "prompts.yaml"


def _prompt_order() -> list[str]:
    """Return prompt IDs in YAML definition order."""
    import yaml
    prompts = yaml.safe_load(PROMPTS_FILE.read_text())
    return [p["id"] for p in prompts]


def _filter_upto(paths: list[Path], upto: str) -> list[Path]:
    """Keep only artifacts whose prompt_id appears at or before `upto` in prompts.yaml."""
    order = _prompt_order()
    if upto not in order:
        raise SystemExit(f"ERROR: prompt_id {upto!r} not found in prompts.yaml. "
                         f"Valid IDs: {order}")
    allowed = set(order[: order.index(upto) + 1])
    kept = []
    for p in paths:
        try:
            pid = json.loads(p.read_text()).get("prompt_id", "")
        except Exception:
            pid = ""
        if pid in allowed:
            kept.append(p)
    return kept

_LOCAL_CONFIG = PROJECT_ROOT / "hpema_config.local.yaml"
if _LOCAL_CONFIG.exists() and not os.environ.get("HPEMA_CONFIG"):
    os.environ["HPEMA_CONFIG"] = str(_LOCAL_CONFIG)

# λ parameter for VS sigmoid: 1 weighted finding per 100 LOC → VS = 0.50
LAMBDA_VS = 100

# DO-178C rule count — measured from data/chromadb on 2026-04-27.
# Re-run: python scripts/harness/count_rules.py if standards corpus changes.
R_RULES = 61


# ---------------------------------------------------------------------------
# Bandit (VS)
# ---------------------------------------------------------------------------

def _bandit_bin() -> str:
    """Prefer venv's bandit over system PATH."""
    import shutil
    venv_bandit = Path(sys.executable).parent / "bandit"
    if venv_bandit.exists():
        return str(venv_bandit)
    found = shutil.which("bandit")
    if found:
        return found
    raise FileNotFoundError("bandit not found — run: uv add bandit")


def run_bandit(source_path: Path) -> dict:
    """Run Bandit static analysis. Returns raw counts by severity."""
    try:
        result = subprocess.run(
            [_bandit_bin(), "-r", "-f", "json", "-q", str(source_path)],
            capture_output=True, text=True, timeout=60,
        )
        data = json.loads(result.stdout or "{}")
        metrics = data.get("metrics", {}).get("_totals", {})
        return {
            "high": int(metrics.get("SEVERITY.HIGH", 0)),
            "medium": int(metrics.get("SEVERITY.MEDIUM", 0)),
            "low": int(metrics.get("SEVERITY.LOW", 0)),
        }
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  [bandit error] {e}")
        return {"high": 0, "medium": 0, "low": 0}


def compute_vs(high: int, medium: int, low: int, loc: int) -> float:
    loc = max(loc, 1)
    vd = (3 * high + 1 * medium + 0.2 * low) / loc
    return 1.0 / (1.0 + LAMBDA_VS * vd)


# ---------------------------------------------------------------------------
# pytest edge suite (EC)
# ---------------------------------------------------------------------------

def run_edge_tests(prompt_id: str, source_path: Path) -> tuple[int, int]:
    """Run the edge test suite for `prompt_id`. Returns (passed, total)."""
    test_file = EDGE_TESTS_DIR / f"test_{prompt_id}.py"
    if not test_file.exists():
        print(f"  [edge tests] no test file for {prompt_id!r} — skipping")
        return 0, 0

    env = os.environ.copy()
    env["ARTIFACT_PATH"] = str(source_path)

    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            str(test_file),
            "-v", "--tb=no",
            "--no-header",
            f"--rootdir={EDGE_TESTS_DIR}",
        ],
        capture_output=True, text=True, timeout=120,
        env=env,
        cwd=str(EDGE_TESTS_DIR),
    )

    # With -v (no -q), each test line ends with " PASSED", " FAILED", or " ERROR"
    passed = result.stdout.count(" PASSED")
    failed = result.stdout.count(" FAILED")
    error = result.stdout.count(" ERROR")
    total = passed + failed + error

    # Fallback: parse summary line "N passed, M failed in Xs" in case format differs
    if total == 0:
        import re
        m = re.search(r"(\d+) passed", result.stdout)
        f = re.search(r"(\d+) failed", result.stdout)
        e = re.search(r"(\d+) error", result.stdout)
        passed = int(m.group(1)) if m else 0
        failed = int(f.group(1)) if f else 0
        error = int(e.group(1)) if e else 0
        total = passed + failed + error

    return passed, total


# ---------------------------------------------------------------------------
# Policy agent (PA)
# ---------------------------------------------------------------------------

async def run_policy(source_code: str, standard: str) -> list[dict]:
    """Run HPEMA Policy agent post-hoc and return list of violations."""
    try:
        from backend.services.agents.policy import PolicyAgent
        from backend.services.llm.client import LLMClient
        from backend.services.rag.retriever import StandardsRetriever
        from backend.config import resolve_data_path, settings

        llm = LLMClient()
        retriever = StandardsRetriever(
            persist_dir=str(resolve_data_path(settings.policies.chromadb_dir, "chromadb")),
            auto_ingest_path=str(resolve_data_path(settings.policies.standards_dir, "standards")),
        )
        agent = PolicyAgent(llm)

        # Build a minimal code candidate for the policy agent
        from backend.api.schemas.agents import CodeCandidate
        from backend.api.schemas.generation import SafetyStandard, TargetLanguage
        from uuid import uuid4

        code_candidate = CodeCandidate(
            source_code=source_code,
            language=TargetLanguage.PYTHON,
        )

        standard_map = {
            "DO_178C": SafetyStandard.DO_178C,
        }
        std = standard_map.get(standard, SafetyStandard.DO_178C)

        verdict = await agent.run(
            run_id=uuid4(),
            source_code=source_code,
            safety_standard=std,
            retriever=retriever,
        )

        if verdict is None:
            return []

        violations = []
        for v in (verdict.violations or []):
            violations.append({
                "rule_id": getattr(v, "rule_id", "unknown"),
                "severity": getattr(v, "severity", "MINOR"),
                "description": getattr(v, "description", ""),
            })
        return violations

    except Exception as e:
        print(f"  [policy error] {e}")
        return []


def compute_pa(violations: list[dict], r_rules: int = R_RULES) -> float:
    weight_map = {"CRITICAL": 2.0, "MAJOR": 1.0, "MINOR": 0.5, "INFO": 0.1}
    total_weight = sum(weight_map.get(v.get("severity", "MINOR"), 0.5) for v in violations)
    return max(0.0, 1.0 - total_weight / (2.0 * max(r_rules, 1)))


# ---------------------------------------------------------------------------
# Dafny verification (FV)
# ---------------------------------------------------------------------------

def _load_requirement(prompt_id: str) -> str:
    """Load requirement text for a prompt_id from prompts.yaml."""
    import yaml
    prompts = yaml.safe_load(PROMPTS_FILE.read_text())
    for p in prompts:
        if p["id"] == prompt_id:
            return p["requirement"].strip()
    return f"Implement the function described by prompt id: {prompt_id}"


async def run_dafny(source_code: str, prompt_id: str) -> bool:
    """Generate a Dafny spec from source_code and verify it. Returns True if verified."""
    try:
        from backend.services.agents.dafny_architect import DafnyArchitectAgent
        from backend.services.llm.client import LLMClient
        from backend.services.verification.dafny_runner import DafnyRunner
        from uuid import uuid4

        llm = LLMClient()
        architect = DafnyArchitectAgent(llm)
        runner = DafnyRunner()

        requirement = _load_requirement(prompt_id)

        # Ask architect to write a Dafny spec
        spec = await architect.run(
            run_id=uuid4(),
            source_code=source_code,
            requirement=requirement,
            language="Python",
        )

        if not spec or not spec.dafny_source:
            return False

        result = await runner.verify(spec.dafny_source)
        return bool(result and result.verified)

    except Exception as e:
        print(f"  [dafny error] {e}")
        return False


# ---------------------------------------------------------------------------
# CSS composite
# ---------------------------------------------------------------------------

def compute_css(fv: float, vs: float, pa: float, ec: float) -> float:
    return 0.35 * fv + 0.30 * vs + 0.25 * pa + 0.10 * ec


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def count_loc(code: str) -> int:
    return len([l for l in code.splitlines() if l.strip() and not l.strip().startswith("#")])


async def evaluate_file(json_path: Path, *, force: bool = False) -> None:
    record = json.loads(json_path.read_text())

    if not force and record.get("VS") is not None:
        print(f"  [skip] {json_path.name} already evaluated")
        return

    code = record.get("source_code", "")
    if not code.strip():
        print(f"  [skip] {json_path.name} has no source code")
        return

    prompt_id = record["prompt_id"]
    standard = "DO_178C"  # default; prompts.yaml has per-prompt standard
    loc = count_loc(code)

    print(f"  evaluating {json_path.name} ...")

    # Write code to temp file for Bandit and pytest
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix=f"hpema_eval_{prompt_id}_"
    ) as tf:
        tf.write(code)
        tmp_path = Path(tf.name)

    try:
        # --- Bandit (VS) ---
        bandit = run_bandit(tmp_path)
        vs = compute_vs(bandit["high"], bandit["medium"], bandit["low"], loc)

        # --- Edge tests (EC) ---
        passed, total = run_edge_tests(prompt_id, tmp_path)
        ec = passed / total if total > 0 else 0.0

        # --- Policy (PA) ---
        violations = await run_policy(code, standard)
        pa = compute_pa(violations)

        # --- Dafny (FV) ---
        # For HPEMA condition: use pipeline's own verification result if available
        if record.get("condition") == "hpema" and record.get("dafny_spec"):
            # Re-verify the existing spec rather than re-generating
            try:
                from backend.services.verification.dafny_runner import DafnyRunner
                runner = DafnyRunner()
                result = await runner.verify(record["dafny_spec"])
                fv = 1.0 if (result and result.verified) else 0.0
            except Exception as e:
                print(f"  [dafny re-verify error] {e}")
                fv = await run_dafny(code, prompt_id)
                fv = 1.0 if fv else 0.0
        else:
            fv_bool = await run_dafny(code, prompt_id)
            fv = 1.0 if fv_bool else 0.0

        css = compute_css(fv, vs, pa, ec)

        # Update record
        record.update({
            "lines_of_code": loc,
            "bandit_high": bandit["high"],
            "bandit_medium": bandit["medium"],
            "bandit_low": bandit["low"],
            "vulnerability_density": round(
                (3 * bandit["high"] + bandit["medium"] + 0.2 * bandit["low"]) / max(loc, 1), 6
            ),
            "VS": round(vs, 4),
            "policy_violations": violations,
            "PA": round(pa, 4),
            "edge_tests_passed": passed,
            "edge_tests_total": total,
            "EC": round(ec, 4),
            "dafny_verified": bool(fv),
            "FV": round(fv, 4),
            "CSS": round(css, 4),
        })

        json_path.write_text(json.dumps(record, indent=2))
        print(f"  [done] VS={vs:.3f} PA={pa:.3f} EC={ec:.3f} FV={fv:.1f} CSS={css:.3f}")

    finally:
        tmp_path.unlink(missing_ok=True)


async def main_async(paths: list[Path], *, force: bool) -> None:
    for p in paths:
        print(f"\n[evaluate] {p.name}")
        await evaluate_file(p, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate artifact JSON files")
    parser.add_argument("files", nargs="*", help="Specific JSON files to evaluate")
    parser.add_argument("--all", action="store_true", help="Evaluate all files in outputs/")
    parser.add_argument("--upto", metavar="PROMPT_ID",
                        help="Evaluate only prompts up to and including this ID (YAML order)")
    parser.add_argument("--model", metavar="MODEL_ID",
                        help="Restrict to a specific model (e.g. mistral-large)")
    parser.add_argument("--condition", choices=["solo", "hpema"],
                        help="Restrict to solo or hpema condition")
    parser.add_argument("--force", action="store_true", help="Re-evaluate even if VS is set")
    parser.add_argument("--flush", action="store_true",
                        help="Reset all evaluate.py fields to null (does not re-evaluate)")
    args = parser.parse_args()

    if args.all:
        paths = sorted(OUTPUTS_DIR.rglob("*.json"))
    elif args.files:
        paths = [Path(f) for f in args.files]
    else:
        parser.error("Pass file paths, --all, or --all --upto <prompt_id>")

    # --upto: keep only prompts at or before the named one in YAML order
    if args.upto:
        paths = _filter_upto(paths, args.upto)

    # --model / --condition: narrow by reading the JSON record
    if args.model or args.condition:
        filtered = []
        for p in paths:
            try:
                rec = json.loads(p.read_text())
            except Exception:
                continue
            if args.model and rec.get("model") != args.model:
                continue
            if args.condition and rec.get("condition") != args.condition:
                continue
            filtered.append(p)
        paths = filtered

    if not paths:
        print("No files to evaluate.")
        return

    if args.flush:
        _EVAL_FIELDS = [
            "bandit_high", "bandit_medium", "bandit_low", "vulnerability_density",
            "VS", "policy_violations", "PA",
            "edge_tests_passed", "edge_tests_total", "EC",
            "dafny_verified", "FV", "CSS",
        ]
        flushed = 0
        for p in paths:
            try:
                rec = json.loads(p.read_text())
                if any(rec.get(f) is not None for f in _EVAL_FIELDS):
                    for f in _EVAL_FIELDS:
                        rec[f] = None
                    p.write_text(json.dumps(rec, indent=2))
                    flushed += 1
            except Exception as e:
                print(f"  [warn] {p.name}: {e}")
        print(f"Flushed {flushed}/{len(paths)} file(s).")
        return

    print(f"Evaluating {len(paths)} artifact(s).")
    asyncio.run(main_async(paths, force=args.force))


if __name__ == "__main__":
    main()
