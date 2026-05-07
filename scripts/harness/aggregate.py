#!/usr/bin/env python3
"""Aggregate per-artifact JSON files into summary metrics.

Computes:
  - Mean VS, PA, EC, FV, CSS per (model, condition, prompt)
  - 95% confidence intervals
  - Delta CSS (hpema - solo) per (model, prompt)
  - Wilcoxon signed-rank test + rank-biserial effect size r

Outputs:
  data/harness/results/summary.csv    — per-(model, condition, prompt) rows
  data/harness/results/delta.csv      — per-(model, prompt) ΔCSS rows
  data/harness/results/wilcoxon.csv   — per-model statistical test results

Usage:
    python scripts/harness/aggregate.py
    python scripts/harness/aggregate.py --outputs data/harness/outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUTS_DIR = PROJECT_ROOT / "data" / "harness" / "outputs"
RESULTS_DIR = PROJECT_ROOT / "data" / "harness" / "results"

# ---------------------------------------------------------------------------
# DO-178C rule count — measured from data/chromadb on 2026-04-27.
# Re-run: python scripts/harness/count_rules.py if standards corpus changes.
# ---------------------------------------------------------------------------
R_RULES = 61


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def sample_std(xs: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def ci95(xs: list[float]) -> float:
    """Half-width of 95% CI: 1.96 * std / sqrt(n)."""
    n = len(xs)
    if n < 2:
        return float("nan")
    return 1.96 * sample_std(xs) / math.sqrt(n)


def wilcoxon_signed_rank(deltas: list[float]) -> tuple[float, float, float]:
    """Manual Wilcoxon signed-rank test for small N.

    Returns (W_plus, p_approx, r) where:
      W_plus  = sum of positive ranks
      p_approx = approximate two-sided p-value using normal approximation
      r       = rank-biserial correlation (effect size)
    """
    non_zero = [d for d in deltas if d != 0.0]
    n = len(non_zero)
    if n == 0:
        return 0.0, 1.0, 0.0

    # Rank by absolute value
    sorted_by_abs = sorted(enumerate(non_zero), key=lambda t: abs(t[1]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and abs(sorted_by_abs[j][1]) == abs(sorted_by_abs[i][1]):
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[sorted_by_abs[k][0]] = avg_rank
        i = j

    W_plus = sum(ranks[k] for k, (_, d) in enumerate(sorted_by_abs) if d > 0)
    W_minus = sum(ranks[k] for k, (_, d) in enumerate(sorted_by_abs) if d < 0)

    # Normal approximation (valid for n >= 10)
    W = min(W_plus, W_minus)
    mu_W = n * (n + 1) / 4.0
    sigma_W = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0) if n > 0 else 1.0
    z = (W - mu_W) / sigma_W if sigma_W > 0 else 0.0

    # Two-sided p-value approximation via standard normal CDF
    # Using Abramowitz & Stegun approximation
    def _norm_cdf(x: float) -> float:
        t = 1.0 / (1.0 + 0.2316419 * abs(x))
        poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
        pdf = math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
        cdf = 1.0 - pdf * poly
        return cdf if x >= 0 else 1.0 - cdf

    p = 2.0 * _norm_cdf(-abs(z))

    # Effect size: r = |Z| / sqrt(n)  (bounded [0,1]; standard for Wilcoxon)
    # Note: eval_plan.md states W+/C(n,2) but that can exceed 1 when most
    # deltas are positive. Z-based formula is the accepted bounded alternative.
    r = abs(z) / math.sqrt(n) if n > 0 else 0.0

    return W_plus, p, r


# ---------------------------------------------------------------------------
# CSS computation (mirrors evaluate.py)
# ---------------------------------------------------------------------------

def compute_css(fv: float, vs: float, pa: float, ec: float) -> float:
    return 0.35 * fv + 0.30 * vs + 0.25 * pa + 0.10 * ec


# ---------------------------------------------------------------------------
# Load and group artifacts
# ---------------------------------------------------------------------------

def load_artifacts(outputs_dir: Path) -> list[dict]:
    records = []
    for p in sorted(outputs_dir.rglob("*.json")):
        try:
            rec = json.loads(p.read_text())
            if rec.get("VS") is not None:  # only evaluated artifacts
                records.append(rec)
        except Exception as e:
            print(f"[warn] could not read {p.name}: {e}")
    return records


def group_by(records: list[dict], keys: tuple[str, ...]) -> dict:
    groups: dict = defaultdict(list)
    for r in records:
        k = tuple(r.get(k) for k in keys)
        groups[k].append(r)
    return groups


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def build_summary(records: list[dict]) -> list[dict]:
    """One row per (model, condition, prompt_id) — averaged over reps."""
    groups = group_by(records, ("model", "condition", "prompt_id"))
    rows = []
    for (model, condition, prompt_id), recs in sorted(groups.items()):
        def vals(key):
            return [r[key] for r in recs if r.get(key) is not None]

        fv_vals = vals("FV")
        vs_vals = vals("VS")
        pa_vals = vals("PA")
        ec_vals = vals("EC")
        total_tok_vals = vals("total_tokens")
        cost_vals = vals("estimated_cost_usd")
        iter_vals = vals("iterations_used")

        fv_m = mean(fv_vals)
        vs_m = mean(vs_vals)
        pa_m = mean(pa_vals)
        ec_m = mean(ec_vals)
        css_m = compute_css(fv_m, vs_m, pa_m, ec_m)
        tok_m = mean(total_tok_vals)
        cost_m = mean(cost_vals)

        # Quality-per-token: CSS per 1000 tokens (higher = more efficient)
        css_per_1k = round(css_m / (tok_m / 1000), 4) if tok_m > 0 else float("nan")

        rows.append({
            "model": model,
            "condition": condition,
            "prompt_id": prompt_id,
            "difficulty": recs[0].get("difficulty", ""),
            "n_reps": len(recs),
            # Safety quality metrics
            "FV_mean": round(fv_m, 4),
            "FV_ci95": round(ci95(fv_vals), 4),
            "VS_mean": round(vs_m, 4),
            "VS_ci95": round(ci95(vs_vals), 4),
            "PA_mean": round(pa_m, 4),
            "PA_ci95": round(ci95(pa_vals), 4),
            "EC_mean": round(ec_m, 4),
            "EC_ci95": round(ci95(ec_vals), 4),
            "CSS_mean": round(css_m, 4),
            # Pipeline efficiency metrics
            "iter_mean": round(mean(iter_vals), 2) if iter_vals else float("nan"),
            "total_tokens_mean": round(tok_m, 0) if tok_m == tok_m else float("nan"),
            "cost_usd_mean": round(cost_m, 6) if cost_m == cost_m else float("nan"),
            "css_per_1k_tokens": css_per_1k,
        })
    return rows


# ---------------------------------------------------------------------------
# Delta CSV
# ---------------------------------------------------------------------------

def build_delta(summary_rows: list[dict]) -> list[dict]:
    """One row per (model, prompt_id) with ΔCSS = CSS_hpema - CSS_solo."""
    solo = {(r["model"], r["prompt_id"]): r for r in summary_rows if r["condition"] == "solo"}
    hpema = {(r["model"], r["prompt_id"]): r for r in summary_rows if r["condition"] == "hpema"}

    rows = []
    for key in sorted(set(solo) | set(hpema)):
        model, prompt_id = key
        s = solo.get(key)
        h = hpema.get(key)
        if s and h:
            delta_css = h["CSS_mean"] - s["CSS_mean"]
            delta_fv = h["FV_mean"] - s["FV_mean"]
            delta_vs = h["VS_mean"] - s["VS_mean"]
            delta_pa = h["PA_mean"] - s["PA_mean"]
            delta_ec = h["EC_mean"] - s["EC_mean"]

            # Business metrics: token overhead and marginal CSS-per-dollar
            def _f(r, key):
                v = r.get(key)
                try:
                    f = float(v)
                    return f if not math.isnan(f) else None
                except (TypeError, ValueError):
                    return None

            tok_solo = _f(s, "total_tokens_mean")
            tok_hpema = _f(h, "total_tokens_mean")
            cost_solo = _f(s, "cost_usd_mean") or 0.0
            cost_hpema = _f(h, "cost_usd_mean") or 0.0
            incremental_cost = cost_hpema - cost_solo

            token_overhead = (
                round(tok_hpema / tok_solo, 4)
                if tok_solo and tok_solo > 0 and tok_hpema is not None
                else float("nan")
            )
            delta_css_per_usd = (
                round(delta_css / incremental_cost, 4)
                if incremental_cost > 0
                else float("nan")
            )

            rows.append({
                "model": model,
                "prompt_id": prompt_id,
                "difficulty": s["difficulty"],
                "CSS_solo": s["CSS_mean"],
                "CSS_hpema": h["CSS_mean"],
                "delta_CSS": round(delta_css, 4),
                "delta_FV": round(delta_fv, 4),
                "delta_VS": round(delta_vs, 4),
                "delta_PA": round(delta_pa, 4),
                "delta_EC": round(delta_ec, 4),
                # Business metrics (tracked separately from CSS formula)
                "tokens_solo": round(tok_solo, 0) if tok_solo is not None else float("nan"),
                "tokens_hpema": round(tok_hpema, 0) if tok_hpema is not None else float("nan"),
                "token_overhead": token_overhead,
                "cost_solo_usd": round(cost_solo, 6),
                "cost_hpema_usd": round(cost_hpema, 6),
                "incremental_cost_usd": round(incremental_cost, 6),
                "delta_css_per_usd": delta_css_per_usd,
            })
    return rows


# ---------------------------------------------------------------------------
# Wilcoxon CSV
# ---------------------------------------------------------------------------

def build_wilcoxon(delta_rows: list[dict]) -> list[dict]:
    """One row per model: Wilcoxon test over 20-prompt ΔCSS vector + business metric aggregates."""
    by_model: dict[str, list] = defaultdict(list)
    for r in delta_rows:
        by_model[r["model"]].append(r)

    def _safe_mean(recs, key):
        vals = []
        for rec in recs:
            v = rec.get(key)
            try:
                f = float(v)
                if not math.isnan(f):
                    vals.append(f)
            except (TypeError, ValueError):
                pass
        return mean(vals) if vals else float("nan")

    def _fmt(v, ndigits):
        return round(v, ndigits) if not math.isnan(v) else float("nan")

    rows = []
    for model, recs in sorted(by_model.items()):
        deltas = [r["delta_CSS"] for r in recs]
        w_plus, p, r = wilcoxon_signed_rank(deltas)
        mean_delta = mean(deltas)
        n = len(deltas)
        rows.append({
            "model": model,
            "n_prompts": n,
            "mean_delta_CSS": round(mean_delta, 4),
            "W_plus": round(w_plus, 2),
            "p_value": round(p, 4),
            "effect_r": round(r, 4),
            "significant_p05": "yes" if p < 0.05 else "no",
            "effect_size": "large" if r > 0.5 else ("medium" if r > 0.3 else "small"),
            # Business metrics averaged over all prompts for this model
            "mean_tokens_solo": _fmt(_safe_mean(recs, "tokens_solo"), 0),
            "mean_tokens_hpema": _fmt(_safe_mean(recs, "tokens_hpema"), 0),
            "mean_token_overhead": _fmt(_safe_mean(recs, "token_overhead"), 3),
            "mean_cost_solo_usd": _fmt(_safe_mean(recs, "cost_solo_usd"), 6),
            "mean_cost_hpema_usd": _fmt(_safe_mean(recs, "cost_hpema_usd"), 6),
            "mean_incremental_cost_usd": _fmt(_safe_mean(recs, "incremental_cost_usd"), 6),
            "mean_delta_css_per_usd": _fmt(_safe_mean(recs, "delta_css_per_usd"), 4),
        })
    return rows


# ---------------------------------------------------------------------------
# Write CSV helpers
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        print(f"  [warn] no rows for {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [wrote] {path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate harness results")
    parser.add_argument(
        "--outputs", default=str(OUTPUTS_DIR),
        help="Directory containing artifact JSON files",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs)
    records = load_artifacts(outputs_dir)
    if not records:
        print("No evaluated artifacts found. Run evaluate.py first.")
        return

    print(f"Loaded {len(records)} evaluated artifacts.")

    summary = build_summary(records)
    delta = build_delta(summary)
    wilcoxon = build_wilcoxon(delta)

    write_csv(summary, RESULTS_DIR / "summary.csv")
    write_csv(delta, RESULTS_DIR / "delta.csv")
    write_csv(wilcoxon, RESULTS_DIR / "wilcoxon.csv")

    # Quick console preview
    print("\n=== Wilcoxon Results ===")
    for row in wilcoxon:
        print(
            f"  {row['model']:20s}  mean_ΔCSS={row['mean_delta_CSS']:+.3f}  "
            f"p={row['p_value']:.4f}  r={row['effect_r']:.3f}  "
            f"sig={row['significant_p05']}  effect={row['effect_size']}"
        )


if __name__ == "__main__":
    main()
