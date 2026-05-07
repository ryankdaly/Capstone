#!/usr/bin/env python3
"""Render aggregated results as Markdown and LaTeX tables.

Reads from data/harness/results/ (output of aggregate.py).
Writes to data/harness/results/report.md and report.tex.

Usage:
    python scripts/harness/report.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "data" / "harness" / "results"

MODEL_DISPLAY = {
    "mistral-large": "Mistral-Large-3",
    "qwen3-coder": "Qwen3-Coder-480B",
    "gemma-3-27b": "Gemma-3-27B",
    "deepseek-v4-pro": "DeepSeek-V4-Pro",
    "deepseek-v4-flash": "DeepSeek-V4-Flash",
    "gpt-5.4-mini": "GPT-5.4-mini",
}

# Ordered for consistent table output (Stage 1 → 2 → 4)
MODEL_ORDER = [
    "mistral-large",
    "qwen3-coder",
    "gemma-3-27b",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "gpt-5.4-mini",
]

CONDITION_DISPLAY = {
    "solo": "Solo",
    "hpema": "HPEMA",
}


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[warn] {path} not found — run aggregate.py first")
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Table 1: Main results (model × condition, averaged over all prompts)
# ---------------------------------------------------------------------------

def build_main_table_rows(summary: list[dict]) -> list[dict]:
    """Average metrics over all prompts per (model, condition)."""
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for r in summary:
        groups[(r["model"], r["condition"])].append(r)

    def avg(rows, key):
        vals = [float(r[key]) for r in rows if r.get(key) not in (None, "", "nan")]
        return sum(vals) / len(vals) if vals else float("nan")

    rows = []
    for model in MODEL_ORDER:
        for condition in ["solo", "hpema"]:
            recs = groups.get((model, condition), [])
            if not recs:
                continue
            rows.append({
                "Model": MODEL_DISPLAY.get(model, model),
                "Condition": CONDITION_DISPLAY.get(condition, condition),
                "FV": f"{avg(recs, 'FV_mean'):.3f}",
                "VS": f"{avg(recs, 'VS_mean'):.3f}",
                "PA": f"{avg(recs, 'PA_mean'):.3f}",
                "EC": f"{avg(recs, 'EC_mean'):.3f}",
                "CSS": f"{avg(recs, 'CSS_mean'):.3f}",
            })
    return rows


def markdown_table(headers: list[str], rows: list[dict]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


def latex_table(headers: list[str], rows: list[dict], caption: str, label: str) -> str:
    col_fmt = "l" * len(headers)
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
        " & ".join(f"\\textbf{{{h}}}" for h in headers) + r" \\",
        r"\midrule",
    ]
    prev_model = None
    for r in rows:
        model = r.get("Model", "")
        if prev_model and model != prev_model:
            lines.append(r"\midrule")
        prev_model = model
        lines.append(" & ".join(str(r.get(h, "")) for h in headers) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 2: Delta table (ΔCSS per model averaged over difficulty tiers)
# ---------------------------------------------------------------------------

def build_delta_by_tier(delta: list[dict]) -> list[dict]:
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for r in delta:
        groups[(r["model"], r.get("difficulty", "L?"))].append(float(r["delta_CSS"]))

    rows = []
    for model in MODEL_ORDER:
        row = {"Model": MODEL_DISPLAY.get(model, model)}
        for tier in ["L1", "L2", "L3", "L4"]:
            vals = groups.get((model, tier), [])
            row[f"ΔCSS {tier}"] = f"{sum(vals)/len(vals):+.3f}" if vals else "—"
        all_vals = [float(r["delta_CSS"]) for r in delta if r["model"] == model]
        row["ΔCSS All"] = f"{sum(all_vals)/len(all_vals):+.3f}" if all_vals else "—"
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Table 3: Wilcoxon results
# ---------------------------------------------------------------------------

def build_wilcoxon_table(wilcoxon: list[dict]) -> list[dict]:
    rows = []
    for r in wilcoxon:
        rows.append({
            "Model": MODEL_DISPLAY.get(r["model"], r["model"]),
            "N": r["n_prompts"],
            "Mean ΔCSS": f"{float(r['mean_delta_CSS']):+.3f}",
            "W+": r["W_plus"],
            "p": r["p_value"],
            "r": r["effect_r"],
            "Significant": r["significant_p05"],
            "Effect": r["effect_size"],
        })
    return rows


# ---------------------------------------------------------------------------
# Table 4: Business metrics (token efficiency, cost overhead, CSS per dollar)
# ---------------------------------------------------------------------------

def build_business_table(summary: list[dict], wilcoxon: list[dict]) -> list[dict]:
    """One row per model: efficiency and cost metrics."""
    from collections import defaultdict

    # Average css_per_1k_tokens per (model, condition) from summary
    groups: dict = defaultdict(list)
    for r in summary:
        v = r.get("css_per_1k_tokens")
        try:
            f = float(v)
            if not (f != f):  # not NaN
                groups[(r["model"], r["condition"])].append(f)
        except (TypeError, ValueError):
            pass

    def avg(lst):
        return sum(lst) / len(lst) if lst else float("nan")

    def _fmt(v, ndigits, prefix=""):
        if v != v:  # nan
            return "—"
        return f"{prefix}{v:.{ndigits}f}"

    # Build wilcoxon index for business fields
    wil_idx = {r["model"]: r for r in wilcoxon}

    rows = []
    for model in MODEL_ORDER:
        w = wil_idx.get(model, {})
        css_1k_solo = avg(groups.get((model, "solo"), []))
        css_1k_hpema = avg(groups.get((model, "hpema"), []))

        def _wf(key):
            v = w.get(key)
            try:
                f = float(v)
                return f if not (f != f) else float("nan")
            except (TypeError, ValueError):
                return float("nan")

        overhead = _wf("mean_token_overhead")
        incr_cost = _wf("mean_incremental_cost_usd")
        css_per_usd = _wf("mean_delta_css_per_usd")

        rows.append({
            "Model": MODEL_DISPLAY.get(model, model),
            "CSS/1k tok Solo": _fmt(css_1k_solo, 4),
            "CSS/1k tok HPEMA": _fmt(css_1k_hpema, 4),
            "Token Overhead": _fmt(overhead, 2) + "×" if overhead == overhead else "—",
            "Incr. Cost/prompt": _fmt(incr_cost, 5, "$") if incr_cost > 0 else "$0 (free)",
            "ΔCSS per $": _fmt(css_per_usd, 3) if css_per_usd == css_per_usd else "∞ (free)",
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    summary = read_csv(RESULTS_DIR / "summary.csv")
    delta = read_csv(RESULTS_DIR / "delta.csv")
    wilcoxon = read_csv(RESULTS_DIR / "wilcoxon.csv")

    # --- Markdown report ---
    md_lines = [
        "# HPEMA Evaluation Results",
        "",
        "Generated by `scripts/harness/report.py`. Run `aggregate.py` first.",
        "",
        "---",
        "",
        "## Table 1: Main Results (all prompts, averaged over N=5 repetitions)",
        "",
    ]

    main_rows = build_main_table_rows(summary)
    main_headers = ["Model", "Condition", "FV", "VS", "PA", "EC", "CSS"]
    md_lines.append(markdown_table(main_headers, main_rows))
    md_lines += [
        "",
        "> Weights: CSS = 0.35·FV + 0.30·VS + 0.25·PA + 0.10·EC",
        "",
        "---",
        "",
        "## Table 2: ΔCSS by Difficulty Tier (HPEMA − Solo)",
        "",
    ]

    delta_rows = build_delta_by_tier(delta)
    delta_headers = ["Model", "ΔCSS L1", "ΔCSS L2", "ΔCSS L3", "ΔCSS L4", "ΔCSS All"]
    md_lines.append(markdown_table(delta_headers, delta_rows))
    md_lines += [
        "",
        "> Positive = HPEMA better. Expected: small gap at L1/L2, large gap at L3/L4.",
        "",
        "---",
        "",
        "## Table 3: Wilcoxon Signed-Rank Test (20-prompt ΔCSS vector per model)",
        "",
    ]

    wilcoxon_rows = build_wilcoxon_table(wilcoxon)
    wilcoxon_headers = ["Model", "N", "Mean ΔCSS", "W+", "p", "r", "Significant", "Effect"]
    md_lines.append(markdown_table(wilcoxon_headers, wilcoxon_rows))
    md_lines += [
        "",
        "> H₀: median ΔCSS = 0. p < 0.05 rejects H₀. Effect size r: >0.3 medium, >0.5 large.",
        "",
        "---",
        "",
        "## Table 4: Business Metrics (Token Efficiency and Cost)",
        "",
    ]

    biz_rows = build_business_table(summary, wilcoxon)
    biz_headers = ["Model", "CSS/1k tok Solo", "CSS/1k tok HPEMA", "Token Overhead", "Incr. Cost/prompt", "ΔCSS per $"]
    md_lines.append(markdown_table(biz_headers, biz_rows))
    md_lines += [
        "",
        "> **CSS/1k tok**: quality per 1000 tokens (higher = more token-efficient). "
        "**Token Overhead**: HPEMA/solo token ratio. "
        "**Incr. Cost/prompt**: mean additional cost to run HPEMA vs solo. "
        "**ΔCSS per $**: marginal safety gain per incremental dollar; ∞ for free-tier models.",
        "",
    ]

    md_text = "\n".join(md_lines)

    md_path = RESULTS_DIR / "report.md"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_text)
    print(f"[wrote] {md_path}")

    # --- LaTeX report ---
    tex_lines = [
        r"\documentclass{article}",
        r"\usepackage{booktabs}",
        r"\begin{document}",
        "",
        latex_table(
            main_headers, main_rows,
            "HPEMA vs Solo: Main Safety Metrics (N=5 reps, all prompts averaged)",
            "tab:main_results",
        ),
        "",
        latex_table(
            delta_headers, delta_rows,
            r"$\Delta$CSS by Difficulty Tier (HPEMA $-$ Solo)",
            "tab:delta_tier",
        ),
        "",
        latex_table(
            wilcoxon_headers, wilcoxon_rows,
            "Wilcoxon Signed-Rank Test: 20-prompt $\\Delta$CSS per Model",
            "tab:wilcoxon",
        ),
        "",
        latex_table(
            biz_headers, biz_rows,
            "Business Metrics: Token Efficiency and Marginal Cost",
            "tab:business",
        ),
        "",
        r"\end{document}",
    ]

    tex_path = RESULTS_DIR / "report.tex"
    tex_path.write_text("\n".join(tex_lines))
    print(f"[wrote] {tex_path}")


if __name__ == "__main__":
    main()
