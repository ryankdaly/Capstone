"""HPEMA Audit Dashboard — Streamlit viewer for pipeline runs.

Reads from the same audit log as the CLI. Launch with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID
from datetime import datetime

import streamlit as st

AUDIT_LOG_DIR = Path("logs/audit")

# jsonl loader helper
def _load_jsonl(path: Path) -> list[dict]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries

def _parse_timestamp(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None

def _count_code_lines(source_code: str) -> int:
    if not source_code:
        return 0
    return len(source_code.splitlines())

def _format_ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def _summarize_run(run_entries: list[dict], run_id: str) -> dict:
    timestamps: list[datetime] = []
    agents: set[str] = set()

    status = "—"
    verdict = "—"
    code_lines = 0
    stage = "—"
    requirement = "—"

    latest_actor_code = ""

    for entry in run_entries:
        ts = _parse_timestamp(entry.get("timestamp", ""))
        if ts:
            timestamps.append(ts)

        agent = entry.get("agent")
        if agent:
            agents.add(agent)

        event_type = entry.get("event_type", "")
        data = entry.get("data", {}) or {}

        if event_type == "pipeline_start":
            requirement = data.get("requirement_text", "—")
            stage = data.get("stage", "—")

        elif event_type == "pipeline_complete":
            status = str(data.get("status", "—")).upper()
            stage = data.get("stage", stage)

        elif event_type == "agent_output":
            if agent == "actor":
                latest_actor_code = data.get("source_code", "") or ""
            elif agent == "checker":
                checker_verdict = data.get("verdict")
                if checker_verdict:
                    verdict = f"checker: {str(checker_verdict).upper()}"
            elif agent == "policy":
                compliant = data.get("compliant")
                if compliant is True:
                    verdict = "policy: COMPLIANT"
                elif compliant is False:
                    verdict = "policy: NON-COMPLIANT"

        elif event_type == "verification_result" and verdict == "—":
            verified = data.get("verified")
            if verified is True:
                verdict = "proof: VERIFIED"
            elif verified is False:
                verdict = "proof: FAILED"

    code_lines = _count_code_lines(latest_actor_code)

    started = _format_ts(min(timestamps)) if timestamps else "—"
    finished = _format_ts(max(timestamps)) if timestamps else "—"

    return {
        "run_id": run_id,
        "started": started,
        "finished": finished,
        "status": status,
        "verdict": verdict,
        "code_lines": code_lines,
        "stage": stage,
        "events": len(run_entries),
        "agents": ", ".join(sorted(agents)) if agents else "—",
        "requirement": requirement,
    }

st.set_page_config(page_title="HPEMA Audit Dashboard", layout="wide")
st.title("HPEMA Audit Dashboard")
st.markdown("View pipeline runs, agent outputs, and traceability matrices.")

# --- Sidebar: select a run ---
if AUDIT_LOG_DIR.exists():
    run_files = sorted(AUDIT_LOG_DIR.glob("*.jsonl"), reverse=True)
    run_ids = [f.stem for f in run_files]
else:
    run_files = []
    run_ids = []

if not run_ids:
    st.info("No pipeline runs found. Run a pipeline first to see results here.")
    st.stop()

selected_run = st.sidebar.selectbox("Select Pipeline Run", run_ids)

# --- Load audit entries ---
run_file = AUDIT_LOG_DIR / f"{selected_run}.jsonl"
entries = _load_jsonl(run_file)

st.sidebar.metric("Total Events", len(entries))

# Run history
st.subheader("Run History")
run_history = []
for path in run_files:
    run_entries = _load_jsonl(path)
    run_history.append(_summarize_run(run_entries, path.stem))
run_history.sort(key=lambda row: row["started"], reverse=True)
st.dataframe(
    run_history,
    use_container_width=True,
    hide_index=True,
    column_order=[
        "run_id",
        "started",
        "finished",
        "status",
        "verdict",
        "code_lines",
        "stage",
        "events",
        "requirement",
        "agents",
    ],
)

# --- Main view ---
st.header(f"Run: {selected_run}")
selected_summary = _summarize_run(entries, selected_run)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Status", selected_summary["status"])
col2.metric("Verdict", selected_summary["verdict"])
col3.metric("Code Lines", selected_summary["code_lines"])
col4.metric("Events", selected_summary["events"])

# Timeline
st.subheader("Event Timeline")
for entry in entries:
    event_type = entry.get("event_type", "")
    agent = entry.get("agent", "")
    timestamp = entry.get("timestamp", "")
    label = f"**{event_type}**" + (f" ({agent})" if agent else "")
    with st.expander(f"{timestamp} — {label}"):
        st.json(entry.get("data", {}))

# Agent outputs
st.subheader("Agent Outputs")
agent_outputs = [e for e in entries if e.get("event_type") == "agent_output"]
if agent_outputs:
    tabs = st.tabs([e.get("agent", "unknown") for e in agent_outputs])
    for tab, output in zip(tabs, agent_outputs):
        with tab:
            st.json(output.get("data", {}))
else:
    st.info("No agent outputs recorded for this run.")
