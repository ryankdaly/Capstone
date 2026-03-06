"""HPEMA Audit Dashboard — Streamlit viewer for pipeline runs.

Reads from the same audit log as the CLI. Launch with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import streamlit as st

AUDIT_LOG_DIR = Path("logs/audit")

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
entries = []
for line in run_file.read_text().splitlines():
    if line.strip():
        entries.append(json.loads(line))

st.sidebar.metric("Total Events", len(entries))

# --- Main view ---
st.header(f"Run: {selected_run}")

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
