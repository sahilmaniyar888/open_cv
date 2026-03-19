"""Streamlit monitoring dashboard for AdaptTrack."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np


def load_tracking_log(log_path: str) -> list[dict]:
    """Load tracking log from JSON lines file."""
    entries = []
    path = Path(log_path)
    if not path.exists():
        return entries
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def main():
    st.set_page_config(page_title="AdaptTrack Monitor", layout="wide")
    st.title("AdaptTrack - Tracking Monitor")

    # sidebar config
    st.sidebar.header("Configuration")
    log_path = st.sidebar.text_input(
        "Log file path", value="tracking_log.jsonl"
    )

    if not Path(log_path).exists():
        st.info("No tracking log found. Run the tracker with `--log` to generate one.")
        st.markdown("```bash\npython main.py --webcam --log tracking_log.jsonl\n```")

        # show demo data
        st.subheader("Demo Dashboard")
        demo_data = _generate_demo_data()
        _render_dashboard(demo_data)
        return

    entries = load_tracking_log(log_path)
    if not entries:
        st.warning("Log file is empty.")
        return

    _render_dashboard(entries)


def _render_dashboard(entries: list[dict]):
    """Render the main dashboard from log entries."""

    col1, col2, col3, col4 = st.columns(4)

    frames = [e.get("frame", i) for i, e in enumerate(entries)]
    active_counts = [e.get("active_tracks", 0) for e in entries]
    uncertainties = [e.get("avg_uncertainty", 0) for e in entries]
    fps_values = [e.get("fps", 0) for e in entries]
    failure_counts = [e.get("failure_events", 0) for e in entries]

    col1.metric("Total Frames", len(entries))
    col2.metric("Avg FPS", f"{np.mean(fps_values):.1f}" if fps_values else "N/A")
    col3.metric("Avg Active Tracks", f"{np.mean(active_counts):.1f}")
    col4.metric("Total Failures", sum(failure_counts))

    # track count over time
    st.subheader("Active Tracks Over Time")
    df_tracks = pd.DataFrame({"Frame": frames, "Active Tracks": active_counts})
    st.line_chart(df_tracks.set_index("Frame"))

    # uncertainty over time
    st.subheader("Average Uncertainty Over Time")
    df_unc = pd.DataFrame({"Frame": frames, "Uncertainty": uncertainties})
    st.line_chart(df_unc.set_index("Frame"))

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("FPS Over Time")
        df_fps = pd.DataFrame({"Frame": frames, "FPS": fps_values})
        st.line_chart(df_fps.set_index("Frame"))

    with col_b:
        st.subheader("Failure Events")
        df_fail = pd.DataFrame({"Frame": frames, "Failures": failure_counts})
        st.bar_chart(df_fail.set_index("Frame"))

    # failure event details
    all_events = []
    for e in entries:
        for evt in e.get("events", []):
            all_events.append(evt)

    if all_events:
        st.subheader("Failure Event Log")
        df_events = pd.DataFrame(all_events)
        st.dataframe(df_events, use_container_width=True)

    # metrics summary if available
    metrics = entries[-1].get("metrics", {}) if entries else {}
    if metrics:
        st.subheader("Evaluation Metrics")
        cols = st.columns(4)
        for i, (k, v) in enumerate(metrics.items()):
            if isinstance(v, float):
                cols[i % 4].metric(k.upper(), f"{v:.4f}")


def _generate_demo_data() -> list[dict]:
    """Generate demo data for the dashboard."""
    np.random.seed(42)
    entries = []
    for i in range(200):
        active = max(0, int(5 + 3 * np.sin(i / 20.0) + np.random.normal(0, 1)))
        unc = 0.2 + 0.15 * np.sin(i / 30.0) + np.random.normal(0, 0.05)
        fps = 25 + np.random.normal(0, 3)
        failures = 1 if np.random.random() < 0.05 else 0

        entries.append({
            "frame": i,
            "active_tracks": active,
            "avg_uncertainty": max(0, min(1, unc)),
            "fps": max(0, fps),
            "failure_events": failures,
            "events": [{"frame": i, "type": "high_uncertainty", "severity": 0.7}] if failures else [],
        })
    return entries


if __name__ == "__main__":
    main()
