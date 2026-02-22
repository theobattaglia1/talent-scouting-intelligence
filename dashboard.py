from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parent
SCORED = ROOT / "outputs" / "scored.csv"
SNAPSHOTS = ROOT / "outputs" / "snapshots.jsonl"
ALERTS = ROOT / "outputs" / "alerts.jsonl"
TASTEMAKERS = ROOT / "outputs" / "tastemakers.csv"


def load_scored() -> list[dict[str, Any]]:
    if not SCORED.exists():
        return []
    with SCORED.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_snapshots() -> list[dict[str, Any]]:
    if not SNAPSHOTS.exists():
        return []
    rows: list[dict[str, Any]] = []
    with SNAPSHOTS.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_alerts() -> list[dict[str, Any]]:
    if not ALERTS.exists():
        return []
    rows: list[dict[str, Any]] = []
    with ALERTS.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_tastemakers() -> list[dict[str, Any]]:
    if not TASTEMAKERS.exists():
        return []
    with TASTEMAKERS.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def summarize_platforms(snapshots: list[dict[str, Any]]) -> dict[str, set[str]]:
    platforms: dict[str, set[str]] = defaultdict(set)
    for row in snapshots:
        platforms[row["track_id"]].add(row["platform"])
    return platforms


def summarize_regions(snapshots: list[dict[str, Any]]) -> dict[str, set[str]]:
    regions: dict[str, set[str]] = defaultdict(set)
    for row in snapshots:
        for region, value in (row.get("region_metrics") or {}).items():
            if value > 0:
                regions[row["track_id"]].add(region)
    return regions


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def main() -> None:
    st.set_page_config(page_title="Talent Scouting Intelligence", layout="wide")
    st.title("Talent Scouting Intelligence Dashboard")

    scored = load_scored()
    snapshots = load_snapshots()
    alerts = load_alerts()
    tastemakers = load_tastemakers()

    if not scored:
        st.warning("No scored output found. Run: tsi run-all --config configs/default.yaml")
        return

    platform_map = summarize_platforms(snapshots)
    region_map = summarize_regions(snapshots)

    genres = sorted({row.get("genre", "unknown") for row in scored})
    stages = ["early", "emerging", "breaking"]
    platforms = sorted({platform for values in platform_map.values() for platform in values})
    regions = sorted({region for values in region_map.values() for region in values})

    col1, col2, col3, col4 = st.columns(4)
    selected_genres = col1.multiselect("Genre", genres, default=genres)
    selected_stages = col2.multiselect("Stage", stages, default=stages)
    selected_platforms = col3.multiselect("Platform", platforms, default=platforms)
    selected_regions = col4.multiselect("Region", regions, default=[])

    flag_col1, flag_col2, flag_col3 = st.columns(3)
    hide_spike = flag_col1.checkbox("Hide spike-only", value=True)
    hide_suspicious = flag_col2.checkbox("Hide suspicious", value=True)
    hide_playlist_dep = flag_col3.checkbox("Hide playlist-dependent", value=False)

    filtered: list[dict[str, Any]] = []
    for row in scored:
        if row.get("genre") not in selected_genres:
            continue
        if row.get("stage") not in selected_stages:
            continue

        track_id = row["track_id"]
        if selected_platforms and not (platform_map.get(track_id, set()) & set(selected_platforms)):
            continue
        if selected_regions and not (region_map.get(track_id, set()) & set(selected_regions)):
            continue

        if hide_spike and as_bool(row.get("spike_only", "false")):
            continue
        if hide_suspicious and as_bool(row.get("suspicious", "false")):
            continue
        if hide_playlist_dep and as_bool(row.get("playlist_dependent", "false")):
            continue

        filtered.append(row)

    filtered.sort(key=lambda item: float(item.get("final_score", 0.0)), reverse=True)

    st.metric("Candidates", len(filtered))
    st.dataframe(
        [
            {
                "track": row.get("track_name"),
                "artist": row.get("artist_name"),
                "genre": row.get("genre"),
                "stage": row.get("stage"),
                "score": round(float(row.get("final_score", 0.0)), 3),
                "inflection": as_bool(row.get("inflection_detected", "false")),
                "spike_only": as_bool(row.get("spike_only", "false")),
                "suspicious": as_bool(row.get("suspicious", "false")),
                "playlist_dependent": as_bool(row.get("playlist_dependent", "false")),
                "why": row.get("explanation", ""),
            }
            for row in filtered
        ],
        use_container_width=True,
    )

    st.subheader("Scout Inbox")
    if alerts:
        st.dataframe(
            [
                {
                    "priority": item.get("priority"),
                    "type": item.get("type"),
                    "track": item.get("track_name"),
                    "artist": item.get("artist_name"),
                    "score": round(float(item.get("score", 0.0)), 3),
                    "stage": item.get("stage"),
                    "reason": item.get("reason"),
                }
                for item in alerts
            ],
            use_container_width=True,
        )
    else:
        st.info("No alerts yet. Run: tsi alerts --config configs/default.yaml")

    st.subheader("Top Tastemakers")
    if tastemakers:
        tastemakers.sort(key=lambda row: float(row.get("quant_score", 0.0)), reverse=True)
        st.dataframe(
            [
                {
                    "tastemaker": row.get("tastemaker_name"),
                    "status": row.get("status"),
                    "quant_score": round(float(row.get("quant_score", 0.0)), 3),
                    "bayes_precision": round(float(row.get("bayes_precision", 0.0)), 3),
                    "avg_lead_days": round(float(row.get("avg_lead_days", 0.0)), 2),
                    "trials": int(float(row.get("trials", 0))),
                }
                for row in tastemakers[:20]
            ],
            use_container_width=True,
        )
    else:
        st.info("No tastemaker profiles yet. Run: tsi tastemakers --config configs/default.yaml")


if __name__ == "__main__":
    main()
