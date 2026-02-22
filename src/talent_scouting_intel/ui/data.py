from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from talent_scouting_intel.utils.io import load_config, read_csv, read_jsonl, resolve_path


BOOL_COLUMNS = {
    "manual_seeded",
    "inflection_detected",
    "spike_only",
    "suspicious",
    "playlist_dependent",
    "insufficient_history",
    "established_artist",
    "weak_evidence",
    "low_trust",
    "established_by_followers",
    "established_by_footprint",
    "prior_boost_capped",
}

NUMERIC_COLUMNS = {
    "final_score",
    "weighted_score",
    "penalty",
    "momentum_score",
    "acceleration_score",
    "depth_score",
    "cross_platform_score",
    "network_score",
    "consistency_score",
    "geo_score",
    "size_norm_accel",
    "spotify_points",
    "history_days",
    "consecutive_positive_accel",
    "comment_specificity",
    "echo_score",
    "anomaly_score",
    "established_penalty",
    "artist_track_footprint",
    "candidate_priority",
    "tastemaker_score",
    "shortform_proxy_score",
    "knowledge_graph_score",
    "affinity_score",
    "affinity_direct_score",
    "affinity_genre_baseline",
    "affinity_text_score",
    "affinity_match_similarity",
    "path_similarity_score",
    "path_template_similarity",
    "weight_affinity",
    "weight_path_similarity",
    "base_final_score",
    "prior_gate",
    "prior_boost",
    "creative_model_score",
    "blended_model_score",
    "affinity_boost",
    "path_similarity_boost",
    "weight_momentum_engine",
    "weight_creative_fit",
    "trust_score",
    "trust_base",
    "trust_penalty",
    "trust_history",
    "trust_evidence",
    "trust_corroboration",
    "trust_data_quality",
    "creator_reuse_growth",
    "engagement_rate",
    "follower_conversion",
    "views_recent",
    "likes_recent",
    "comments_recent",
    "spotify_growth_share",
    "playlist_ratio",
    "quant_score",
    "bayes_precision",
    "avg_lead_days",
    "lead_score",
    "early_capture",
    "novelty_ratio",
    "genre_alignment",
    "reliability",
    "centrality",
}


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _coerce_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    for column in df.columns:
        if column in BOOL_COLUMNS:
            df[column] = df[column].map(_parse_bool)
        elif column in NUMERIC_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    return df


def load_outputs(config_path: str, project_root: Path, *, load_snapshots: bool = False) -> dict[str, Any]:
    config = load_config(config_path)

    def path(key: str) -> Path:
        return resolve_path(config["paths"][key], project_root)

    scored = _coerce_frame(read_csv(path("scored")))
    features = _coerce_frame(read_csv(path("features")))
    candidates = _coerce_frame(read_jsonl(path("candidates")))
    alerts = _coerce_frame(read_jsonl(path("alerts_jsonl")))
    tastemakers = _coerce_frame(read_csv(path("tastemakers_csv")))
    inflections = _coerce_frame(read_jsonl(path("inflections")))

    snapshots = pd.DataFrame()
    if load_snapshots:
        snapshots_rows = read_jsonl(path("snapshots"))
        snapshots = _coerce_frame(snapshots_rows)
        if "date" in snapshots.columns:
            snapshots["date"] = pd.to_datetime(snapshots["date"], errors="coerce")
    if "date" in inflections.columns:
        inflections["date"] = pd.to_datetime(inflections["date"], errors="coerce")

    backtest_payload: dict[str, Any] = {}
    backtest_path = path("backtest_json")
    if backtest_path.exists():
        try:
            loaded = json.loads(backtest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                backtest_payload = loaded
        except Exception:
            backtest_payload = {}

    source_registry_path = resolve_path(config["ingest"]["auto"]["source_registry"], project_root)
    source_registry: dict[str, Any] = {}
    if source_registry_path.exists():
        try:
            loaded = json.loads(source_registry_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                source_registry = loaded
        except Exception:
            source_registry = {}

    return {
        "config": config,
        "scored": scored,
        "features": features,
        "candidates": candidates,
        "alerts": alerts,
        "tastemakers": tastemakers,
        "inflections": inflections,
        "snapshots": snapshots,
        "backtest": backtest_payload,
        "source_registry": source_registry,
    }


def load_snapshots(config_path: str, project_root: Path) -> pd.DataFrame:
    config = load_config(config_path)
    snapshot_path = resolve_path(config["paths"]["snapshots"], project_root)
    snapshots_rows = read_jsonl(snapshot_path)
    snapshots = _coerce_frame(snapshots_rows)
    if "date" in snapshots.columns:
        snapshots["date"] = pd.to_datetime(snapshots["date"], errors="coerce")
    return snapshots


def track_platform_map(snapshots: pd.DataFrame) -> dict[str, list[str]]:
    if snapshots.empty or "track_id" not in snapshots.columns:
        return {}
    if "platform" not in snapshots.columns:
        return {}

    platform_map: dict[str, set[str]] = {}
    for _, row in snapshots[["track_id", "platform"]].dropna().iterrows():
        track_id = str(row["track_id"])
        platform = str(row["platform"])
        platform_map.setdefault(track_id, set()).add(platform)

    return {track_id: sorted(values) for track_id, values in platform_map.items()}


def track_region_map(snapshots: pd.DataFrame) -> dict[str, list[str]]:
    if snapshots.empty or "track_id" not in snapshots.columns:
        return {}
    if "region_metrics" not in snapshots.columns:
        return {}

    region_map: dict[str, set[str]] = {}
    for _, row in snapshots[["track_id", "region_metrics"]].iterrows():
        metrics = row["region_metrics"]
        if not isinstance(metrics, dict):
            continue
        track_id = str(row["track_id"])
        for region, value in metrics.items():
            try:
                if float(value) > 0:
                    region_map.setdefault(track_id, set()).add(str(region))
            except Exception:
                continue

    return {track_id: sorted(values) for track_id, values in region_map.items()}


def sparkline(values: list[float]) -> str:
    if not values:
        return ""
    ticks = "▁▂▃▄▅▆▇█"
    minimum = min(values)
    maximum = max(values)
    if maximum <= minimum:
        return ticks[0] * len(values)

    output = []
    for value in values:
        scaled = (value - minimum) / (maximum - minimum)
        idx = int(round(scaled * (len(ticks) - 1)))
        output.append(ticks[idx])
    return "".join(output)


def track_sparkline_map(snapshots: pd.DataFrame, days: int = 28) -> dict[str, str]:
    if snapshots.empty:
        return {}

    required = {"track_id", "platform", "date", "streams", "views"}
    if not required.issubset(set(snapshots.columns)):
        return {}

    rows = snapshots.copy()
    rows = rows.dropna(subset=["track_id", "date", "platform"])
    rows = rows.sort_values("date")
    out: dict[str, str] = {}

    for track_id, group in rows.groupby("track_id"):
        spotify = group[group["platform"] == "spotify"].tail(days)
        if not spotify.empty:
            values = pd.to_numeric(spotify["streams"], errors="coerce").fillna(0.0).tolist()
            out[str(track_id)] = sparkline(values)
            continue

        fallback = group.tail(days)
        values = pd.to_numeric(fallback["views"], errors="coerce").fillna(0.0).tolist()
        out[str(track_id)] = sparkline(values)

    return out


def find_track_row(scored: pd.DataFrame, track_id: str) -> dict[str, Any] | None:
    if scored.empty:
        return None
    match = scored[scored["track_id"].astype(str) == str(track_id)]
    if match.empty:
        return None
    return dict(match.iloc[0])


def track_timeseries(snapshots: pd.DataFrame, track_id: str) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame(columns=["date", "platform", "value"])

    rows = snapshots[snapshots["track_id"].astype(str) == str(track_id)].copy()
    if rows.empty:
        return pd.DataFrame(columns=["date", "platform", "value"])

    rows["streams"] = pd.to_numeric(rows.get("streams", 0.0), errors="coerce").fillna(0.0)
    rows["views"] = pd.to_numeric(rows.get("views", 0.0), errors="coerce").fillna(0.0)
    rows["value"] = rows.apply(lambda r: r["streams"] if r["platform"] == "spotify" else r["views"], axis=1)
    rows = rows[["date", "platform", "value"]].sort_values("date")
    return rows


def artist_timeseries(snapshots: pd.DataFrame, artist_id: str) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame(columns=["date", "platform", "value"])

    rows = snapshots[snapshots["artist_id"].astype(str) == str(artist_id)].copy()
    if rows.empty:
        return pd.DataFrame(columns=["date", "platform", "value"])

    rows["streams"] = pd.to_numeric(rows.get("streams", 0.0), errors="coerce").fillna(0.0)
    rows["views"] = pd.to_numeric(rows.get("views", 0.0), errors="coerce").fillna(0.0)
    rows["value"] = rows.apply(lambda r: r["streams"] if r["platform"] == "spotify" else r["views"], axis=1)
    grouped = rows.groupby(["date", "platform"], as_index=False)["value"].sum()
    return grouped.sort_values("date")
