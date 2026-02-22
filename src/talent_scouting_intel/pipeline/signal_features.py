from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from talent_scouting_intel.utils.io import read_csv, resolve_path, write_csv
from talent_scouting_intel.utils.math_utils import clamp01, minmax_scale


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def _to_date(value: Any) -> dt.date | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1]
    if "T" in raw:
        raw = raw.split("T", 1)[0]
    try:
        return dt.date.fromisoformat(raw)
    except Exception:
        return None


def _specificity(comments: list[str], terms: list[str]) -> float:
    if not comments:
        return 0.0
    lowered = [str(text).lower() for text in comments if str(text).strip()]
    if not lowered:
        return 0.0
    hits = 0
    for text in lowered:
        if any(term in text for term in terms):
            hits += 1
    return hits / len(lowered)


def _zscore(values: list[float], idx: int) -> float:
    subset = values[: idx + 1]
    if len(subset) < 3:
        return 0.0
    mean = sum(subset) / len(subset)
    variance = sum((value - mean) ** 2 for value in subset) / max(1, len(subset) - 1)
    std = variance**0.5
    if std <= 1e-9:
        return 0.0
    return (values[idx] - mean) / std


def _load_schema_signal_names(config: dict[str, Any], root: Path) -> list[str]:
    path = resolve_path(str(config.get("paths", {}).get("signals_feature_schema", "")), root)
    if not path.exists():
        return []
    rows = read_csv(path)
    out: list[str] = []
    for row in rows:
        name = str(row.get("signal_name", "")).strip()
        if name:
            out.append(name)
    return out


def _ensure_schema_columns(rows: list[dict[str, Any]], signal_names: list[str]) -> None:
    if not rows or not signal_names:
        return
    for row in rows:
        for name in signal_names:
            if name not in row:
                row[name] = 0.0


def _load_tastemaker_weights(config: dict[str, Any], root: Path) -> dict[str, float]:
    path = resolve_path(str(config.get("paths", {}).get("tastemakers_csv", "outputs/tastemakers.csv")), root)
    rows = read_csv(path)
    out: dict[str, float] = {}
    for row in rows:
        tm_id = str(row.get("tastemaker_id", "")).strip()
        if not tm_id:
            continue
        out[tm_id] = _as_float(row.get("quant_score", 0.0))
    return out


def _aggregate_daily(
    snapshots: list[dict[str, Any]],
    *,
    by_artist: bool,
    tastemaker_weights: dict[str, float],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in snapshots:
        day = _to_date(row.get("date"))
        if day is None:
            continue
        entity_id = str(row.get("artist_id", "")).strip() if by_artist else str(row.get("track_id", "")).strip()
        if not entity_id:
            continue
        key = (entity_id, day.isoformat())
        existing = grouped.get(key)
        if existing is None:
            existing = {
                "entity_id": entity_id,
                "entity_name": str(row.get("artist_name", "")).strip() if by_artist else str(row.get("track_name", "")).strip(),
                "artist_id": str(row.get("artist_id", "")).strip(),
                "artist_name": str(row.get("artist_name", "")).strip(),
                "track_id": str(row.get("track_id", "")).strip(),
                "track_name": str(row.get("track_name", "")).strip(),
                "date": day.isoformat(),
                "followers": 0.0,
                "metric_y": 0.0,
                "views": 0.0,
                "likes": 0.0,
                "comments": 0.0,
                "shares": 0.0,
                "playlist_adds": 0.0,
                "wiki_views": 0.0,
                "reddit_weight": 0.0,
                "rss_mentions": 0.0,
                "tastemaker_weighted_hits": 0.0,
                "musicbrainz_collab_proximity_raw": 0.0,
                "platform_metrics": defaultdict(float),
                "comments_text": [],
            }
            grouped[key] = existing

        platform = str(row.get("platform", "")).strip().lower()
        metric = _as_float(row.get("streams", 0.0)) if platform == "spotify" else _as_float(row.get("views", 0.0))
        existing["metric_y"] += metric
        existing["views"] += _as_float(row.get("views", 0.0))
        existing["likes"] += _as_float(row.get("likes", 0.0))
        existing["comments"] += _as_float(row.get("comments", 0.0))
        existing["shares"] += _as_float(row.get("shares", 0.0))
        existing["playlist_adds"] += _as_float(row.get("playlist_adds", 0.0))
        existing["followers"] = max(existing["followers"], _as_float(row.get("artist_followers", 0.0)))
        existing["platform_metrics"][platform] += metric

        if platform == "wikipedia":
            existing["wiki_views"] += _as_float(row.get("views", 0.0))
        if platform == "reddit":
            existing["reddit_weight"] += max(
                1.0,
                _as_float(row.get("likes", 0.0))
                + _as_float(row.get("comments", 0.0))
                + (0.5 * _as_float(row.get("shares", 0.0))),
            )
        if platform == "rss":
            existing["rss_mentions"] += 1.0
        if platform == "musicbrainz":
            collaborators = row.get("collaborators", [])
            collab_count = len(collaborators) if isinstance(collaborators, list) else 0
            existing["musicbrainz_collab_proximity_raw"] += min(8.0, float(collab_count))

        tm_id = str(row.get("tastemaker_id", "")).strip()
        if tm_id:
            existing["tastemaker_weighted_hits"] += max(0.05, tastemaker_weights.get(tm_id, 0.15))

        comments_text = row.get("comments_text", [])
        if isinstance(comments_text, list):
            existing["comments_text"].extend([str(item) for item in comments_text if str(item).strip()])

    out = list(grouped.values())
    out.sort(key=lambda item: (item["entity_id"], item["date"]))
    return out


def _compute_signal_rows(
    daily_rows: list[dict[str, Any]],
    *,
    comment_terms: list[str],
    lag_days: int,
) -> list[dict[str, Any]]:
    by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in daily_rows:
        by_entity[str(row["entity_id"])].append(row)

    out: list[dict[str, Any]] = []
    for entity_id, rows in by_entity.items():
        rows.sort(key=lambda item: item["date"])
        metric_series = [float(row["metric_y"]) for row in rows]
        spotify_metric = [float(row["platform_metrics"].get("spotify", 0.0)) for row in rows]
        non_spotify_metric = [float(row["metric_y"]) - float(row["platform_metrics"].get("spotify", 0.0)) for row in rows]
        wiki_series = [float(row["wiki_views"]) for row in rows]
        reddit_series = [float(row["reddit_weight"]) for row in rows]
        rss_series = [float(row["rss_mentions"]) for row in rows]
        playlist_series = [float(row["playlist_adds"]) for row in rows]

        growth_series: list[float] = []
        accel_series: list[float] = []
        wiki_growth_series: list[float] = []
        for idx, row in enumerate(rows):
            prev_metric = metric_series[idx - 1] if idx > 0 else 0.0
            g = math.log1p(metric_series[idx]) - math.log1p(prev_metric) if idx > 0 else 0.0
            growth_series.append(g)
            a = g - growth_series[idx - 1] if idx > 0 else 0.0
            accel_series.append(a)

            prev_wiki = wiki_series[idx - 1] if idx > 0 else 0.0
            wiki_g = math.log1p(wiki_series[idx]) - math.log1p(prev_wiki) if idx > 0 else 0.0
            wiki_growth_series.append(wiki_g)

            delta_metric = metric_series[idx] - prev_metric if idx > 0 else 0.0
            size_norm_velocity = delta_metric / max(1.0, math.sqrt(float(row["followers"]) + 1.0))
            engagement_rate = (row["likes"] + row["comments"] + row["shares"]) / (max(1.0, row["views"] or row["metric_y"]))
            playlist_add_velocity = row["playlist_adds"] - (playlist_series[idx - 1] if idx > 0 else 0.0)
            comment_specificity_score = _specificity(row["comments_text"], comment_terms)

            sp_growth = (
                math.log1p(spotify_metric[idx]) - math.log1p(spotify_metric[idx - 1])
                if idx > 0
                else 0.0
            )
            non_sp_growth = (
                math.log1p(non_spotify_metric[idx]) - math.log1p(non_spotify_metric[idx - 1])
                if idx > 0
                else 0.0
            )
            sp_g_series = [
                (math.log1p(spotify_metric[k]) - math.log1p(spotify_metric[k - 1])) if k > 0 else 0.0
                for k in range(len(rows))
            ]
            non_g_series = [
                (math.log1p(non_spotify_metric[k]) - math.log1p(non_spotify_metric[k - 1])) if k > 0 else 0.0
                for k in range(len(rows))
            ]
            z_sp = _zscore(sp_g_series, idx)
            lag_idx = idx - max(0, lag_days)
            z_non = _zscore(non_g_series, lag_idx) if lag_idx >= 0 else 0.0
            echo_score = clamp01(minmax_scale(z_sp + z_non, -3.0, 4.0))

            jump = (delta_metric / (prev_metric + 1.0)) if idx > 0 else 0.0
            spike_penalty = clamp01(max(0.0, jump - 2.0) * (1.0 - echo_score))

            wiki_accel = wiki_g - wiki_growth_series[idx - 1] if idx > 0 else 0.0
            reddit_mention_velocity = reddit_series[idx] - (reddit_series[idx - 1] if idx > 0 else 0.0)
            rss_mention_velocity = rss_series[idx] - (rss_series[idx - 1] if idx > 0 else 0.0)
            collab_proximity = clamp01(minmax_scale(float(row["musicbrainz_collab_proximity_raw"]), 0.0, 8.0))

            out.append(
                {
                    "entity_id": entity_id,
                    "entity_name": row["entity_name"],
                    "artist_id": row["artist_id"],
                    "artist_name": row["artist_name"],
                    "track_id": row["track_id"],
                    "track_name": row["track_name"],
                    "date": row["date"],
                    "momentum_log_growth": round(g, 6),
                    "acceleration": round(a, 6),
                    "sustained_accel_windows": 0,
                    "size_normalized_velocity": round(size_norm_velocity, 6),
                    "engagement_rate": round(engagement_rate, 6),
                    "comment_specificity_score": round(comment_specificity_score, 6),
                    "playlist_add_velocity": round(playlist_add_velocity, 6),
                    "tastemaker_weighted_hits": round(float(row["tastemaker_weighted_hits"]), 6),
                    "echo_score": round(echo_score, 6),
                    "spike_penalty": round(spike_penalty, 6),
                    "wiki_pageviews_accel": round(wiki_accel, 6),
                    "reddit_mention_velocity": round(reddit_mention_velocity, 6),
                    "rss_mention_velocity": round(rss_mention_velocity, 6),
                    "musicbrainz_collab_proximity": round(collab_proximity, 6),
                    "followers": int(row["followers"]),
                    "metric_y": round(float(row["metric_y"]), 6),
                }
            )
    return out


def _weekly_rollup(daily_signal_rows: list[dict[str, Any]], sustained_weeks: int = 3) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_entity_weeks: dict[str, list[str]] = defaultdict(list)
    for row in daily_signal_rows:
        day = _to_date(row.get("date"))
        if day is None:
            continue
        iso = day.isocalendar()
        week_id = f"{iso.year}-W{iso.week:02d}"
        key = (str(row["entity_id"]), week_id)
        grouped[key].append(row)
        by_entity_weeks[str(row["entity_id"])].append(week_id)

    weekly_rows: list[dict[str, Any]] = []
    for (entity_id, week_id), rows in grouped.items():
        rows.sort(key=lambda item: item["date"])
        agg = {
            "entity_id": entity_id,
            "entity_name": rows[-1]["entity_name"],
            "artist_id": rows[-1]["artist_id"],
            "artist_name": rows[-1]["artist_name"],
            "track_id": rows[-1]["track_id"],
            "track_name": rows[-1]["track_name"],
            "week": week_id,
            "momentum_log_growth": sum(_as_float(r["momentum_log_growth"]) for r in rows) / len(rows),
            "acceleration": sum(_as_float(r["acceleration"]) for r in rows) / len(rows),
            "size_normalized_velocity": sum(_as_float(r["size_normalized_velocity"]) for r in rows) / len(rows),
            "engagement_rate": sum(_as_float(r["engagement_rate"]) for r in rows) / len(rows),
            "comment_specificity_score": sum(_as_float(r["comment_specificity_score"]) for r in rows) / len(rows),
            "playlist_add_velocity": sum(_as_float(r["playlist_add_velocity"]) for r in rows),
            "tastemaker_weighted_hits": sum(_as_float(r["tastemaker_weighted_hits"]) for r in rows),
            "echo_score": sum(_as_float(r["echo_score"]) for r in rows) / len(rows),
            "spike_penalty": sum(_as_float(r["spike_penalty"]) for r in rows) / len(rows),
            "wiki_pageviews_accel": sum(_as_float(r["wiki_pageviews_accel"]) for r in rows) / len(rows),
            "reddit_mention_velocity": sum(_as_float(r["reddit_mention_velocity"]) for r in rows),
            "rss_mention_velocity": sum(_as_float(r["rss_mention_velocity"]) for r in rows),
            "musicbrainz_collab_proximity": sum(_as_float(r["musicbrainz_collab_proximity"]) for r in rows) / len(rows),
            "sustained_accel_windows": 0,
        }
        weekly_rows.append(agg)

    weekly_rows.sort(key=lambda item: (item["entity_id"], item["week"]))
    by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in weekly_rows:
        by_entity[str(row["entity_id"])].append(row)
    for rows in by_entity.values():
        for idx, row in enumerate(rows):
            start = max(0, idx - max(1, sustained_weeks) + 1)
            window = rows[start : idx + 1]
            count = sum(1 for item in window if _as_float(item["acceleration"]) > 0)
            row["sustained_accel_windows"] = int(count)

    return weekly_rows


def build_signal_feature_tables(
    snapshots: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    signal_names = _load_schema_signal_names(config, project_root)
    terms = list(config.get("features", {}).get("depth", {}).get("comment_specificity_terms", []))
    lag_days = int(config.get("candidate_generation", {}).get("cross_platform_echo", {}).get("lag_days", {}).get("tiktok_to_spotify", [3, 10])[0])
    tastemaker_weights = _load_tastemaker_weights(config, project_root)

    track_daily_base = _aggregate_daily(snapshots, by_artist=False, tastemaker_weights=tastemaker_weights)
    artist_daily_base = _aggregate_daily(snapshots, by_artist=True, tastemaker_weights=tastemaker_weights)
    track_daily = _compute_signal_rows(track_daily_base, comment_terms=terms, lag_days=max(1, lag_days))
    artist_daily = _compute_signal_rows(artist_daily_base, comment_terms=terms, lag_days=max(1, lag_days))
    track_weekly = _weekly_rollup(track_daily, sustained_weeks=3)
    artist_weekly = _weekly_rollup(artist_daily, sustained_weeks=3)
    _ensure_schema_columns(track_daily, signal_names)
    _ensure_schema_columns(artist_daily, signal_names)
    _ensure_schema_columns(track_weekly, signal_names)
    _ensure_schema_columns(artist_weekly, signal_names)

    paths = config.get("paths", {})
    track_daily_path = resolve_path(str(paths.get("features_daily_track", "outputs/features_daily_track.csv")), project_root)
    track_weekly_path = resolve_path(str(paths.get("features_weekly_track", "outputs/features_weekly_track.csv")), project_root)
    artist_daily_path = resolve_path(str(paths.get("features_daily_artist", "outputs/features_daily_artist.csv")), project_root)
    artist_weekly_path = resolve_path(str(paths.get("features_weekly_artist", "outputs/features_weekly_artist.csv")), project_root)
    write_csv(track_daily_path, track_daily)
    write_csv(track_weekly_path, track_weekly)
    write_csv(artist_daily_path, artist_daily)
    write_csv(artist_weekly_path, artist_weekly)

    return {
        "signal_schema_columns": signal_names,
        "track_daily_rows": len(track_daily),
        "track_weekly_rows": len(track_weekly),
        "artist_daily_rows": len(artist_daily),
        "artist_weekly_rows": len(artist_weekly),
        "paths": {
            "features_daily_track": str(track_daily_path),
            "features_weekly_track": str(track_weekly_path),
            "features_daily_artist": str(artist_daily_path),
            "features_weekly_artist": str(artist_weekly_path),
        },
    }
