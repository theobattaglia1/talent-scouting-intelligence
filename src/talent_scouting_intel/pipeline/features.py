from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from talent_scouting_intel.pipeline.common import filter_as_of, group_by_track, group_by_track_platform
from talent_scouting_intel.pipeline.signal_features import build_signal_feature_tables
from talent_scouting_intel.utils.io import load_config, read_csv, read_jsonl, resolve_path, write_csv
from talent_scouting_intel.utils.math_utils import (
    acceleration_series,
    clamp01,
    consecutive_positive_tail,
    growth_series,
    mean,
    minmax_scale,
)


def _series(rows: list[dict[str, Any]], metric: str) -> list[float]:
    return [float(row.get(metric, 0.0) or 0.0) for row in sorted(rows, key=lambda item: item["date"])]


def _spotify_rows(track_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in track_rows if row["platform"] == "spotify"]


def _shortform_proxy_score(track_rows: list[dict[str, Any]]) -> tuple[float, float]:
    proxy_rows = [row for row in track_rows if row["platform"] == "tiktok_proxy"]
    if len(proxy_rows) < 2:
        return 0.0, 0.0

    views = _series(proxy_rows, "views")
    reuse = _series(proxy_rows, "creator_reuse")
    view_growth = mean(growth_series(views)[-7:]) if len(views) >= 8 else 0.0
    reuse_growth = mean(growth_series(reuse)[-7:]) if len(reuse) >= 8 else 0.0

    shortform_score = clamp01(
        0.55 * minmax_scale(view_growth, -0.03, 0.28)
        + 0.45 * minmax_scale(reuse_growth, -0.03, 0.35)
    )
    return shortform_score, reuse_growth


def _knowledge_graph_score(track_rows: list[dict[str, Any]]) -> tuple[float, float]:
    wiki_rows = [row for row in track_rows if row["platform"] == "wikipedia"]
    mb_rows = [row for row in track_rows if row["platform"] == "musicbrainz"]

    wiki_views = _series(wiki_rows, "views")
    wiki_growth = mean(growth_series(wiki_views)[-7:]) if len(wiki_views) >= 8 else 0.0
    mb_views = _series(mb_rows, "views")
    mb_strength = mean(mb_views[-7:]) if mb_views else 0.0

    score = clamp01(
        0.55 * minmax_scale(wiki_growth, -0.05, 0.25)
        + 0.45 * minmax_scale(mb_strength, 20.0, 500.0)
    )
    return score, wiki_growth


def compute_acceleration_features(streams: list[float]) -> dict[str, float]:
    if len(streams) < 21:
        return {
            "size_norm_accel": 0.0,
            "acceleration_score": 0.0,
            "positive_accel_rate": 0.0,
            "consecutive_positive_accel": 0.0,
            "avg_growth": 0.0,
        }

    growth = growth_series(streams)

    # Weekly windows reduce day-level noise and enforce multi-window persistence.
    weekly_growth: list[float] = []
    for idx in range(7, len(streams), 7):
        weekly_growth.append(math.log1p(streams[idx]) - math.log1p(streams[idx - 7]))

    if len(weekly_growth) < 2:
        accel = acceleration_series(streams)
    else:
        accel = [weekly_growth[idx] - weekly_growth[idx - 1] for idx in range(1, len(weekly_growth))]

    recent_accel = accel[-3:] if len(accel) >= 3 else accel
    recent_stream_level = mean(streams[-14:]) if streams else 0.0

    # size-normalized second derivative style momentum
    size_norm_accel = 100.0 * (mean(recent_accel) / max(1.0, math.log1p(recent_stream_level)))
    positive_accel_rate = sum(1 for value in accel if value > 0) / max(1, len(accel))
    consecutive = consecutive_positive_tail(accel)

    acceleration_score = clamp01(
        0.55 * minmax_scale(size_norm_accel, -0.02, 0.12)
        + 0.25 * minmax_scale(positive_accel_rate, 0.2, 0.9)
        + 0.2 * minmax_scale(consecutive, 0, 4)
    )

    return {
        "size_norm_accel": size_norm_accel,
        "acceleration_score": acceleration_score,
        "positive_accel_rate": positive_accel_rate,
        "consecutive_positive_accel": float(consecutive),
        "avg_growth": mean(growth[-7:]) if growth else 0.0,
    }


def compute_consistency_score(streams: list[float]) -> tuple[float, float]:
    growth = growth_series(streams)
    if len(growth) < 3:
        return 0.0, 0.0
    positive_rate = sum(1 for value in growth if value > 0) / len(growth)
    sign_changes = 0
    for idx in range(1, len(growth)):
        if (growth[idx] >= 0) != (growth[idx - 1] >= 0):
            sign_changes += 1
    stability = 1.0 - (sign_changes / max(1, len(growth) - 1))
    consistency = clamp01(0.6 * positive_rate + 0.4 * stability)
    return consistency, positive_rate


def _comment_specificity(comments_text: list[str], terms: list[str]) -> float:
    if not comments_text:
        return 0.0
    lowered = [entry.lower() for entry in comments_text]
    specific = 0
    for text in lowered:
        if any(term in text for term in terms):
            specific += 1
    return specific / len(lowered)


def detect_spike_only(
    streams: list[float],
    depth_score: float,
    echo_score: float,
    anti_cfg: dict[str, Any],
) -> bool:
    if len(streams) < 6:
        return False

    growth = [
        (streams[idx] - streams[idx - 1]) / (streams[idx - 1] + 1.0)
        for idx in range(1, len(streams))
    ]
    jump_threshold = float(anti_cfg["spike_jump_threshold"])
    reversion_threshold = float(anti_cfg["spike_reversion_threshold"])

    max_jump = max(growth)
    jump_idx = growth.index(max_jump)
    next_growth = growth[jump_idx + 1] if jump_idx + 1 < len(growth) else 0.0

    return (
        max_jump >= jump_threshold
        and next_growth <= reversion_threshold
        and depth_score < 0.45
        and echo_score < 0.5
    )


def _detect_suspicious_ratios(
    views: float,
    likes: float,
    comments: float,
    echo_score: float,
    anti_cfg: dict[str, Any],
) -> bool:
    if comments <= 0:
        return views > 50000
    view_comment_ratio = views / comments
    like_comment_ratio = likes / comments
    return (
        view_comment_ratio >= float(anti_cfg["suspicious_view_comment_ratio"])
        and like_comment_ratio >= float(anti_cfg["suspicious_like_comment_ratio"])
        and echo_score <= float(anti_cfg["suspicious_no_echo_max"])
    )


def _platform_dependency_score(track_rows: list[dict[str, Any]]) -> tuple[float, float]:
    grouped = group_by_track_platform(track_rows)
    growth_by_platform: dict[str, float] = {}
    platforms = sorted({platform for key_track, platform in grouped.keys() if key_track == track_rows[0]["track_id"]})
    for platform in platforms:
        rows = grouped.get((track_rows[0]["track_id"], platform), [])
        if len(rows) < 15:
            growth_by_platform[platform] = 0.0
            continue
        metric_name = "streams" if platform == "spotify" else "views"
        values = _series(rows, metric_name)
        growth_by_platform[platform] = max(0.0, values[-1] - values[-8])

    total_growth = sum(growth_by_platform.values())
    if total_growth <= 0:
        return 0.0, 0.0
    spotify_share = growth_by_platform.get("spotify", 0.0) / total_growth

    spotify_rows = grouped.get((track_rows[0]["track_id"], "spotify"), [])
    playlist_adds = sum(float(row.get("playlist_adds", 0.0)) for row in spotify_rows[-14:])
    streams = sum(float(row.get("streams", 0.0)) for row in spotify_rows[-14:])
    playlist_ratio = playlist_adds / (streams + 1.0)
    return spotify_share, playlist_ratio


def _depth_score(
    track_rows: list[dict[str, Any]],
    specific_terms: list[str],
    artist_followers: int,
) -> tuple[float, dict[str, float]]:
    recent_rows = track_rows[-56:] if len(track_rows) >= 56 else track_rows
    views = sum(float(row.get("views", 0.0)) for row in recent_rows)
    likes = sum(float(row.get("likes", 0.0)) for row in recent_rows)
    comments = sum(float(row.get("comments", 0.0)) for row in recent_rows)
    shares = sum(float(row.get("shares", 0.0)) for row in recent_rows)
    listeners = sum(float(row.get("listeners", 0.0)) for row in recent_rows)
    playlist_adds = sum(float(row.get("playlist_adds", 0.0)) for row in recent_rows)
    comments_text = [text for row in recent_rows for text in row.get("comments_text", [])]

    comments_per_view = comments / (views + 1.0)
    shares_per_view = shares / (views + 1.0)
    engagement_rate = (likes + comments + shares) / (views + 1.0)
    follower_conversion = listeners / (float(artist_followers) + 1.0)
    save_proxy = playlist_adds / (listeners + 1.0)
    specificity = _comment_specificity(comments_text, specific_terms)

    depth_score = clamp01(
        0.25 * minmax_scale(engagement_rate, 0.02, 0.2)
        + 0.2 * minmax_scale(comments_per_view, 0.001, 0.02)
        + 0.15 * minmax_scale(shares_per_view, 0.001, 0.03)
        + 0.15 * minmax_scale(save_proxy, 0.002, 0.08)
        + 0.15 * minmax_scale(follower_conversion, 0.05, 4.0)
        + 0.1 * minmax_scale(specificity, 0.05, 0.65)
    )

    return depth_score, {
        "views": views,
        "likes": likes,
        "comments": comments,
        "comments_per_view": comments_per_view,
        "shares_per_view": shares_per_view,
        "engagement_rate": engagement_rate,
        "follower_conversion": follower_conversion,
        "save_proxy": save_proxy,
        "comment_specificity": specificity,
    }


def build_features(
    snapshots: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    as_of: str | None = None,
) -> list[dict[str, Any]]:
    scoped = filter_as_of(snapshots, as_of)
    track_rows = group_by_track(scoped)
    candidate_by_track = {row["track_id"]: row for row in candidates}
    anti_cfg = config["features"]["anti_gaming"]
    terms = list(config["features"]["depth"]["comment_specificity_terms"])
    min_spotify_points = int(config.get("features", {}).get("min_spotify_points", 14))

    features: list[dict[str, Any]] = []
    for track_id, rows in track_rows.items():
        if track_id not in candidate_by_track:
            continue

        spotify = _spotify_rows(rows)
        if len(spotify) < max(1, min_spotify_points):
            continue

        candidate = candidate_by_track[track_id]
        streams = _series(spotify, "streams")
        spotify_points = len(spotify)
        history_days = len({row["date"] for row in spotify})
        accel_feats = compute_acceleration_features(streams)
        consistency_score, positive_growth_rate = compute_consistency_score(streams)

        artist_followers = int(spotify[-1].get("artist_followers", 0))
        depth_score, depth_meta = _depth_score(rows, terms, artist_followers)

        momentum_level = minmax_scale(math.log1p(mean(streams[-7:])), 4.0, 13.0)
        growth = growth_series(streams)
        momentum_growth = minmax_scale(mean(growth[-7:]) if growth else 0.0, -0.02, 0.35)
        momentum_score = clamp01(0.55 * momentum_level + 0.45 * momentum_growth)

        collaborator_count = len({collab for row in rows for collab in row.get("collaborators", [])})
        knowledge_score, wiki_growth = _knowledge_graph_score(rows)
        network_score = clamp01(
            0.6 * float(candidate["tastemaker_score"])
            + 0.25 * minmax_scale(collaborator_count, 0, 6)
            + 0.15 * knowledge_score
        )

        cross_platform_score = float(candidate["echo_score"])
        shortform_proxy_score, shortform_reuse_growth = _shortform_proxy_score(rows)
        cross_platform_score = clamp01(0.82 * cross_platform_score + 0.18 * shortform_proxy_score)
        geo_score = float(candidate["geo_score"])

        spike_only = detect_spike_only(streams, depth_score, cross_platform_score, anti_cfg)
        suspicious = _detect_suspicious_ratios(
            depth_meta["views"],
            depth_meta["likes"],
            depth_meta["comments"],
            cross_platform_score,
            anti_cfg,
        )

        spotify_share, playlist_ratio = _platform_dependency_score(rows)
        playlist_dependent = (
            spotify_share >= float(anti_cfg["playlist_dependency_source_share"])
            and playlist_ratio >= 0.008
            and cross_platform_score < 0.5
        )

        creator_reuse = _series(
            [row for row in rows if row["platform"] in {"tiktok", "tiktok_proxy"}],
            "creator_reuse",
        )
        creator_reuse_growth = mean(growth_series(creator_reuse)[-7:]) if len(creator_reuse) >= 8 else 0.0
        content_velocity = (
            len({(row["date"], row["platform"]) for row in rows[-28:]}) / 28.0
        )

        latest = rows[-1]
        feature_row = {
            "track_id": track_id,
            "track_name": latest["track_name"],
            "artist_id": latest["artist_id"],
            "artist_name": latest["artist_name"],
            "genre_hint": latest.get("genre_hint", "unknown"),
            "metadata_text": latest.get("metadata_text", ""),
            "artist_followers": artist_followers,
            "spotify_points": spotify_points,
            "history_days": history_days,
            "latest_date": latest["date"],
            "momentum_score": round(momentum_score, 6),
            "acceleration_score": round(float(accel_feats["acceleration_score"]), 6),
            "size_norm_accel": round(float(accel_feats["size_norm_accel"]), 6),
            "positive_accel_rate": round(float(accel_feats["positive_accel_rate"]), 6),
            "consecutive_positive_accel": int(accel_feats["consecutive_positive_accel"]),
            "avg_growth": round(float(accel_feats["avg_growth"]), 6),
            "depth_score": round(depth_score, 6),
            "cross_platform_score": round(cross_platform_score, 6),
            "network_score": round(network_score, 6),
            "consistency_score": round(consistency_score, 6),
            "positive_growth_rate": round(positive_growth_rate, 6),
            "geo_score": round(geo_score, 6),
            "shortform_proxy_score": round(shortform_proxy_score, 6),
            "knowledge_graph_score": round(knowledge_score, 6),
            "wikipedia_growth": round(wiki_growth, 6),
            "content_velocity": round(content_velocity, 6),
            "creator_reuse_growth": round(max(creator_reuse_growth, shortform_reuse_growth), 6),
            "engagement_rate": round(depth_meta["engagement_rate"], 6),
            "comments_per_view": round(depth_meta["comments_per_view"], 6),
            "shares_per_view": round(depth_meta["shares_per_view"], 6),
            "save_proxy": round(depth_meta["save_proxy"], 6),
            "follower_conversion": round(depth_meta["follower_conversion"], 6),
            "comment_specificity": round(depth_meta["comment_specificity"], 6),
            "candidate_priority": round(float(candidate["candidate_priority"]), 6),
            "tastemaker_score": round(float(candidate["tastemaker_score"]), 6),
            "low_base_accel": round(float(candidate["low_base_accel"]), 6),
            "anomaly_score": round(float(candidate["anomaly_score"]), 6),
            "echo_score": round(float(candidate["echo_score"]), 6),
            "established_artist": bool(candidate.get("established_artist", False)),
            "established_by_followers": bool(candidate.get("established_by_followers", False)),
            "established_by_footprint": bool(candidate.get("established_by_footprint", False)),
            "artist_track_footprint": int(float(candidate.get("artist_track_footprint", 0) or 0)),
            "established_penalty": round(float(candidate.get("established_penalty", 0.0) or 0.0), 6),
            "manual_seeded": bool(candidate.get("manual_seeded", False)),
            "spike_only": spike_only,
            "suspicious": suspicious,
            "playlist_dependent": playlist_dependent,
            "spotify_growth_share": round(spotify_share, 6),
            "playlist_ratio": round(playlist_ratio, 6),
            "views_recent": int(depth_meta["views"]),
            "likes_recent": int(depth_meta["likes"]),
            "comments_recent": int(depth_meta["comments"]),
            "collaborator_count": collaborator_count,
        }
        features.append(feature_row)

    features.sort(key=lambda row: float(row["candidate_priority"]), reverse=True)
    return features


def run_features(
    config_path: str,
    *,
    project_root: Path | None = None,
    as_of: str | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    root = project_root or Path.cwd()

    snapshots = read_jsonl(resolve_path(config["paths"]["snapshots"], root))
    candidates = read_jsonl(resolve_path(config["paths"]["candidates"], root))
    features = build_features(snapshots, candidates, config, as_of=as_of)
    signal_tables = build_signal_feature_tables(snapshots, config, project_root=root)

    daily_track_rows = read_csv(resolve_path(config["paths"]["features_daily_track"], root))
    weekly_track_rows = read_csv(resolve_path(config["paths"]["features_weekly_track"], root))
    latest_daily_by_track: dict[str, dict[str, Any]] = {}
    for row in daily_track_rows:
        track_id = str(row.get("track_id", "")).strip()
        day = str(row.get("date", "")).strip()
        if not track_id:
            continue
        prev = latest_daily_by_track.get(track_id)
        if prev is None or day > str(prev.get("date", "")):
            latest_daily_by_track[track_id] = row
    latest_weekly_by_track: dict[str, dict[str, Any]] = {}
    for row in weekly_track_rows:
        track_id = str(row.get("track_id", "")).strip()
        week = str(row.get("week", "")).strip()
        if not track_id:
            continue
        prev = latest_weekly_by_track.get(track_id)
        if prev is None or week > str(prev.get("week", "")):
            latest_weekly_by_track[track_id] = row

    for row in features:
        track_id = str(row.get("track_id", "")).strip()
        daily = latest_daily_by_track.get(track_id, {})
        weekly = latest_weekly_by_track.get(track_id, {})
        row["momentum_log_growth"] = float(daily.get("momentum_log_growth", 0.0) or 0.0)
        row["acceleration"] = float(daily.get("acceleration", 0.0) or 0.0)
        row["size_normalized_velocity"] = float(daily.get("size_normalized_velocity", 0.0) or 0.0)
        row["comment_specificity_score"] = float(daily.get("comment_specificity_score", row.get("comment_specificity", 0.0)) or 0.0)
        row["playlist_add_velocity"] = float(daily.get("playlist_add_velocity", 0.0) or 0.0)
        row["tastemaker_weighted_hits"] = float(daily.get("tastemaker_weighted_hits", row.get("tastemaker_score", 0.0)) or 0.0)
        row["echo_score_signal"] = float(daily.get("echo_score", row.get("echo_score", 0.0)) or 0.0)
        row["spike_penalty"] = float(daily.get("spike_penalty", 0.0) or 0.0)
        row["wiki_pageviews_accel"] = float(daily.get("wiki_pageviews_accel", 0.0) or 0.0)
        row["reddit_mention_velocity"] = float(daily.get("reddit_mention_velocity", 0.0) or 0.0)
        row["rss_mention_velocity"] = float(daily.get("rss_mention_velocity", 0.0) or 0.0)
        row["musicbrainz_collab_proximity"] = float(
            weekly.get("musicbrainz_collab_proximity", daily.get("musicbrainz_collab_proximity", 0.0)) or 0.0
        )
        row["sustained_accel_windows"] = int(float(weekly.get("sustained_accel_windows", 0) or 0))

    output_path = resolve_path(config["paths"]["features"], root)
    write_csv(output_path, features)

    return {
        "features": len(features),
        "output_path": str(output_path),
        "signal_tables": signal_tables,
    }
