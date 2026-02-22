from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from talent_scouting_intel.pipeline.common import filter_as_of, group_by_track, group_by_track_platform
from talent_scouting_intel.utils.io import load_config, read_csv, read_jsonl, resolve_path, write_jsonl
from talent_scouting_intel.utils.math_utils import acceleration_series, clamp01, corr, growth_series, mean, minmax_scale, zscore


def _latest_date(rows: list[dict[str, Any]]) -> dt.date:
    return max(dt.date.fromisoformat(row["date"]) for row in rows)


def _series(rows: list[dict[str, Any]], metric: str) -> list[float]:
    return [float(row.get(metric, 0.0) or 0.0) for row in sorted(rows, key=lambda item: item["date"])]


def _spotify_rows(track_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in track_rows if row["platform"] == "spotify"]


def _artist_key(row: dict[str, Any]) -> str:
    artist_id = str(row.get("artist_id", "")).strip()
    if artist_id:
        return artist_id
    return str(row.get("artist_name", "")).strip().lower()


def _load_prior_tastemaker_scores(config: dict[str, Any], root: Path) -> dict[str, float]:
    path_value = config.get("paths", {}).get("tastemakers_csv")
    if not path_value:
        return {}
    csv_path = resolve_path(str(path_value), root)
    rows = read_csv(csv_path)

    out: dict[str, float] = {}
    for row in rows:
        tm_id = str(row.get("tastemaker_id", "")).strip()
        if not tm_id:
            continue
        try:
            out[tm_id] = float(row.get("quant_score", 0.0))
        except Exception:
            out[tm_id] = 0.0
    return out


def _broke_proxy(track_rows: list[dict[str, Any]], config: dict[str, Any]) -> bool:
    rules = config["thresholds"]["broke_proxy"]
    spotify = _spotify_rows(track_rows)
    if len(spotify) < 21:
        return False
    streams = _series(spotify, "streams")
    if max(streams, default=0.0) < float(rules["min_streams"]):
        return False

    weekly_growths: list[float] = []
    for idx in range(7, len(streams)):
        base = streams[idx - 7]
        weekly_growths.append((streams[idx] - base) / (base + 1.0))

    target = float(rules["min_weekly_growth"])
    sustain = int(rules["sustain_weeks"])
    run = 0
    for growth in weekly_growths:
        if growth >= target:
            run += 1
            if run >= sustain:
                return True
        else:
            run = 0
    return False


def _curator_scores(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    prior_scores: dict[str, float] | None = None,
) -> dict[str, float]:
    if not rows:
        return {}

    grouped = group_by_track(rows)
    track_outcomes = {track_id: _broke_proxy(track_rows, config) for track_id, track_rows in grouped.items()}
    latest_day = _latest_date(rows)

    prior_success = float(config["candidate_generation"]["tastemakers"]["bayes_prior_success"])
    prior_trials = float(config["candidate_generation"]["tastemakers"]["bayes_prior_trials"])
    half_life = float(config["candidate_generation"]["tastemakers"]["recency_half_life_days"])

    successes: dict[str, float] = defaultdict(float)
    trials: dict[str, float] = defaultdict(float)

    for row in rows:
        tastemaker_id = row.get("tastemaker_id")
        if not tastemaker_id:
            continue
        days_old = (latest_day - dt.date.fromisoformat(row["date"])).days
        weight = 0.5 ** (days_old / max(half_life, 1.0))
        trials[tastemaker_id] += weight
        if track_outcomes.get(row["track_id"], False):
            successes[tastemaker_id] += weight

    scores: dict[str, float] = {}
    tastemakers = set(trials.keys()) | set(successes.keys())
    blend = float(config.get("candidate_generation", {}).get("tastemakers", {}).get("prior_blend", 0.35))
    prior = prior_scores or {}
    tastemakers |= set(prior.keys())
    for tastemaker_id in tastemakers:
        live_score = (prior_success + successes[tastemaker_id]) / (
            prior_trials + trials[tastemaker_id]
        )
        if tastemaker_id in prior:
            scores[tastemaker_id] = (1.0 - blend) * live_score + blend * float(prior[tastemaker_id])
        else:
            scores[tastemaker_id] = live_score
    return scores


def _track_pair_corr(
    track_rows: list[dict[str, Any]],
    src_platform: str,
    dst_platform: str,
    min_lag: int,
    max_lag: int,
) -> float:
    pair_group = group_by_track_platform(track_rows)
    src_rows = pair_group.get((track_rows[0]["track_id"], src_platform), [])
    dst_rows = pair_group.get((track_rows[0]["track_id"], dst_platform), [])
    if len(src_rows) < 10 or len(dst_rows) < 10:
        return 0.0

    src_growth = growth_series(_series(src_rows, "views" if src_platform != "spotify" else "streams"))
    dst_growth = growth_series(_series(dst_rows, "views" if dst_platform != "spotify" else "streams"))
    if len(src_growth) < 5 or len(dst_growth) < 5:
        return 0.0

    best = 0.0
    for lag in range(min_lag, max_lag + 1):
        if lag >= len(src_growth) or lag >= len(dst_growth):
            continue
        a = src_growth[:-lag] if lag > 0 else src_growth
        b = dst_growth[lag:] if lag > 0 else dst_growth
        length = min(len(a), len(b))
        if length < 5:
            continue
        candidate = corr(a[-length:], b[-length:])
        best = max(best, candidate)
    return max(0.0, best)


def _geo_score(track_rows: list[dict[str, Any]]) -> tuple[float, int]:
    spotify_rows = _spotify_rows(track_rows)
    if len(spotify_rows) < 14:
        return 0.0, 0

    first = spotify_rows[:7]
    last = spotify_rows[-7:]

    def region_count(rows: list[dict[str, Any]]) -> int:
        summed: dict[str, int] = defaultdict(int)
        for row in rows:
            for region, value in (row.get("region_metrics") or {}).items():
                summed[region] += int(value)
        total = sum(summed.values())
        if total <= 0:
            return 0
        return sum(1 for value in summed.values() if value / total >= 0.05)

    def hhi(rows: list[dict[str, Any]]) -> float:
        summed: dict[str, int] = defaultdict(int)
        for row in rows:
            for region, value in (row.get("region_metrics") or {}).items():
                summed[region] += int(value)
        total = sum(summed.values())
        if total <= 0:
            return 1.0
        shares = [value / total for value in summed.values()]
        return sum(share * share for share in shares)

    first_regions = region_count(first)
    last_regions = region_count(last)
    diffusion_gain = last_regions - first_regions
    concentration = hhi(first)
    score = clamp01(0.5 * minmax_scale(1.0 - concentration, 0.0, 0.9) + 0.5 * minmax_scale(diffusion_gain, -1, 4))
    return score, diffusion_gain


def build_candidates(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    as_of: str | None = None,
    prior_tastemaker_scores: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    scoped_rows = filter_as_of(rows, as_of)
    if not scoped_rows:
        return []
    grouped = group_by_track(scoped_rows)
    curator_scores = _curator_scores(scoped_rows, config, prior_scores=prior_tastemaker_scores)
    min_spotify_points = int(config.get("candidate_generation", {}).get("min_spotify_points", 14))
    low_follower_cfg = config.get("candidate_generation", {}).get("low_follower_anomaly", {})
    max_followers_for_focus = int(low_follower_cfg.get("max_artist_followers_for_focus", 250000))
    established_min_points_override = int(config.get("candidate_generation", {}).get("established_min_points_override", 4))
    established_track_footprint_min = int(config.get("candidate_generation", {}).get("established_artist_track_footprint_min", 8))
    established_require_strong = bool(config.get("candidate_generation", {}).get("established_artist_require_strong_signals", True))
    established_min_low_base_accel = float(config.get("candidate_generation", {}).get("established_artist_min_low_base_accel", 0.68))
    established_min_echo = float(config.get("candidate_generation", {}).get("established_artist_min_echo", 0.62))
    established_min_anomaly = float(config.get("candidate_generation", {}).get("established_artist_min_anomaly", 0.58))

    artist_track_map: dict[str, set[str]] = defaultdict(set)
    for row in scoped_rows:
        key = _artist_key(row)
        track_id = str(row.get("track_id", "")).strip()
        if not key or not track_id:
            continue
        artist_track_map[key].add(track_id)

    raw_velocity: dict[str, float] = {}
    raw_engagement: dict[str, float] = {}
    raw_conversion: dict[str, float] = {}

    for track_id, track_rows in grouped.items():
        spotify = _spotify_rows(track_rows)
        if len(spotify) < max(1, min_spotify_points):
            raw_velocity[track_id] = 0.0
            raw_engagement[track_id] = 0.0
            raw_conversion[track_id] = 0.0
            continue
        streams = _series(spotify, "streams")
        if len(streams) >= 8:
            velocity_delta = streams[-1] - streams[-8]
        elif len(streams) >= 2:
            velocity_delta = streams[-1] - streams[0]
        else:
            velocity_delta = 0.0
        velocity = velocity_delta / math.sqrt(float(spotify[-1]["artist_followers"]) + 1.0)
        last_rows = track_rows[-56:]
        views = sum(float(row.get("views", 0)) for row in last_rows)
        interactions = sum(float(row.get("likes", 0) + row.get("comments", 0) + row.get("shares", 0)) for row in last_rows)
        listeners = sum(float(row.get("listeners", 0)) for row in last_rows)
        followers = float(spotify[-1]["artist_followers"]) + 1.0

        raw_velocity[track_id] = velocity
        raw_engagement[track_id] = interactions / (views + 1.0)
        raw_conversion[track_id] = listeners / followers

    velocity_population = list(raw_velocity.values())
    engagement_population = list(raw_engagement.values())
    conversion_population = list(raw_conversion.values())

    lag_cfg = config["candidate_generation"]["cross_platform_echo"]["lag_days"]
    min_echo = float(config["candidate_generation"]["cross_platform_echo"]["min_echo_score"])

    candidates: list[dict[str, Any]] = []
    for track_id, track_rows in grouped.items():
        spotify = _spotify_rows(track_rows)
        if len(spotify) < max(1, min_spotify_points):
            continue
        manual_seeded = any(bool(row.get("manual_seeded")) for row in track_rows)
        artist_followers = float(spotify[-1].get("artist_followers", 0) or 0)
        artist_track_footprint = len(artist_track_map.get(_artist_key(spotify[-1]), set()))
        established_by_followers = artist_followers > max_followers_for_focus
        established_by_footprint = artist_track_footprint >= max(2, established_track_footprint_min)
        established_artist = established_by_followers or established_by_footprint

        # For mature artists, require more than a single-day snapshot before treating them
        # as "early breakout" candidates.
        if (
            established_by_followers
            and len(spotify) < max(1, established_min_points_override)
            and not manual_seeded
        ):
            continue

        streams = _series(spotify, "streams")
        acc = acceleration_series(streams)
        recent_acc = mean(acc[-7:]) if acc else 0.0
        early_base = mean(streams[:7]) if streams[:7] else 0.0
        size_factor = 1.0 / max(1.0, math.log10(early_base + 10.0))
        low_base_accel = clamp01(0.7 * minmax_scale(recent_acc, -0.06, 0.16) + 0.3 * minmax_scale(size_factor, 0.2, 1.0))

        tm_weights = [
            curator_scores.get(row.get("tastemaker_id") or "", 0.0)
            for row in track_rows
            if row.get("tastemaker_id")
        ]
        tastemaker_score = clamp01(mean(tm_weights) * 1.35 if tm_weights else 0.0)

        velocity_z = zscore(raw_velocity[track_id], velocity_population)
        engagement_z = zscore(raw_engagement[track_id], engagement_population)
        conversion_z = zscore(raw_conversion[track_id], conversion_population)
        anomaly_score = clamp01(
            0.45 * minmax_scale(velocity_z, -1.5, 2.2)
            + 0.3 * minmax_scale(engagement_z, -1.2, 2.0)
            + 0.25 * minmax_scale(conversion_z, -1.2, 2.0)
        )

        pair_corrs: list[float] = []
        for pair_name, lag_window in lag_cfg.items():
            if not str(pair_name).endswith("_to_spotify"):
                continue
            src_platform = str(pair_name).replace("_to_spotify", "")
            if not isinstance(lag_window, list) or len(lag_window) != 2:
                continue
            pair_corrs.append(
                _track_pair_corr(
                    track_rows,
                    src_platform,
                    "spotify",
                    int(lag_window[0]),
                    int(lag_window[1]),
                )
            )

        echo_corr = mean(pair_corrs)
        platform_accels: list[float] = []
        by_platform = group_by_track_platform(track_rows)
        platforms = sorted({platform for key_track, platform in by_platform.keys() if key_track == track_id})
        for platform in platforms:
            platform_rows = by_platform.get((track_id, platform), [])
            metric = _series(platform_rows, "streams" if platform == "spotify" else "views")
            platform_accels.append(mean(acceleration_series(metric)[-5:]))
        accel_strength = minmax_scale(mean(platform_accels), -0.05, 0.12)
        echo_score = clamp01(0.55 * accel_strength + 0.45 * echo_corr)

        geo_score, diffusion_gain = _geo_score(track_rows)

        if established_artist and established_require_strong and not manual_seeded:
            if not (
                low_base_accel >= established_min_low_base_accel
                and echo_score >= established_min_echo
                and anomaly_score >= established_min_anomaly
            ):
                continue

        seed_boost = 0.12 if manual_seeded else 0.0
        established_penalty = 0.0
        if established_artist and not manual_seeded:
            follower_penalty = 0.0
            if established_by_followers:
                follower_penalty = clamp01(
                    minmax_scale(
                        math.log10(artist_followers + 1.0),
                        math.log10(max_followers_for_focus + 1.0),
                        math.log10((max_followers_for_focus * 50.0) + 1.0),
                    )
                )
            footprint_penalty = clamp01(
                minmax_scale(
                    float(artist_track_footprint),
                    float(established_track_footprint_min),
                    float(max(established_track_footprint_min + 4, established_track_footprint_min * 3)),
                )
            )
            established_penalty = clamp01(0.65 * follower_penalty + 0.35 * footprint_penalty)

        candidate_priority = clamp01(
            0.35 * low_base_accel
            + 0.25 * tastemaker_score
            + 0.2 * anomaly_score
            + 0.15 * echo_score
            + 0.05 * geo_score
            + seed_boost
            - (0.18 * established_penalty)
        )

        signal_pass = any(
            [
                low_base_accel >= 0.45,
                anomaly_score >= 0.45,
                echo_score >= max(0.45, min_echo - 0.1),
                tastemaker_score >= 0.35,
                geo_score >= 0.35,
                seed_boost > 0,
            ]
        )
        if not signal_pass:
            continue

        latest = track_rows[-1]
        candidates.append(
            {
                "track_id": track_id,
                "track_name": latest["track_name"],
                "artist_id": latest["artist_id"],
                "artist_name": latest["artist_name"],
                "genre_hint": latest.get("genre_hint", "unknown"),
                "metadata_text": latest.get("metadata_text", ""),
                "manual_seeded": manual_seeded,
                "established_artist": established_artist,
                "established_by_followers": established_by_followers,
                "established_by_footprint": established_by_footprint,
                "established_penalty": round(established_penalty, 6),
                "artist_track_footprint": int(artist_track_footprint),
                "low_base_accel": round(low_base_accel, 6),
                "tastemaker_score": round(tastemaker_score, 6),
                "anomaly_score": round(anomaly_score, 6),
                "echo_score": round(echo_score, 6),
                "geo_score": round(geo_score, 6),
                "echo_corr": round(echo_corr, 6),
                "velocity_norm": round(raw_velocity[track_id], 4),
                "engagement_rate": round(raw_engagement[track_id], 6),
                "follower_conversion": round(raw_conversion[track_id], 6),
                "diffusion_gain": diffusion_gain,
                "early_base_streams": round(early_base, 2),
                "candidate_priority": round(candidate_priority, 6),
                "tastemaker_edges": len(tm_weights),
                "latest_date": latest["date"],
            }
        )

    candidates.sort(key=lambda row: row["candidate_priority"], reverse=True)
    return candidates


def run_candidates(
    config_path: str,
    *,
    project_root: Path | None = None,
    as_of: str | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    root = project_root or Path.cwd()
    snapshot_path = resolve_path(config["paths"]["snapshots"], root)
    candidate_path = resolve_path(config["paths"]["candidates"], root)

    rows = read_jsonl(snapshot_path)
    prior_scores = _load_prior_tastemaker_scores(config, root)
    candidates = build_candidates(rows, config, as_of=as_of, prior_tastemaker_scores=prior_scores)
    write_jsonl(candidate_path, candidates)

    return {
        "input_rows": len(rows),
        "candidates": len(candidates),
        "output_path": str(candidate_path),
    }
