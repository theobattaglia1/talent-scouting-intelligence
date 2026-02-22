from __future__ import annotations

import datetime as dt
from collections import defaultdict
from pathlib import Path
from typing import Any

from talent_scouting_intel.pipeline.common import group_by_track
from talent_scouting_intel.utils.io import load_config, read_csv, read_jsonl, resolve_path, write_csv, write_json
from talent_scouting_intel.utils.math_utils import clamp01, mean, minmax_scale


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_day(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def _track_break_dates(scored_rows: list[dict[str, Any]], success_score_min: float) -> dict[str, dt.date]:
    break_dates: dict[str, dt.date] = {}
    for row in scored_rows:
        score = _as_float(row.get("final_score"))
        inflection = _as_bool(row.get("inflection_detected"))
        if not inflection and score < success_score_min:
            continue
        track_id = str(row.get("track_id", ""))
        date_str = str(row.get("latest_date", ""))
        if not track_id or not date_str:
            continue
        break_dates[track_id] = _parse_day(date_str)
    return break_dates


def build_tastemaker_profiles(
    snapshots: list[dict[str, Any]],
    scored_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    if not snapshots:
        return []

    discovery_cfg = config.get("tastemaker_discovery", {})
    weights = discovery_cfg.get("weights", {})
    min_trials = int(discovery_cfg.get("min_trials", 3))
    novelty_cutoff = int(discovery_cfg.get("novelty_follower_threshold", 120000))
    min_lead_days = int(discovery_cfg.get("min_lead_days", 3))
    max_lead_score_days = int(discovery_cfg.get("max_lead_score_days", 30))
    success_score_min = float(discovery_cfg.get("success_score_min", 0.52))

    prior_success = float(config.get("candidate_generation", {}).get("tastemakers", {}).get("bayes_prior_success", 1.0))
    prior_trials = float(config.get("candidate_generation", {}).get("tastemakers", {}).get("bayes_prior_trials", 6.0))

    priority_genres = set(config.get("genres", {}).get("priority", []))
    scored_by_track = {str(row.get("track_id", "")): row for row in scored_rows}
    break_dates = _track_break_dates(scored_rows, success_score_min)

    grouped = group_by_track(snapshots)
    total_tracks = max(1, len(grouped))

    tm_track_first_pick: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    tm_name: dict[str, str] = {}

    for row in snapshots:
        tm_id = str(row.get("tastemaker_id") or "").strip()
        if not tm_id:
            continue
        track_id = str(row.get("track_id", ""))
        if not track_id:
            continue

        name = str(row.get("tastemaker_name") or tm_id)
        tm_name[tm_id] = name

        pick_date = _parse_day(str(row.get("date")))
        existing = tm_track_first_pick[tm_id].get(track_id)
        if existing is None or pick_date < existing["pick_date"]:
            tm_track_first_pick[tm_id][track_id] = {
                "pick_date": pick_date,
                "artist_followers": int(row.get("artist_followers", 0) or 0),
                "track_id": track_id,
            }

    profiles: list[dict[str, Any]] = []
    for tm_id, picks in tm_track_first_pick.items():
        trials = len(picks)
        if trials == 0:
            continue

        success = 0
        early_success = 0
        lead_days: list[int] = []
        novelty_hits = 0
        genre_hits = 0
        risky_hits = 0

        for pick in picks.values():
            track_id = pick["track_id"]
            scored = scored_by_track.get(track_id, {})

            genre = str(scored.get("genre", "unknown"))
            if genre in priority_genres:
                genre_hits += 1

            if pick["artist_followers"] <= novelty_cutoff:
                novelty_hits += 1

            risky = (
                _as_bool(scored.get("spike_only"))
                or _as_bool(scored.get("suspicious"))
                or _as_bool(scored.get("playlist_dependent"))
            )
            if risky:
                risky_hits += 1

            broke_day = break_dates.get(track_id)
            if broke_day is None:
                continue

            success += 1
            lead = (broke_day - pick["pick_date"]).days
            if lead >= 0:
                lead_days.append(lead)
            if lead >= min_lead_days:
                early_success += 1

        bayes_precision = (prior_success + success) / (prior_trials + trials)
        avg_lead_days = mean([float(value) for value in lead_days]) if lead_days else 0.0
        lead_score = minmax_scale(avg_lead_days, 0.0, float(max_lead_score_days))
        early_capture = early_success / max(1, success)
        novelty_ratio = novelty_hits / trials
        genre_alignment = genre_hits / trials
        reliability = 1.0 - (risky_hits / trials)
        centrality = (trials / total_tracks) ** 0.5

        quant_score = clamp01(
            float(weights.get("precision", 0.36)) * bayes_precision
            + float(weights.get("lead_time", 0.2)) * lead_score
            + float(weights.get("early_capture", 0.12)) * early_capture
            + float(weights.get("novelty", 0.12)) * novelty_ratio
            + float(weights.get("genre_alignment", 0.1)) * genre_alignment
            + float(weights.get("reliability", 0.08)) * reliability
            + float(weights.get("centrality", 0.02)) * centrality
        )

        if trials >= min_trials:
            status = "qualified"
        elif quant_score >= 0.55 and early_success >= 1:
            status = "emergent"
        else:
            status = "incubating"

        profiles.append(
            {
                "tastemaker_id": tm_id,
                "tastemaker_name": tm_name.get(tm_id, tm_id),
                "trials": trials,
                "successes": success,
                "bayes_precision": round(bayes_precision, 6),
                "early_successes": early_success,
                "avg_lead_days": round(avg_lead_days, 3),
                "lead_score": round(lead_score, 6),
                "early_capture": round(early_capture, 6),
                "novelty_ratio": round(novelty_ratio, 6),
                "genre_alignment": round(genre_alignment, 6),
                "reliability": round(reliability, 6),
                "centrality": round(centrality, 6),
                "quant_score": round(quant_score, 6),
                "status": status,
            }
        )

    profiles.sort(key=lambda row: (_as_float(row.get("quant_score")), _as_float(row.get("bayes_precision"))), reverse=True)
    return profiles


def run_tastemakers(
    config_path: str,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    root = project_root or Path.cwd()

    snapshot_path = resolve_path(config["paths"]["snapshots"], root)
    scored_path = resolve_path(config["paths"]["scored"], root)
    out_csv = resolve_path(config["paths"]["tastemakers_csv"], root)
    out_json = resolve_path(config["paths"]["tastemakers_json"], root)

    snapshots = read_jsonl(snapshot_path)
    scored_rows = read_csv(scored_path)
    profiles = build_tastemaker_profiles(snapshots, scored_rows, config)

    write_csv(out_csv, profiles)
    top_n = int(config.get("tastemaker_discovery", {}).get("top_n", 25))
    payload = {
        "generated_at": dt.datetime.now().replace(microsecond=0).isoformat(),
        "total": len(profiles),
        "top": profiles[:top_n],
    }
    write_json(out_json, payload)

    return {
        "profiles": len(profiles),
        "tastemakers_csv": str(out_csv),
        "tastemakers_json": str(out_json),
    }
