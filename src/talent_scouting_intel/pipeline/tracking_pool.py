from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from talent_scouting_intel.utils.io import ensure_parent, read_csv, read_jsonl, resolve_path

DEFAULT_PATH = "outputs/state/tracking_pool.json"


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _parse_day(value: Any) -> dt.date | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if "T" in raw:
        raw = raw.split("T", 1)[0]
    try:
        return dt.date.fromisoformat(raw)
    except Exception:
        return None


def _suffix(value: str, prefix: str) -> str:
    if not value.startswith(prefix):
        return ""
    return value[len(prefix) :].strip()


def _pool_path(config: dict[str, Any], root: Path) -> Path:
    path_value = str(config.get("paths", {}).get("tracking_pool_state", DEFAULT_PATH))
    return resolve_path(path_value, root)


def _scored_path(config: dict[str, Any], root: Path) -> Path:
    return resolve_path(str(config.get("paths", {}).get("scored", "outputs/scored.csv")), root)


def _candidates_path(config: dict[str, Any], root: Path) -> Path:
    return resolve_path(str(config.get("paths", {}).get("candidates", "outputs/candidates.jsonl")), root)


def _ui_state_path(root: Path) -> Path:
    return root / "outputs" / "state" / "ui_state.json"


def _load_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in values:
        value = str(raw).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _load_pool(config: dict[str, Any], root: Path) -> tuple[dict[str, Any], Path]:
    path = _pool_path(config, root)
    if not path.exists():
        return {"version": 1, "updated_at": "", "items": []}, path
    payload = _load_json_dict(path)
    items = payload.get("items", [])
    if not isinstance(items, list):
        items = []
    return {
        "version": int(payload.get("version", 1) or 1),
        "updated_at": str(payload.get("updated_at", "")),
        "items": items,
    }, path


def _save_pool(path: Path, pool: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(pool, indent=2, ensure_ascii=True), encoding="utf-8")


def _item_from_row(
    row: dict[str, Any],
    *,
    today: dt.date,
    ttl_days: int,
    reason: str,
    priority: float,
) -> dict[str, Any] | None:
    track_id = str(row.get("track_id", "")).strip()
    if not track_id:
        return None

    artist_id = str(row.get("artist_id", "")).strip()
    spotify_track_id = _suffix(track_id, "sp_")
    spotify_artist_id = _suffix(artist_id, "sp_")
    youtube_video_id = _suffix(track_id, "yt_")
    youtube_channel_id = _suffix(artist_id, "ytc_")

    return {
        "track_id": track_id,
        "artist_id": artist_id,
        "track_name": str(row.get("track_name", "")).strip(),
        "artist_name": str(row.get("artist_name", "")).strip(),
        "genre": str(row.get("genre", "")).strip(),
        "spotify_track_id": spotify_track_id,
        "spotify_artist_id": spotify_artist_id,
        "youtube_video_id": youtube_video_id,
        "youtube_channel_id": youtube_channel_id,
        "priority": round(max(0.0, float(priority)), 6),
        "sources": [reason],
        "first_added": today.isoformat(),
        "last_seen": today.isoformat(),
        "expires_on": (today + dt.timedelta(days=max(1, ttl_days))).isoformat(),
        "last_final_score": round(_as_float(row.get("final_score")), 6),
        "last_candidate_priority": round(_as_float(row.get("candidate_priority")), 6),
        "last_trust_score": round(_as_float(row.get("trust_score")), 6),
    }


def _normalize_item(item: dict[str, Any], *, today: dt.date, ttl_days: int) -> dict[str, Any] | None:
    track_id = str(item.get("track_id", "")).strip()
    if not track_id:
        return None

    artist_id = str(item.get("artist_id", "")).strip()
    expires = _parse_day(item.get("expires_on")) or (today + dt.timedelta(days=max(1, ttl_days)))
    sources_raw = item.get("sources", [])
    if not isinstance(sources_raw, list):
        sources_raw = []

    return {
        "track_id": track_id,
        "artist_id": artist_id,
        "track_name": str(item.get("track_name", "")).strip(),
        "artist_name": str(item.get("artist_name", "")).strip(),
        "genre": str(item.get("genre", "")).strip(),
        "spotify_track_id": str(item.get("spotify_track_id", "")).strip() or _suffix(track_id, "sp_"),
        "spotify_artist_id": str(item.get("spotify_artist_id", "")).strip() or _suffix(artist_id, "sp_"),
        "youtube_video_id": str(item.get("youtube_video_id", "")).strip() or _suffix(track_id, "yt_"),
        "youtube_channel_id": str(item.get("youtube_channel_id", "")).strip() or _suffix(artist_id, "ytc_"),
        "priority": round(_as_float(item.get("priority")), 6),
        "sources": _dedupe([str(value).strip() for value in sources_raw if str(value).strip()]),
        "first_added": str(item.get("first_added", today.isoformat())),
        "last_seen": str(item.get("last_seen", today.isoformat())),
        "expires_on": expires.isoformat(),
        "last_final_score": round(_as_float(item.get("last_final_score")), 6),
        "last_candidate_priority": round(_as_float(item.get("last_candidate_priority")), 6),
        "last_trust_score": round(_as_float(item.get("last_trust_score")), 6),
    }


def _load_ui_feedback(root: Path) -> tuple[list[str], list[str]]:
    payload = _load_json_dict(_ui_state_path(root))
    tracked = payload.get("tracked_track_ids", [])
    ignored = payload.get("ignored_track_ids", [])
    tracked_ids = _dedupe([str(value) for value in tracked if str(value).strip()]) if isinstance(tracked, list) else []
    ignored_ids = _dedupe([str(value) for value in ignored if str(value).strip()]) if isinstance(ignored, list) else []
    return tracked_ids, ignored_ids


def refresh_tracking_pool(
    config: dict[str, Any],
    root: Path,
    *,
    scored_rows: list[dict[str, Any]] | None = None,
    candidates_rows: list[dict[str, Any]] | None = None,
    today: dt.date | None = None,
) -> dict[str, Any]:
    cfg = config.get("tracking_pool", {})
    enabled = bool(cfg.get("enabled", True))
    if not enabled:
        return {"enabled": False}

    now = today or dt.date.today()
    ttl_days = int(cfg.get("ttl_days", 30))
    max_items = int(cfg.get("max_items", 320))
    auto_cfg = cfg.get("auto_add", {})

    pool, pool_path = _load_pool(config, root)
    active_by_track: dict[str, dict[str, Any]] = {}
    expired_removed = 0

    for raw in pool.get("items", []):
        if not isinstance(raw, dict):
            continue
        item = _normalize_item(raw, today=now, ttl_days=ttl_days)
        if not item:
            continue
        expires = _parse_day(item.get("expires_on"))
        if expires and expires < now:
            expired_removed += 1
            continue
        active_by_track[item["track_id"]] = item

    added_from_scored = 0
    added_from_candidates = 0
    added_from_ui = 0
    removed_ignored = 0
    ignored_set: set[str] = set()

    def upsert(row: dict[str, Any], *, reason: str, priority: float) -> bool:
        nonlocal added_from_scored, added_from_candidates, added_from_ui
        item = _item_from_row(row, today=now, ttl_days=ttl_days, reason=reason, priority=priority)
        if not item:
            return False
        track_id = item["track_id"]
        if remove_ignored and track_id in ignored_set:
            return False
        existing = active_by_track.get(track_id)
        if existing is None:
            active_by_track[track_id] = item
            if reason == "scored_auto":
                added_from_scored += 1
            elif reason == "candidate_auto":
                added_from_candidates += 1
            elif reason == "ui_tracked":
                added_from_ui += 1
            return True

        existing["last_seen"] = now.isoformat()
        existing["expires_on"] = (now + dt.timedelta(days=max(1, ttl_days))).isoformat()
        existing["priority"] = round(max(_as_float(existing.get("priority")), _as_float(item.get("priority"))), 6)
        existing["sources"] = _dedupe(list(existing.get("sources", [])) + [reason])

        for key in [
            "artist_id",
            "track_name",
            "artist_name",
            "genre",
            "spotify_track_id",
            "spotify_artist_id",
            "youtube_video_id",
            "youtube_channel_id",
        ]:
            existing_value = str(existing.get(key, "")).strip()
            candidate_value = str(item.get(key, "")).strip()
            if (not existing_value) and candidate_value:
                existing[key] = candidate_value

        existing["last_final_score"] = round(max(_as_float(existing.get("last_final_score")), _as_float(item.get("last_final_score"))), 6)
        existing["last_candidate_priority"] = round(
            max(_as_float(existing.get("last_candidate_priority")), _as_float(item.get("last_candidate_priority"))), 6
        )
        existing["last_trust_score"] = round(max(_as_float(existing.get("last_trust_score")), _as_float(item.get("last_trust_score"))), 6)
        return False

    include_ui_feedback = bool(cfg.get("include_ui_feedback", True))
    remove_ignored = bool(cfg.get("remove_ignored", True))
    if include_ui_feedback:
        tracked_ids, ignored_ids = _load_ui_feedback(root)
        ignored_set = set(ignored_ids)
        for track_id in tracked_ids:
            if remove_ignored and track_id in ignored_set:
                continue
            upsert({"track_id": track_id}, reason="ui_tracked", priority=1.0)
        if remove_ignored:
            for track_id in ignored_ids:
                if track_id in active_by_track:
                    del active_by_track[track_id]
                    removed_ignored += 1

    if bool(auto_cfg.get("from_scored", True)):
        rows = scored_rows
        if rows is None:
            rows = read_csv(_scored_path(config, root))
        rows = [row for row in rows if isinstance(row, dict)]
        rows_sorted = sorted(rows, key=lambda row: _as_float(row.get("final_score")), reverse=True)
        top_n = int(auto_cfg.get("from_scored_top_n", 120))
        include_baseline = bool(auto_cfg.get("include_baseline", True))
        include_inflections = bool(auto_cfg.get("include_inflections", True))
        skip_flagged = bool(auto_cfg.get("skip_flagged", True))
        min_final = _as_float(auto_cfg.get("min_final_score", 0.0))
        min_prior_gate = _as_float(auto_cfg.get("min_prior_gate", 0.0))
        min_trust = _as_float(auto_cfg.get("min_trust_score", 0.0))

        for row in rows_sorted[: max(1, top_n)]:
            if not include_baseline and str(row.get("stage", "")).strip().lower() == "baseline":
                continue
            if skip_flagged and (_as_bool(row.get("spike_only")) or _as_bool(row.get("suspicious"))):
                continue
            if _as_float(row.get("final_score")) < min_final:
                continue
            if _as_float(row.get("prior_gate")) < min_prior_gate:
                continue
            if _as_float(row.get("trust_score")) < min_trust:
                continue
            priority = clamp_priority(
                0.5 * _as_float(row.get("final_score"))
                + 0.2 * _as_float(row.get("trust_score"))
                + 0.2 * _as_float(row.get("prior_gate"))
                + 0.1 * _as_float(row.get("candidate_priority"))
            )
            upsert(row, reason="scored_auto", priority=priority)

        if include_inflections:
            for row in rows:
                if not _as_bool(row.get("inflection_detected")):
                    continue
                priority = clamp_priority(0.75 + 0.25 * _as_float(row.get("final_score")))
                upsert(row, reason="scored_auto", priority=priority)

    if bool(auto_cfg.get("from_candidates", True)):
        rows = candidates_rows
        if rows is None:
            rows = read_jsonl(_candidates_path(config, root))
        rows = [row for row in rows if isinstance(row, dict)]
        rows_sorted = sorted(rows, key=lambda row: _as_float(row.get("candidate_priority")), reverse=True)
        top_n = int(auto_cfg.get("candidate_top_n", 160))
        min_priority = _as_float(auto_cfg.get("candidate_min_priority", 0.2))
        skip_established = bool(auto_cfg.get("candidate_skip_established", True))

        for row in rows_sorted[: max(1, top_n)]:
            if skip_established and _as_bool(row.get("established_artist")) and not _as_bool(row.get("manual_seeded")):
                continue
            priority = _as_float(row.get("candidate_priority"))
            if priority < min_priority:
                continue
            upsert(row, reason="candidate_auto", priority=clamp_priority(priority))

    items = list(active_by_track.values())
    items.sort(key=_priority_key, reverse=True)

    overflow_pruned = max(0, len(items) - max(1, max_items))
    if overflow_pruned > 0:
        items = items[: max(1, max_items)]

    pool["version"] = 1
    pool["updated_at"] = now.isoformat()
    pool["items"] = items
    _save_pool(pool_path, pool)

    return {
        "enabled": True,
        "path": str(pool_path),
        "items_active": len(items),
        "expired_removed": expired_removed,
        "removed_ignored": removed_ignored,
        "added_from_scored": added_from_scored,
        "added_from_candidates": added_from_candidates,
        "added_from_ui": added_from_ui,
        "overflow_pruned": overflow_pruned,
        "ttl_days": ttl_days,
    }


def clamp_priority(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _priority_key(item: dict[str, Any]) -> tuple[int, float, str, str]:
    sources = item.get("sources", [])
    has_ui_track = 1 if isinstance(sources, list) and "ui_tracked" in sources else 0
    return (
        has_ui_track,
        _as_float(item.get("priority")),
        str(item.get("last_seen", "")),
        str(item.get("track_id", "")),
    )


def build_tracking_targets(
    config: dict[str, Any],
    root: Path,
    *,
    today: dt.date | None = None,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    cfg = config.get("tracking_pool", {})
    enabled = bool(cfg.get("enabled", True))
    if not enabled:
        return {}, {"enabled": False}

    now = today or dt.date.today()
    ttl_days = int(cfg.get("ttl_days", 30))
    pool, pool_path = _load_pool(config, root)

    active_items: list[dict[str, Any]] = []
    expired_removed = 0
    for raw in pool.get("items", []):
        if not isinstance(raw, dict):
            continue
        item = _normalize_item(raw, today=now, ttl_days=ttl_days)
        if not item:
            continue
        expires = _parse_day(item.get("expires_on"))
        if expires and expires < now:
            expired_removed += 1
            continue
        active_items.append(item)

    if expired_removed:
        pool["version"] = 1
        pool["updated_at"] = now.isoformat()
        pool["items"] = active_items
        _save_pool(pool_path, pool)

    active_items.sort(key=_priority_key, reverse=True)
    caps = cfg.get("adapter_targets", {})

    def collect(field: str, cap_key: str, default: int) -> list[str]:
        cap = int(caps.get(cap_key, default))
        out: list[str] = []
        seen: set[str] = set()
        for item in active_items:
            value = str(item.get(field, "")).strip()
            if not value or value in seen:
                continue
            seen.add(value)
            out.append(value)
            if len(out) >= max(1, cap):
                break
        return out

    track_ids = collect("track_id", "max_track_ids", 320)
    spotify_track_ids = collect("spotify_track_id", "max_spotify_tracks", 180)
    spotify_artist_ids = collect("spotify_artist_id", "max_spotify_artists", 120)
    youtube_video_ids = collect("youtube_video_id", "max_youtube_videos", 120)
    youtube_channel_ids = collect("youtube_channel_id", "max_youtube_channels", 80)
    artist_names = collect("artist_name", "max_artist_names", 200)

    targets = {
        "track_ids": track_ids,
        "spotify_track_ids": spotify_track_ids,
        "spotify_artist_ids": spotify_artist_ids,
        "youtube_video_ids": youtube_video_ids,
        "youtube_channel_ids": youtube_channel_ids,
        "artist_names": artist_names,
    }

    stats = {
        "enabled": True,
        "path": str(pool_path),
        "items_active": len(active_items),
        "expired_removed": expired_removed,
        "targets": {key: len(value) for key, value in targets.items()},
    }
    return targets, stats
