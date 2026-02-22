from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from talent_scouting_intel.utils.io import ensure_parent


DEFAULT_GENRES = [
    "pop",
    "indie pop",
    "singer-songwriter",
    "country-pop",
    "indie folk",
    "alt rock",
]

_SOURCE_KEYS = [
    "youtube_channels",
    "spotify_playlists",
    "reddit_subreddits",
    "lastfm_tags",
    "rss_feeds",
]

DEFAULT_STATE: dict[str, Any] = {
    "onboarding_complete": False,
    "onboarding_step": 1,
    "genre_focus": list(DEFAULT_GENRES),
    "discovery_source_mode": "starter_pack",
    "run_mode": "auto",
    "run_with_backtest": False,
    "tour_dismissed": False,
    "tracked_track_ids": [],
    "ignored_track_ids": [],
    "thumbs": {},
    "custom_sources": {key: [] for key in _SOURCE_KEYS},
    "hide_flags_default": {
        "spike_only": True,
        "suspicious": True,
        "playlist_dependent": False,
        "established_artist": True,
        "insufficient_history": True,
        "low_trust": True,
    },
}


def ui_state_path(project_root: Path) -> Path:
    return project_root / "outputs" / "state" / "ui_state.json"


def _coerce_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _coerce_thumbs(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, int] = {}
    for key, raw in value.items():
        score = 1 if int(raw) > 0 else -1
        out[str(key)] = score
    return out


def _normalize_source_entry(source_type: str, entry: Any) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None

    out = {"genre_tags": _coerce_string_list(entry.get("genre_tags", []))}

    if source_type == "youtube_channels":
        channel_id = str(entry.get("channel_id", "")).strip()
        if not channel_id:
            return None
        out.update(
            {
                "channel_id": channel_id,
                "name": str(entry.get("name", channel_id)).strip(),
                "estimated_followers": int(float(entry.get("estimated_followers", 0) or 0)),
            }
        )
        return out

    if source_type == "spotify_playlists":
        playlist_id = str(entry.get("playlist_id", "")).strip()
        if not playlist_id:
            return None
        out.update(
            {
                "playlist_id": playlist_id,
                "name": str(entry.get("name", playlist_id)).strip(),
                "region": str(entry.get("region", "United States")).strip() or "United States",
            }
        )
        return out

    if source_type == "reddit_subreddits":
        name = str(entry.get("name", "")).strip()
        if not name:
            return None
        out.update({"name": name})
        return out

    if source_type == "lastfm_tags":
        name = str(entry.get("name", "")).strip()
        if not name:
            return None
        out.update({"name": name})
        return out

    if source_type == "rss_feeds":
        feed_id = str(entry.get("id", "")).strip()
        url = str(entry.get("url", "")).strip()
        if not feed_id or not url:
            return None
        out.update(
            {
                "id": feed_id,
                "name": str(entry.get("name", feed_id)).strip() or feed_id,
                "url": url,
            }
        )
        return out

    return None


def normalize_custom_sources(payload: Any) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {key: [] for key in _SOURCE_KEYS}
    if not isinstance(payload, dict):
        return out

    for key in _SOURCE_KEYS:
        seen: set[str] = set()
        for item in payload.get(key, []) if isinstance(payload.get(key), list) else []:
            normalized = _normalize_source_entry(key, item)
            if normalized is None:
                continue

            if key == "youtube_channels":
                dedupe_key = normalized["channel_id"]
            elif key == "spotify_playlists":
                dedupe_key = normalized["playlist_id"]
            elif key == "rss_feeds":
                dedupe_key = normalized["id"]
            else:
                dedupe_key = normalized["name"]

            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            out[key].append(normalized)

    return out


def load_ui_state(project_root: Path) -> dict[str, Any]:
    path = ui_state_path(project_root)
    state = copy.deepcopy(DEFAULT_STATE)
    if not path.exists():
        return state

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return state

    if not isinstance(payload, dict):
        return state

    state["onboarding_complete"] = bool(payload.get("onboarding_complete", state["onboarding_complete"]))
    state["onboarding_step"] = int(payload.get("onboarding_step", state["onboarding_step"]))
    state["genre_focus"] = _coerce_string_list(payload.get("genre_focus", state["genre_focus"])) or list(DEFAULT_GENRES)

    source_mode = str(payload.get("discovery_source_mode", state["discovery_source_mode"])).strip().lower()
    state["discovery_source_mode"] = source_mode if source_mode in {"starter_pack", "custom"} else "starter_pack"

    run_mode = str(payload.get("run_mode", state["run_mode"])).strip().lower()
    state["run_mode"] = run_mode if run_mode in {"auto", "mock", "hybrid"} else "auto"

    state["run_with_backtest"] = bool(payload.get("run_with_backtest", state["run_with_backtest"]))
    state["tour_dismissed"] = bool(payload.get("tour_dismissed", state["tour_dismissed"]))

    state["tracked_track_ids"] = _coerce_string_list(payload.get("tracked_track_ids", []))
    state["ignored_track_ids"] = _coerce_string_list(payload.get("ignored_track_ids", []))
    state["thumbs"] = _coerce_thumbs(payload.get("thumbs", {}))

    hide_flags = payload.get("hide_flags_default", {})
    if isinstance(hide_flags, dict):
        state["hide_flags_default"] = {
            "spike_only": bool(hide_flags.get("spike_only", state["hide_flags_default"]["spike_only"])),
            "suspicious": bool(hide_flags.get("suspicious", state["hide_flags_default"]["suspicious"])),
            "playlist_dependent": bool(
                hide_flags.get("playlist_dependent", state["hide_flags_default"]["playlist_dependent"])
            ),
            "established_artist": bool(
                hide_flags.get("established_artist", state["hide_flags_default"]["established_artist"])
            ),
            "insufficient_history": bool(
                hide_flags.get("insufficient_history", state["hide_flags_default"]["insufficient_history"])
            ),
            "low_trust": bool(hide_flags.get("low_trust", state["hide_flags_default"]["low_trust"])),
        }

    state["custom_sources"] = normalize_custom_sources(payload.get("custom_sources"))
    return state


def save_ui_state(project_root: Path, state: dict[str, Any]) -> None:
    payload = copy.deepcopy(DEFAULT_STATE)
    payload.update(state)
    payload["custom_sources"] = normalize_custom_sources(payload.get("custom_sources"))

    ensure_parent(ui_state_path(project_root))
    ui_state_path(project_root).write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def merge_source_registry(starter: dict[str, Any], custom: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    normalized_custom = normalize_custom_sources(custom)

    for key in _SOURCE_KEYS:
        base_rows = starter.get(key, []) if isinstance(starter.get(key), list) else []
        custom_rows = normalized_custom.get(key, [])

        combined: list[dict[str, Any]] = []
        dedupe: set[str] = set()

        for row in base_rows + custom_rows:
            normalized = _normalize_source_entry(key, row)
            if normalized is None:
                continue

            if key == "youtube_channels":
                dedupe_key = normalized["channel_id"]
            elif key == "spotify_playlists":
                dedupe_key = normalized["playlist_id"]
            elif key == "rss_feeds":
                dedupe_key = normalized["id"]
            else:
                dedupe_key = normalized["name"]

            if dedupe_key in dedupe:
                continue
            dedupe.add(dedupe_key)
            combined.append(normalized)

        merged[key] = combined

    return merged
