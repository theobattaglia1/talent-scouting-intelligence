from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from talent_scouting_intel.utils.io import ensure_parent, read_csv, resolve_path

SPLIT_RE = re.compile(r"[\/,;+|]")

ALLOWED_PLATFORMS = {
    "reddit",
    "spotify",
    "youtube",
    "lastfm",
    "rss",
}


def _as_list(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    out: list[str] = []
    for token in SPLIT_RE.split(text):
        cleaned = token.strip().lower()
        if not cleaned:
            continue
        if cleaned in {"alt", "alternative"}:
            cleaned = "alt rock"
        out.append(cleaned)
    seen: set[str] = set()
    deduped: list[str] = []
    for item in out:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _read_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "youtube_channels": [],
            "spotify_playlists": [],
            "reddit_subreddits": [],
            "lastfm_tags": [],
            "rss_feeds": [],
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("youtube_channels", [])
    payload.setdefault("spotify_playlists", [])
    payload.setdefault("reddit_subreddits", [])
    payload.setdefault("lastfm_tags", [])
    payload.setdefault("rss_feeds", [])
    return payload


def _write_registry(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _dedupe_by_key(rows: list[dict[str, Any]], key_field: str) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get(key_field, "")).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def merge_starter_source_additions(config: dict[str, Any], root: Path) -> dict[str, Any]:
    ingest_cfg = config.get("ingest", {}).get("source_registry_bootstrap", {})
    if not bool(ingest_cfg.get("enabled", True)):
        return {"enabled": False}

    starter_path = resolve_path(str(config.get("paths", {}).get("starter_sources_additions", "")), root)
    registry_path = resolve_path(str(config.get("ingest", {}).get("auto", {}).get("source_registry", "")), root)
    if not starter_path.exists():
        return {
            "enabled": True,
            "starter_sources_path": str(starter_path),
            "registry_path": str(registry_path),
            "rows_input": 0,
            "added": 0,
            "reason": "starter sources csv not found",
        }

    starter_rows = read_csv(starter_path)
    registry = _read_registry(registry_path)
    strict_allowlist = bool(ingest_cfg.get("strict_platform_allowlist", True))
    added = 0
    skipped = 0
    unsupported = 0

    for row in starter_rows:
        platform = str(row.get("platform", "")).strip().lower()
        source_type = str(row.get("source_type", "")).strip().lower()
        key = str(row.get("id_or_key", "")).strip()
        name = str(row.get("name", "")).strip() or key
        if not key:
            skipped += 1
            continue
        if strict_allowlist and platform not in ALLOWED_PLATFORMS:
            unsupported += 1
            continue

        genres = _as_list(row.get("genre_focus", ""))
        region = str(row.get("region_focus", "")).strip()

        if platform == "reddit" or source_type == "subreddit":
            before = len(registry["reddit_subreddits"])
            registry["reddit_subreddits"].append({"name": key, "genre_tags": genres})
            registry["reddit_subreddits"] = _dedupe_by_key(registry["reddit_subreddits"], "name")
            added += 1 if len(registry["reddit_subreddits"]) > before else 0
            continue

        if platform == "spotify" or source_type == "playlist":
            before = len(registry["spotify_playlists"])
            registry["spotify_playlists"].append(
                {
                    "playlist_id": key,
                    "name": name,
                    "genre_tags": genres,
                    "region": region,
                }
            )
            registry["spotify_playlists"] = _dedupe_by_key(registry["spotify_playlists"], "playlist_id")
            added += 1 if len(registry["spotify_playlists"]) > before else 0
            continue

        if platform == "youtube" or source_type == "youtube_channel":
            before = len(registry["youtube_channels"])
            registry["youtube_channels"].append(
                {
                    "channel_id": key,
                    "name": name,
                    "genre_tags": genres,
                    "estimated_followers": 0,
                }
            )
            registry["youtube_channels"] = _dedupe_by_key(registry["youtube_channels"], "channel_id")
            added += 1 if len(registry["youtube_channels"]) > before else 0
            continue

        if platform == "lastfm" or source_type == "tag":
            before = len(registry["lastfm_tags"])
            registry["lastfm_tags"].append({"name": key, "genre_tags": genres or [name.lower()]})
            registry["lastfm_tags"] = _dedupe_by_key(registry["lastfm_tags"], "name")
            added += 1 if len(registry["lastfm_tags"]) > before else 0
            continue

        if platform == "rss" or source_type == "rss_feed":
            before = len(registry["rss_feeds"])
            registry["rss_feeds"].append(
                {
                    "id": re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "rss_feed",
                    "name": name,
                    "url": key,
                    "genre_tags": genres,
                }
            )
            registry["rss_feeds"] = _dedupe_by_key(registry["rss_feeds"], "url")
            added += 1 if len(registry["rss_feeds"]) > before else 0
            continue

        unsupported += 1

    _write_registry(registry_path, registry)
    return {
        "enabled": True,
        "starter_sources_path": str(starter_path),
        "registry_path": str(registry_path),
        "rows_input": len(starter_rows),
        "added": added,
        "skipped": skipped,
        "unsupported": unsupported,
        "counts": {
            "youtube_channels": len(registry.get("youtube_channels", [])),
            "spotify_playlists": len(registry.get("spotify_playlists", [])),
            "reddit_subreddits": len(registry.get("reddit_subreddits", [])),
            "lastfm_tags": len(registry.get("lastfm_tags", [])),
            "rss_feeds": len(registry.get("rss_feeds", [])),
        },
    }
