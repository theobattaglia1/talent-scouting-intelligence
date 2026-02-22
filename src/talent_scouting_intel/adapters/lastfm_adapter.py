from __future__ import annotations

import datetime as dt
import json
import os
import re
import urllib.parse
import urllib.request
from typing import Any

SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    value = SLUG_RE.sub("-", text.lower()).strip("-")
    return value or "unknown"


def _request_json(url: str, timeout: int = 4) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "tsi-bot/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = response.read().decode("utf-8", errors="ignore")
    obj = json.loads(payload)
    if isinstance(obj, dict):
        return obj
    return {}


def _extract_seed_tokens(seed_urls: list[str]) -> list[str]:
    tokens: list[str] = []
    for url in seed_urls:
        lowered = url.lower()
        for splitter in ["/", "?", "=", "&", "-"]:
            lowered = lowered.replace(splitter, " ")
        for token in lowered.split():
            if len(token) >= 4:
                tokens.append(token)
    return tokens


def _is_seeded(seed_tokens: list[str], *values: str) -> bool:
    haystack = " ".join(values).lower()
    return any(token in haystack for token in seed_tokens)


def collect_lastfm_snapshots(
    config: dict[str, Any],
    registry: dict[str, Any],
    *,
    today: dt.date,
    seed_urls: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = config.get("ingest", {}).get("auto", {}).get("lastfm", {})
    if not bool(cfg.get("enabled", True)):
        return [], {"enabled": False, "reason": "lastfm adapter disabled"}

    tags = registry.get("lastfm_tags", []) if isinstance(registry, dict) else []
    if not isinstance(tags, list) or not tags:
        return [], {"enabled": True, "reason": "no lastfm tags configured", "tags": 0}

    api_key = os.getenv(str(cfg.get("api_key_env", "LASTFM_API_KEY")), "")
    if not api_key:
        return [], {
            "enabled": True,
            "reason": "missing Last.fm API key",
            "required_env": [str(cfg.get("api_key_env", "LASTFM_API_KEY"))],
            "tags": len(tags),
        }

    limit = int(cfg.get("limit_per_tag", 40))
    seed_tokens = _extract_seed_tokens(seed_urls or [])

    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for tag in tags:
        if not isinstance(tag, dict):
            continue
        tag_name = str(tag.get("name", "")).strip()
        if not tag_name:
            continue

        params = {
            "method": "tag.gettoptracks",
            "tag": tag_name,
            "limit": limit,
            "api_key": api_key,
            "format": "json",
        }
        url = "https://ws.audioscrobbler.com/2.0/?" + urllib.parse.urlencode(params)

        try:
            payload = _request_json(url)
        except Exception:
            errors.append(f"tag {tag_name}: request failed")
            continue

        toptracks = ((payload.get("tracks") or {}).get("track") or [])
        if isinstance(toptracks, dict):
            toptracks = [toptracks]
        if not isinstance(toptracks, list):
            continue

        genre_hint = " ".join(str(item) for item in tag.get("genre_tags", [tag_name]))
        for track in toptracks:
            track_name = str(track.get("name", "")).strip()
            artist_name = str((track.get("artist") or {}).get("name", "")).strip()
            if not track_name or not artist_name:
                continue

            listeners = int(float(track.get("listeners", 0) or 0))
            playcount = int(float(track.get("playcount", 0) or 0))
            proxy_streams = max(listeners, int(playcount * 0.04))
            proxy_views = max(proxy_streams, int(playcount * 0.07))
            track_id = f"lfm_{_slug(artist_name)}_{_slug(track_name)}"

            rows.append(
                {
                    "date": today.isoformat(),
                    "platform": "lastfm",
                    "track_id": track_id,
                    "track_name": track_name,
                    "artist_id": f"lfm_{_slug(artist_name)}",
                    "artist_name": artist_name,
                    "artist_followers": listeners,
                    "release_date": today.isoformat(),
                    "genre_hint": genre_hint,
                    "views": proxy_views,
                    "likes": int(proxy_views * 0.04),
                    "comments": int(proxy_views * 0.0009),
                    "shares": int(proxy_views * 0.003),
                    "streams": proxy_streams,
                    "listeners": listeners,
                    "playlist_adds": 0,
                    "creator_reuse": 0,
                    "region_metrics": {},
                    "tastemaker_id": f"lfm_tag_{_slug(tag_name)}",
                    "tastemaker_name": f"Last.fm:{tag_name}",
                    "event_type": "tag_chart",
                    "source": "lastfm_api",
                    "comments_text": [],
                    "collaborators": [],
                    "manual_seeded": _is_seeded(seed_tokens, track_name, artist_name, tag_name),
                    "metadata_text": f"{track_name} {artist_name} {genre_hint}".strip(),
                }
            )

    return rows, {
        "enabled": True,
        "tags": len(tags),
        "rows": len(rows),
        "errors": errors[:10],
    }
