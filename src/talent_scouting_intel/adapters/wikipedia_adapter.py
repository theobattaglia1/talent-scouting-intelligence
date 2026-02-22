from __future__ import annotations

import datetime as dt
import json
import re
import urllib.parse
import urllib.request
from collections import defaultdict
from typing import Any

SLUG_RE = re.compile(r"[^A-Za-z0-9_\-]+")


def _sanitize_title(title: str) -> str:
    clean = title.replace(" ", "_")
    clean = SLUG_RE.sub("", clean)
    return clean.strip("_")


def _request_json(url: str, timeout: int = 4) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "tsi-bot/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = response.read().decode("utf-8", errors="ignore")
    obj = json.loads(payload)
    if isinstance(obj, dict):
        return obj
    return {}


def _resolve_wikipedia_title(name: str) -> str | None:
    params = urllib.parse.urlencode(
        {
            "action": "opensearch",
            "search": name,
            "limit": 1,
            "namespace": 0,
            "format": "json",
        }
    )
    url = f"https://en.wikipedia.org/w/api.php?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "tsi-bot/0.1"})
        with urllib.request.urlopen(req, timeout=4) as response:
            payload = json.loads(response.read().decode("utf-8", errors="ignore"))
    except Exception:
        return None

    if isinstance(payload, list) and len(payload) >= 2 and isinstance(payload[1], list) and payload[1]:
        title = str(payload[1][0]).strip()
        return title if title else None
    return None


def _pageviews(title: str, start: dt.date, end: dt.date) -> list[int]:
    safe = urllib.parse.quote(_sanitize_title(title), safe="")
    if not safe:
        return []
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")
    url = (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"en.wikipedia.org/all-access/user/{safe}/daily/{start_str}/{end_str}"
    )
    try:
        payload = _request_json(url)
    except Exception:
        return []
    items = payload.get("items", [])
    if not isinstance(items, list):
        return []
    out: list[int] = []
    for item in items:
        try:
            out.append(int(item.get("views", 0) or 0))
        except Exception:
            out.append(0)
    return out


def collect_wikipedia_snapshots(
    config: dict[str, Any],
    base_rows: list[dict[str, Any]],
    *,
    today: dt.date,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = config.get("ingest", {}).get("auto", {}).get("wikipedia", {})
    if not bool(cfg.get("enabled", True)):
        return [], {"enabled": False, "reason": "wikipedia adapter disabled"}

    lookback_days = int(cfg.get("lookback_days", 21))
    max_artists = int(cfg.get("max_artists", 35))

    by_artist: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in base_rows:
        artist = str(row.get("artist_name", "")).strip()
        if not artist:
            continue
        by_artist[artist].append(row)

    ranked_artists = sorted(
        by_artist.keys(),
        key=lambda name: sum(float(item.get("views", 0) or 0) for item in by_artist[name]),
        reverse=True,
    )[:max_artists]

    rows: list[dict[str, Any]] = []
    resolved = 0

    start = today - dt.timedelta(days=lookback_days)
    end = today - dt.timedelta(days=1)

    for artist in ranked_artists:
        title = _resolve_wikipedia_title(artist)
        if not title:
            continue
        resolved += 1
        series = _pageviews(title, start, end)
        if not series:
            continue

        latest_views = series[-1]
        trailing = series[-7:] if len(series) >= 7 else series
        avg_recent = sum(trailing) / max(1, len(trailing))
        growth = (latest_views - avg_recent) / (avg_recent + 1.0)

        artist_rows = by_artist[artist]
        unique_tracks: dict[str, dict[str, Any]] = {}
        for row in artist_rows:
            track_id = str(row.get("track_id", "")).strip()
            if not track_id:
                continue
            if track_id not in unique_tracks:
                unique_tracks[track_id] = row

        for track_id, rep in list(unique_tracks.items())[:6]:
            views = max(1, int(latest_views))
            rows.append(
                {
                    "date": today.isoformat(),
                    "platform": "wikipedia",
                    "track_id": track_id,
                    "track_name": rep.get("track_name", ""),
                    "artist_id": rep.get("artist_id", ""),
                    "artist_name": artist,
                    "artist_followers": int(rep.get("artist_followers", 0) or 0),
                    "release_date": str(rep.get("release_date", today.isoformat())),
                    "genre_hint": str(rep.get("genre_hint", "")),
                    "views": views,
                    "likes": max(0, int(views * 0.01)),
                    "comments": max(0, int(views * 0.0008)),
                    "shares": max(0, int(views * 0.002)),
                    "streams": int(views * 0.35),
                    "listeners": int(views * 0.22),
                    "playlist_adds": 0,
                    "creator_reuse": 0,
                    "region_metrics": {},
                    "tastemaker_id": "wiki_pageviews",
                    "tastemaker_name": "Wikipedia Pageviews",
                    "event_type": "knowledge_attention",
                    "source": "wikimedia",
                    "comments_text": [f"wiki_growth={growth:.4f}", f"wiki_title={title}"],
                    "collaborators": list(rep.get("collaborators", [])),
                    "manual_seeded": bool(rep.get("manual_seeded", False)),
                    "metadata_text": f"wikipedia {title} growth {growth:.4f}",
                }
            )

    return rows, {
        "enabled": True,
        "artists_seen": len(ranked_artists),
        "artists_resolved": resolved,
        "rows": len(rows),
    }
