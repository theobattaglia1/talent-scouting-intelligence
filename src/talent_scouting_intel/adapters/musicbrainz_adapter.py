from __future__ import annotations

import datetime as dt
import json
import re
import urllib.parse
import urllib.request
from collections import defaultdict
from typing import Any

SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    value = SLUG_RE.sub("-", text.lower()).strip("-")
    return value or "unknown"


def _request_json(url: str, timeout: int = 4) -> dict[str, Any]:
    headers = {
        "User-Agent": "talent-scouting-intelligence/0.1 (research@localhost)",
        "Accept": "application/json",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = response.read().decode("utf-8", errors="ignore")
    obj = json.loads(payload)
    if isinstance(obj, dict):
        return obj
    return {}


def _search_artist(name: str) -> tuple[str | None, float]:
    params = urllib.parse.urlencode({"query": f"artist:{name}", "fmt": "json", "limit": 1})
    url = f"https://musicbrainz.org/ws/2/artist/?{params}"
    try:
        payload = _request_json(url)
    except Exception:
        return None, 0.0

    artists = payload.get("artists", [])
    if not isinstance(artists, list) or not artists:
        return None, 0.0
    artist = artists[0]
    artist_id = str(artist.get("id", "")).strip()
    score = float(artist.get("score", 0.0) or 0.0) / 100.0
    if not artist_id:
        return None, 0.0
    return artist_id, score


def _artist_detail(artist_id: str) -> dict[str, Any]:
    params = urllib.parse.urlencode({"fmt": "json", "inc": "artist-rels+tags+url-rels"})
    url = f"https://musicbrainz.org/ws/2/artist/{urllib.parse.quote(artist_id)}?{params}"
    try:
        return _request_json(url)
    except Exception:
        return {}


def collect_musicbrainz_snapshots(
    config: dict[str, Any],
    base_rows: list[dict[str, Any]],
    *,
    today: dt.date,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = config.get("ingest", {}).get("auto", {}).get("musicbrainz", {})
    if not bool(cfg.get("enabled", True)):
        return [], {"enabled": False, "reason": "musicbrainz adapter disabled"}

    max_artists = int(cfg.get("max_artists", 25))

    by_artist: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in base_rows:
        artist = str(row.get("artist_name", "")).strip()
        if artist:
            by_artist[artist].append(row)

    ranked_artists = sorted(
        by_artist.keys(),
        key=lambda name: sum(float(item.get("views", 0) or 0) for item in by_artist[name]),
        reverse=True,
    )[:max_artists]

    rows: list[dict[str, Any]] = []
    resolved = 0

    for artist in ranked_artists:
        mbid, match_score = _search_artist(artist)
        if not mbid:
            continue
        resolved += 1
        detail = _artist_detail(mbid)

        rels = detail.get("relations", [])
        tags = detail.get("tags", [])
        relation_count = len(rels) if isinstance(rels, list) else 0
        tag_count = len(tags) if isinstance(tags, list) else 0
        confidence = min(1.0, 0.45 * match_score + 0.35 * min(1.0, relation_count / 20.0) + 0.2 * min(1.0, tag_count / 15.0))

        tag_names: list[str] = []
        if isinstance(tags, list):
            for tag in tags[:8]:
                name = str(tag.get("name", "")).strip()
                if name:
                    tag_names.append(name)

        artist_rows = by_artist[artist]
        unique_tracks: dict[str, dict[str, Any]] = {}
        for row in artist_rows:
            track_id = str(row.get("track_id", "")).strip()
            if track_id and track_id not in unique_tracks:
                unique_tracks[track_id] = row

        for track_id, rep in list(unique_tracks.items())[:6]:
            views = max(1, int(40 + relation_count * 12 + tag_count * 9 + match_score * 50))
            rows.append(
                {
                    "date": today.isoformat(),
                    "platform": "musicbrainz",
                    "track_id": track_id,
                    "track_name": rep.get("track_name", ""),
                    "artist_id": rep.get("artist_id", ""),
                    "artist_name": artist,
                    "artist_followers": int(rep.get("artist_followers", 0) or 0),
                    "release_date": str(rep.get("release_date", today.isoformat())),
                    "genre_hint": f"{rep.get('genre_hint', '')} {' '.join(tag_names[:4])}".strip(),
                    "views": views,
                    "likes": max(0, int(views * 0.025)),
                    "comments": max(0, int(views * 0.001)),
                    "shares": max(0, int(views * 0.0018)),
                    "streams": int(views * 0.2),
                    "listeners": int(views * 0.15),
                    "playlist_adds": 0,
                    "creator_reuse": 0,
                    "region_metrics": {},
                    "tastemaker_id": "musicbrainz_graph",
                    "tastemaker_name": "MusicBrainz Graph",
                    "event_type": "graph_enrichment",
                    "source": "musicbrainz",
                    "comments_text": [f"mb_confidence={confidence:.4f}", f"mb_rels={relation_count}", f"mb_tags={tag_count}"],
                    "collaborators": list(set(list(rep.get("collaborators", [])) + [f"mb_rel_{relation_count}"])),
                    "manual_seeded": bool(rep.get("manual_seeded", False)),
                    "metadata_text": f"musicbrainz confidence {confidence:.4f} tags {' '.join(tag_names)}".strip(),
                }
            )

    return rows, {
        "enabled": True,
        "artists_seen": len(ranked_artists),
        "artists_resolved": resolved,
        "rows": len(rows),
    }
