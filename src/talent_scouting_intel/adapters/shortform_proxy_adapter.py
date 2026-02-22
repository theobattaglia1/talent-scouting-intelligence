from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from typing import Any


def _contains_keywords(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def _engagement_rate(row: dict[str, Any]) -> float:
    views = float(row.get("views", 0.0) or 0.0)
    likes = float(row.get("likes", 0.0) or 0.0)
    comments = float(row.get("comments", 0.0) or 0.0)
    shares = float(row.get("shares", 0.0) or 0.0)
    return (likes + comments + shares) / (views + 1.0)


def synthesize_shortform_proxy_rows(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    today: dt.date,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = config.get("ingest", {}).get("auto", {}).get("shortform_proxy", {})
    if not bool(cfg.get("enabled", True)):
        return [], {"enabled": False, "reason": "shortform proxy disabled"}

    keywords = [str(item).lower() for item in cfg.get(
        "keywords",
        [
            "tiktok",
            "tik tok",
            "fyp",
            "for you page",
            "viral",
            "trending sound",
            "sound trend",
            "capcut",
            "reel",
            "shorts",
        ],
    )]
    source_weights = {
        key: float(value)
        for key, value in cfg.get(
            "source_weights",
            {
                "reddit": 1.0,
                "youtube": 0.8,
                "rss": 0.65,
                "instagram": 0.9,
                "spotify": 0.25,
                "lastfm": 0.4,
            },
        ).items()
    }
    min_strength = float(cfg.get("min_strength", 2.0))

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        try:
            day = dt.date.fromisoformat(str(row.get("date", "")))
        except Exception:
            continue
        if day > today:
            continue
        grouped[(str(row.get("date", "")), str(row.get("track_id", "")))].append(row)

    proxy_rows: list[dict[str, Any]] = []
    for (day_str, track_id), items in grouped.items():
        if not day_str or not track_id:
            continue

        strength = 0.0
        mention_count = 0
        source_seen: set[str] = set()
        representative: dict[str, Any] | None = None
        rep_views = -1.0

        for row in items:
            platform = str(row.get("platform", "")).lower()
            if platform in {"tiktok", "tiktok_proxy"}:
                continue

            base_text = " ".join(
                [
                    str(row.get("track_name", "")),
                    str(row.get("metadata_text", "")),
                    " ".join(str(entry) for entry in row.get("comments_text", [])),
                ]
            )
            mention = _contains_keywords(base_text, keywords)
            if not mention:
                continue

            source_seen.add(platform)
            mention_count += 1

            weight = source_weights.get(platform, 0.55)
            views = float(row.get("views", 0.0) or 0.0)
            engagement = _engagement_rate(row)
            strength += weight * (1.0 + math.log1p(max(0.0, views))) * (1.0 + 0.7 * engagement)

            if views > rep_views:
                rep_views = views
                representative = row

        if strength < min_strength or representative is None:
            continue

        proxy_views = max(1, int(strength * 65.0))
        proxy_likes = max(1, int(proxy_views * 0.075))
        proxy_comments = max(1, int(proxy_views * 0.0055))
        proxy_shares = max(1, int(proxy_views * 0.016))
        proxy_streams = max(0, int(proxy_views * 0.42))
        proxy_listeners = max(0, int(proxy_streams * 0.61))
        proxy_reuse = max(1, int(mention_count * 2 + len(source_seen)))

        proxy_rows.append(
            {
                "date": day_str,
                "platform": "tiktok_proxy",
                "track_id": track_id,
                "track_name": representative.get("track_name", ""),
                "artist_id": representative.get("artist_id", ""),
                "artist_name": representative.get("artist_name", ""),
                "artist_followers": int(representative.get("artist_followers", 0) or 0),
                "release_date": str(representative.get("release_date", day_str)),
                "genre_hint": str(representative.get("genre_hint", "")),
                "views": proxy_views,
                "likes": proxy_likes,
                "comments": proxy_comments,
                "shares": proxy_shares,
                "streams": proxy_streams,
                "listeners": proxy_listeners,
                "playlist_adds": 0,
                "creator_reuse": proxy_reuse,
                "region_metrics": representative.get("region_metrics", {}) if isinstance(representative.get("region_metrics"), dict) else {},
                "tastemaker_id": "proxy_shortform",
                "tastemaker_name": "Short-Form Proxy",
                "event_type": "proxy_signal",
                "source": "shortform_proxy",
                "comments_text": [f"shortform proxy mentions={mention_count} sources={len(source_seen)}"],
                "collaborators": list(representative.get("collaborators", [])),
                "manual_seeded": bool(representative.get("manual_seeded", False)),
                "metadata_text": f"shortform proxy signal keywords={mention_count}",
            }
        )

    return proxy_rows, {
        "enabled": True,
        "rows": len(proxy_rows),
        "keywords": len(keywords),
    }
