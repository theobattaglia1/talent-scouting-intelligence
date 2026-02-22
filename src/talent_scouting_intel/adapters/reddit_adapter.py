from __future__ import annotations

import datetime as dt
import json
import re
import urllib.parse
import urllib.request
from typing import Any

SPOTIFY_TRACK_RE = re.compile(r"open\.spotify\.com/track/([A-Za-z0-9]+)")
YOUTUBE_RE = re.compile(r"(?:youtu\.be/|youtube\.com/watch\?v=)([A-Za-z0-9_-]{6,})")


def _request_json(url: str, user_agent: str, timeout: int = 4) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
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


def _extract_cross_platform_id(url: str, fallback_id: str) -> tuple[str, str]:
    spotify = SPOTIFY_TRACK_RE.search(url)
    if spotify:
        return "sp", spotify.group(1)
    youtube = YOUTUBE_RE.search(url)
    if youtube:
        return "yt", youtube.group(1)
    return "rd", fallback_id


def collect_reddit_snapshots(
    config: dict[str, Any],
    registry: dict[str, Any],
    *,
    today: dt.date,
    seed_urls: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = config.get("ingest", {}).get("auto", {}).get("reddit", {})
    if not bool(cfg.get("enabled", True)):
        return [], {"enabled": False, "reason": "reddit adapter disabled"}

    subreddits = registry.get("reddit_subreddits", []) if isinstance(registry, dict) else []
    if not isinstance(subreddits, list) or not subreddits:
        return [], {"enabled": True, "reason": "no reddit subreddits configured", "subreddits": 0}

    user_agent = str(cfg.get("user_agent", "tsi-bot/0.1"))
    max_posts = int(cfg.get("max_posts_per_subreddit", 80))
    max_age_days = int(cfg.get("max_age_days", 14))

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    seed_tokens = _extract_seed_tokens(seed_urls or [])

    for sub in subreddits:
        if not isinstance(sub, dict):
            continue
        name = str(sub.get("name", "")).strip()
        if not name:
            continue

        params = urllib.parse.urlencode({"limit": max_posts, "raw_json": 1})
        url = f"https://www.reddit.com/r/{urllib.parse.quote(name)}/new.json?{params}"
        try:
            payload = _request_json(url, user_agent)
        except Exception:
            errors.append(f"r/{name}: request failed")
            continue

        posts = (((payload.get("data") or {}).get("children") or []))
        if not isinstance(posts, list):
            continue

        genre_hint = " ".join(str(tag) for tag in sub.get("genre_tags", []))
        for wrapper in posts:
            data = wrapper.get("data") or {}
            created = float(data.get("created_utc", 0.0) or 0.0)
            if created <= 0:
                continue
            day = dt.datetime.utcfromtimestamp(created).date()
            if (today - day).days > max_age_days:
                continue

            title = str(data.get("title", "")).strip()
            if not title:
                continue

            permalink = str(data.get("permalink", ""))
            external_url = str(data.get("url", ""))
            post_id = str(data.get("id", "")).strip() or f"post_{int(created)}"
            prefix, external_id = _extract_cross_platform_id(external_url, post_id)

            if prefix == "sp":
                track_id = f"sp_{external_id}"
            elif prefix == "yt":
                track_id = f"yt_{external_id}"
            else:
                track_id = f"rd_{post_id}"

            score = int(data.get("score", 0) or 0)
            comments = int(data.get("num_comments", 0) or 0)
            upvote_ratio = float(data.get("upvote_ratio", 0.9) or 0.9)
            proxy_streams = max(0, int(score * (6.0 + upvote_ratio * 4.0)))

            selftext = str(data.get("selftext", ""))
            comments_text = [title]
            if selftext:
                comments_text.append(selftext[:240])

            rows.append(
                {
                    "date": day.isoformat(),
                    "platform": "reddit",
                    "track_id": track_id,
                    "track_name": title[:120],
                    "artist_id": f"rd_author_{str(data.get('author', 'unknown'))}",
                    "artist_name": str(data.get("author", "unknown")),
                    "artist_followers": int(data.get("subreddit_subscribers", 0) or 0),
                    "release_date": day.isoformat(),
                    "genre_hint": genre_hint,
                    "views": max(1, int(score * 10)),
                    "likes": max(0, score),
                    "comments": max(0, comments),
                    "shares": max(1, int(score * 0.08)),
                    "streams": proxy_streams,
                    "listeners": int(proxy_streams * 0.55),
                    "playlist_adds": 0,
                    "creator_reuse": 0,
                    "region_metrics": {},
                    "tastemaker_id": f"rd_{name}",
                    "tastemaker_name": f"r/{name}",
                    "event_type": "mention",
                    "source": "reddit_public",
                    "comments_text": comments_text,
                    "collaborators": [],
                    "manual_seeded": _is_seeded(seed_tokens, title, external_url, permalink),
                    "metadata_text": f"{title} {selftext[:180]} {genre_hint}".strip(),
                }
            )

    return rows, {
        "enabled": True,
        "subreddits": len(subreddits),
        "rows": len(rows),
        "errors": errors[:10],
    }
