from __future__ import annotations

import datetime as dt
import json
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from talent_scouting_intel.adapters.lastfm_adapter import collect_lastfm_snapshots
from talent_scouting_intel.adapters.mock_adapter import generate_mock_snapshots
from talent_scouting_intel.adapters.musicbrainz_adapter import collect_musicbrainz_snapshots
from talent_scouting_intel.adapters.reddit_adapter import collect_reddit_snapshots
from talent_scouting_intel.adapters.rss_adapter import collect_rss_snapshots
from talent_scouting_intel.adapters.shortform_proxy_adapter import synthesize_shortform_proxy_rows
from talent_scouting_intel.adapters.spotify_adapter import collect_spotify_snapshots
from talent_scouting_intel.adapters.wikipedia_adapter import collect_wikipedia_snapshots
from talent_scouting_intel.utils.io import resolve_path

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "yt": "http://www.youtube.com/xml/schemas/2015"}


def _chunks(values: list[str], size: int) -> list[list[str]]:
    out: list[list[str]] = []
    for idx in range(0, len(values), size):
        out.append(values[idx : idx + size])
    return out


def _request_text(url: str, timeout: int = 4) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "tsi-bot/0.1 (+autonomous scouting)"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = response.read()
    return payload.decode("utf-8", errors="ignore")


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


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


def _youtube_stats(video_ids: list[str], api_key: str) -> dict[str, dict[str, int]]:
    if not video_ids:
        return {}
    query = urllib.parse.urlencode(
        {
            "part": "statistics",
            "id": ",".join(video_ids),
            "key": api_key,
            "maxResults": len(video_ids),
        }
    )
    url = f"https://www.googleapis.com/youtube/v3/videos?{query}"
    try:
        payload = json.loads(_request_text(url))
    except Exception:
        return {}

    out: dict[str, dict[str, int]] = {}
    for item in payload.get("items", []):
        video_id = str(item.get("id", ""))
        stats = item.get("statistics", {})
        if not video_id:
            continue
        out[video_id] = {
            "views": int(stats.get("viewCount", 0) or 0),
            "likes": int(stats.get("likeCount", 0) or 0),
            "comments": int(stats.get("commentCount", 0) or 0),
        }
    return out


def _youtube_videos(video_ids: list[str], api_key: str) -> list[dict[str, Any]]:
    if not video_ids:
        return []

    rows: list[dict[str, Any]] = []
    for batch in _chunks(video_ids, 50):
        query = urllib.parse.urlencode(
            {
                "part": "snippet,statistics",
                "id": ",".join(batch),
                "key": api_key,
                "maxResults": len(batch),
            }
        )
        url = f"https://www.googleapis.com/youtube/v3/videos?{query}"
        try:
            payload = json.loads(_request_text(url))
        except Exception:
            continue
        items = payload.get("items", [])
        if isinstance(items, list):
            rows.extend([item for item in items if isinstance(item, dict)])
    return rows


def _youtube_comment_threads(video_id: str, api_key: str, max_results: int) -> list[str]:
    if not video_id or not api_key or max_results <= 0:
        return []

    query = urllib.parse.urlencode(
        {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": min(100, max_results),
            "order": "relevance",
            "textFormat": "plainText",
            "key": api_key,
        }
    )
    url = f"https://www.googleapis.com/youtube/v3/commentThreads?{query}"
    try:
        payload = json.loads(_request_text(url))
    except Exception:
        return []

    out: list[str] = []
    for item in payload.get("items", []):
        top = ((item.get("snippet") or {}).get("topLevelComment") or {}).get("snippet", {})
        text = str(top.get("textDisplay") or top.get("textOriginal") or "").strip()
        if not text:
            continue
        out.append(text[:280])
    return out


def _parse_day(value: str) -> dt.date:
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(value)
    return parsed.date()


def _youtube_rss_rows(
    channel: dict[str, Any],
    config: dict[str, Any],
    *,
    seed_tokens: list[str],
    today: dt.date,
) -> list[dict[str, Any]]:
    channel_id = str(channel.get("channel_id", "")).strip()
    if not channel_id:
        return []

    yt_cfg = config.get("ingest", {}).get("auto", {}).get("youtube", {})
    max_entries = int(yt_cfg.get("max_recent_entries_per_channel", 12))
    max_age_days = int(yt_cfg.get("max_age_days", 45))
    fallback_view_floor = int(yt_cfg.get("fallback_view_floor", 300))
    enrich_comment_threads = bool(yt_cfg.get("enrich_comment_threads", True))
    comment_threads_per_video = int(yt_cfg.get("comment_threads_per_video", 12))
    comment_enrichment_video_cap = int(yt_cfg.get("comment_enrichment_video_cap", 18))

    url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    try:
        xml_text = _request_text(url)
    except Exception:
        return []

    root = ET.fromstring(xml_text)
    entries = root.findall("atom:entry", ATOM_NS)
    entries = entries[:max_entries]

    api_key_env = str(yt_cfg.get("api_key_env", "YOUTUBE_API_KEY"))
    use_stats = bool(yt_cfg.get("use_youtube_data_api_stats", True))
    api_key = os.getenv(api_key_env, "") if use_stats else ""
    video_ids = [entry.findtext("yt:videoId", default="", namespaces=ATOM_NS) for entry in entries]
    stats_by_video = _youtube_stats([vid for vid in video_ids if vid], api_key) if api_key else {}
    comments_by_video: dict[str, list[str]] = {}
    if api_key and enrich_comment_threads and comment_threads_per_video > 0:
        for vid in [item for item in video_ids if item][:comment_enrichment_video_cap]:
            comments_by_video[vid] = _youtube_comment_threads(vid, api_key, comment_threads_per_video)

    genre_tags = channel.get("genre_tags", [])
    genre_hint = " ".join(str(tag) for tag in genre_tags)
    name = str(channel.get("name", channel_id))

    rows: list[dict[str, Any]] = []
    for entry in entries:
        video_id = entry.findtext("yt:videoId", default="", namespaces=ATOM_NS)
        published = entry.findtext("atom:published", default="", namespaces=ATOM_NS)
        title = entry.findtext("atom:title", default="", namespaces=ATOM_NS)
        if not video_id or not published:
            continue

        day = _parse_day(published)
        age_days = (today - day).days
        if age_days < 0 or age_days > max_age_days:
            continue

        stats = stats_by_video.get(video_id, {})
        comments_text = comments_by_video.get(video_id, [])
        fallback_views = max(fallback_view_floor, fallback_view_floor + (max_age_days - age_days) * 40)
        views = int(stats.get("views", fallback_views))
        likes = int(stats.get("likes", max(10, int(views * 0.05))))
        comments = int(stats.get("comments", max(2, int(views * 0.004))))
        if comments_text:
            comments = max(comments, len(comments_text))
        shares = max(1, int(views * 0.003))
        streams = max(0, int(views * 0.35))
        listeners = max(0, int(streams * 0.62))

        rows.append(
            {
                "date": day.isoformat(),
                "platform": "youtube",
                "track_id": f"yt_{video_id}",
                "track_name": title,
                "artist_id": f"ytc_{channel_id}",
                "artist_name": name,
                "artist_followers": int(channel.get("estimated_followers", 0) or 0),
                "release_date": day.isoformat(),
                "genre_hint": genre_hint,
                "views": views,
                "likes": likes,
                "comments": comments,
                "shares": shares,
                "streams": streams,
                "listeners": listeners,
                "playlist_adds": 0,
                "creator_reuse": 0,
                "region_metrics": {},
                "tastemaker_id": f"ytc_{channel_id}",
                "tastemaker_name": name,
                "event_type": "upload",
                "source": "youtube_rss",
                "comments_text": comments_text,
                "collaborators": [],
                "manual_seeded": _is_seeded(seed_tokens, video_id, title, name),
                "metadata_text": f"{title} {genre_hint}",
            }
        )
    return rows


def _youtube_follow_rows(
    config: dict[str, Any],
    tracking_targets: dict[str, list[str]] | None,
    *,
    today: dt.date,
    existing_track_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    yt_cfg = config.get("ingest", {}).get("auto", {}).get("youtube", {})
    if not bool(yt_cfg.get("enabled", True)):
        return [], {"enabled": False, "reason": "youtube adapter disabled"}

    targets = tracking_targets or {}
    requested_ids = targets.get("youtube_video_ids", [])
    if not isinstance(requested_ids, list) or not requested_ids:
        return [], {"enabled": True, "reason": "no tracked youtube videos", "requested": 0}

    deduped: list[str] = []
    seen: set[str] = set()
    for raw in requested_ids:
        video_id = str(raw).strip()
        if not video_id or video_id in seen:
            continue
        seen.add(video_id)
        deduped.append(video_id)

    max_videos = int(yt_cfg.get("tracking_max_videos_per_run", 90))
    selected = deduped[: max(1, max_videos)]
    if not selected:
        return [], {"enabled": True, "reason": "no valid tracked youtube ids", "requested": 0}

    api_key_env = str(yt_cfg.get("api_key_env", "YOUTUBE_API_KEY"))
    api_key = os.getenv(api_key_env, "") if bool(yt_cfg.get("use_youtube_data_api_stats", True)) else ""
    if not api_key:
        return [], {
            "enabled": True,
            "reason": "missing YouTube API key for tracked-video polling",
            "required_env": [api_key_env],
            "requested": len(selected),
        }

    items = _youtube_videos(selected, api_key)
    comment_threads_per_video = int(yt_cfg.get("tracking_comment_threads_per_video", 6))
    comment_video_cap = int(yt_cfg.get("tracking_comment_video_cap", 30))
    rows: list[dict[str, Any]] = []
    skipped_existing = 0

    for item in items:
        video_id = str(item.get("id", "")).strip()
        if not video_id:
            continue
        track_id = f"yt_{video_id}"
        if track_id in existing_track_ids:
            skipped_existing += 1
            continue

        snippet = item.get("snippet", {}) or {}
        stats = item.get("statistics", {}) or {}
        title = str(snippet.get("title", "")).strip()
        channel_id = str(snippet.get("channelId", "")).strip()
        channel_title = str(snippet.get("channelTitle", "")).strip() or "Tracked Channel"
        published = str(snippet.get("publishedAt", "")).strip()
        day = _parse_day(published) if published else today

        tags = snippet.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        genre_hint = " ".join(str(tag) for tag in tags[:10])
        comments_text: list[str] = []
        if comment_threads_per_video > 0 and len(rows) < comment_video_cap:
            comments_text = _youtube_comment_threads(video_id, api_key, comment_threads_per_video)

        views = int(stats.get("viewCount", 0) or 0)
        likes = int(stats.get("likeCount", 0) or 0)
        comments = int(stats.get("commentCount", 0) or 0)
        if comments_text:
            comments = max(comments, len(comments_text))
        shares = max(1, int(views * 0.003))
        streams = max(0, int(views * 0.35))
        listeners = max(0, int(streams * 0.62))

        rows.append(
            {
                "date": today.isoformat(),
                "platform": "youtube",
                "track_id": track_id,
                "track_name": title or video_id,
                "artist_id": f"ytc_{channel_id}" if channel_id else "ytc_tracked_pool",
                "artist_name": channel_title,
                "artist_followers": 0,
                "release_date": day.isoformat(),
                "genre_hint": genre_hint,
                "views": views,
                "likes": likes,
                "comments": comments,
                "shares": shares,
                "streams": streams,
                "listeners": listeners,
                "playlist_adds": 0,
                "creator_reuse": 0,
                "region_metrics": {},
                "tastemaker_id": "yt_tracked_pool",
                "tastemaker_name": "Tracking Pool",
                "event_type": "tracked_video",
                "source": "youtube_video_follow",
                "comments_text": comments_text,
                "collaborators": [],
                "manual_seeded": False,
                "metadata_text": f"{title} {channel_title} tracked follow {genre_hint}".strip(),
            }
        )

    return rows, {
        "enabled": True,
        "requested": len(selected),
        "resolved_items": len(items),
        "rows": len(rows),
        "skipped_existing": skipped_existing,
    }


def collect_autonomous_snapshots(
    config: dict[str, Any],
    *,
    project_root: Path,
    seed_urls: list[str] | None = None,
    tracking_targets: dict[str, list[str]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed_tokens = _extract_seed_tokens(seed_urls or [])
    ingest_cfg = config.get("ingest", {}).get("auto", {})
    registry_path = resolve_path(str(ingest_cfg.get("source_registry", "data/sources/source_registry.json")), project_root)
    registry = _load_registry(registry_path)
    tracking_targets = tracking_targets or {}

    rows: list[dict[str, Any]] = []
    today = dt.date.today()
    source_stats: dict[str, Any] = {}

    youtube_enabled = bool(ingest_cfg.get("youtube", {}).get("enabled", True))
    youtube_channels = registry.get("youtube_channels", []) if isinstance(registry, dict) else []
    youtube_before = len(rows)
    if youtube_enabled and isinstance(youtube_channels, list):
        for channel in youtube_channels:
            if isinstance(channel, dict):
                rows.extend(_youtube_rss_rows(channel, config, seed_tokens=seed_tokens, today=today))
    source_stats["youtube_rss"] = {
        "enabled": youtube_enabled,
        "channels": len(youtube_channels) if isinstance(youtube_channels, list) else 0,
        "rows": len(rows) - youtube_before,
    }

    youtube_follow_before = len(rows)
    youtube_follow_rows, youtube_follow_meta = _youtube_follow_rows(
        config,
        tracking_targets,
        today=today,
        existing_track_ids={str(row.get("track_id", "")) for row in rows if str(row.get("platform", "")) == "youtube"},
    )
    rows.extend(youtube_follow_rows)
    source_stats["youtube_follow"] = dict(youtube_follow_meta)
    source_stats["youtube_follow"]["rows"] = len(rows) - youtube_follow_before

    spotify_rows, spotify_meta = collect_spotify_snapshots(
        config,
        registry,
        today=today,
        seed_urls=seed_urls,
        tracking_targets=tracking_targets,
    )
    rows.extend(spotify_rows)
    source_stats["spotify_api"] = spotify_meta

    reddit_rows, reddit_meta = collect_reddit_snapshots(config, registry, today=today, seed_urls=seed_urls)
    rows.extend(reddit_rows)
    source_stats["reddit_public"] = reddit_meta

    lastfm_rows, lastfm_meta = collect_lastfm_snapshots(config, registry, today=today, seed_urls=seed_urls)
    rows.extend(lastfm_rows)
    source_stats["lastfm_api"] = lastfm_meta

    rss_rows, rss_meta = collect_rss_snapshots(config, registry, today=today, seed_urls=seed_urls)
    rows.extend(rss_rows)
    source_stats["rss"] = rss_meta

    # Short-form proxy is a ToS-safe workaround signal synthesizer from public mentions.
    short_rows, short_meta = synthesize_shortform_proxy_rows(rows, config, today=today)
    rows.extend(short_rows)
    source_stats["shortform_proxy"] = short_meta

    wiki_rows, wiki_meta = collect_wikipedia_snapshots(config, rows, today=today)
    rows.extend(wiki_rows)
    source_stats["wikipedia"] = wiki_meta

    mb_rows, mb_meta = collect_musicbrainz_snapshots(config, rows, today=today)
    rows.extend(mb_rows)
    source_stats["musicbrainz"] = mb_meta

    include_mock_bootstrap = bool(ingest_cfg.get("include_mock_bootstrap", True))
    mock_before = len(rows)
    if include_mock_bootstrap:
        mock_days = int(config.get("project", {}).get("mock_days", 112))
        rows.extend(generate_mock_snapshots(config, days=mock_days, seed_urls=seed_urls or []))
    source_stats["mock_bootstrap"] = {
        "enabled": include_mock_bootstrap,
        "rows": len(rows) - mock_before,
    }

    meta = {
        "registry_path": str(registry_path),
        "youtube_channels": len(youtube_channels) if isinstance(youtube_channels, list) else 0,
        "rows_collected": len(rows),
        "include_mock_bootstrap": include_mock_bootstrap,
        "sources": source_stats,
    }
    return rows, meta
