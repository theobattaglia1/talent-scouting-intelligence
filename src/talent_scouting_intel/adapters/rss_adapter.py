from __future__ import annotations

import datetime as dt
import email.utils
import re
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

SLUG_RE = re.compile(r"[^a-z0-9]+")
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _slug(text: str) -> str:
    value = SLUG_RE.sub("-", text.lower()).strip("-")
    return value or "unknown"


def _request_text(url: str, timeout: int = 4) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "tsi-bot/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = response.read()
    return payload.decode("utf-8", errors="ignore")


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


def _parse_date(text: str, default: dt.date) -> dt.date:
    value = (text or "").strip()
    if not value:
        return default
    try:
        if value.endswith("Z"):
            return dt.datetime.fromisoformat(value[:-1] + "+00:00").date()
        return dt.datetime.fromisoformat(value).date()
    except Exception:
        pass

    try:
        return email.utils.parsedate_to_datetime(value).date()
    except Exception:
        return default


def _title_to_artist_track(title: str, fallback_artist: str) -> tuple[str, str]:
    if " - " in title:
        left, right = title.split(" - ", 1)
        if left.strip() and right.strip():
            return left.strip(), right.strip()
    return fallback_artist, title.strip() or "Unknown Track"


def collect_rss_snapshots(
    config: dict[str, Any],
    registry: dict[str, Any],
    *,
    today: dt.date,
    seed_urls: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = config.get("ingest", {}).get("auto", {}).get("rss", {})
    if not bool(cfg.get("enabled", True)):
        return [], {"enabled": False, "reason": "rss adapter disabled"}

    feeds = registry.get("rss_feeds", []) if isinstance(registry, dict) else []
    if not isinstance(feeds, list) or not feeds:
        return [], {"enabled": True, "reason": "no rss feeds configured", "feeds": 0}

    max_entries = int(cfg.get("max_entries_per_feed", 30))
    max_age_days = int(cfg.get("max_age_days", 21))
    seed_tokens = _extract_seed_tokens(seed_urls or [])

    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for feed in feeds:
        if not isinstance(feed, dict):
            continue
        url = str(feed.get("url", "")).strip()
        if not url:
            continue

        try:
            xml_text = _request_text(url)
            root = ET.fromstring(xml_text)
        except Exception:
            errors.append(f"feed {url}: request or parse failed")
            continue

        name = str(feed.get("name", url))
        feed_id = str(feed.get("id", _slug(name)))
        genre_hint = " ".join(str(tag) for tag in feed.get("genre_tags", []))

        items = root.findall("./channel/item")
        atom_entries = root.findall("atom:entry", ATOM_NS)

        candidates: list[dict[str, str]] = []
        if items:
            for item in items[:max_entries]:
                candidates.append(
                    {
                        "title": item.findtext("title", default="").strip(),
                        "link": item.findtext("link", default="").strip(),
                        "date": item.findtext("pubDate", default="").strip(),
                    }
                )
        elif atom_entries:
            for entry in atom_entries[:max_entries]:
                link_el = entry.find("atom:link", ATOM_NS)
                link = ""
                if link_el is not None:
                    link = str(link_el.attrib.get("href", "")).strip()
                candidates.append(
                    {
                        "title": entry.findtext("atom:title", default="", namespaces=ATOM_NS).strip(),
                        "link": link,
                        "date": entry.findtext("atom:updated", default="", namespaces=ATOM_NS).strip(),
                    }
                )

        for entry in candidates:
            title = entry["title"]
            if not title:
                continue

            day = _parse_date(entry["date"], today)
            if (today - day).days > max_age_days:
                continue

            artist_name, track_name = _title_to_artist_track(title, name)
            recency = max(1, max_age_days - (today - day).days)
            proxy_streams = int(140 + recency * 26)
            proxy_views = int(proxy_streams * 1.35)
            track_id = f"rss_{feed_id}_{_slug(track_name)}_{_slug(artist_name)}"

            rows.append(
                {
                    "date": day.isoformat(),
                    "platform": "rss",
                    "track_id": track_id,
                    "track_name": track_name,
                    "artist_id": f"rss_artist_{_slug(artist_name)}",
                    "artist_name": artist_name,
                    "artist_followers": 0,
                    "release_date": day.isoformat(),
                    "genre_hint": genre_hint,
                    "views": proxy_views,
                    "likes": int(proxy_views * 0.03),
                    "comments": int(proxy_views * 0.001),
                    "shares": int(proxy_views * 0.0025),
                    "streams": proxy_streams,
                    "listeners": int(proxy_streams * 0.54),
                    "playlist_adds": 0,
                    "creator_reuse": 0,
                    "region_metrics": {},
                    "tastemaker_id": f"rss_{feed_id}",
                    "tastemaker_name": name,
                    "event_type": "feature",
                    "source": "rss",
                    "comments_text": [title],
                    "collaborators": [],
                    "manual_seeded": _is_seeded(seed_tokens, title, entry.get("link", ""), name),
                    "metadata_text": f"{title} {genre_hint}".strip(),
                }
            )

    return rows, {
        "enabled": True,
        "feeds": len(feeds),
        "rows": len(rows),
        "errors": errors[:10],
    }
