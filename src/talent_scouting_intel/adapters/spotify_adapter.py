from __future__ import annotations

import base64
import datetime as dt
import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    value = SLUG_RE.sub("-", text.lower()).strip("-")
    return value or "unknown"


def _request_json(url: str, *, headers: dict[str, str], data: bytes | None = None, timeout: int = 4) -> dict[str, Any]:
    req = urllib.request.Request(url, headers=headers, data=data)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = response.read().decode("utf-8", errors="ignore")
    obj = json.loads(payload)
    if isinstance(obj, dict):
        return obj
    return {}


def _spotify_token(client_id: str, client_secret: str) -> str | None:
    token_url = "https://accounts.spotify.com/api/token"
    auth = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("ascii")
    headers = {
        "Authorization": f"Basic {auth}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode("utf-8")
    try:
        payload = _request_json(token_url, headers=headers, data=data)
    except Exception:
        return None
    token = payload.get("access_token")
    return str(token) if token else None


def _spotify_get(path: str, token: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    query = urllib.parse.urlencode(params or {})
    url = f"https://api.spotify.com/v1/{path}"
    if query:
        url += f"?{query}"
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "tsi-bot/0.1",
    }
    return _request_json(url, headers=headers)


def _chunks(values: list[str], size: int) -> list[list[str]]:
    out: list[list[str]] = []
    for idx in range(0, len(values), size):
        out.append(values[idx : idx + size])
    return out


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


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


def _artist_map(artist_ids: list[str], token: str) -> dict[str, dict[str, Any]]:
    if not artist_ids:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for batch in _chunks(sorted(set(artist_ids)), 50):
        payload = _spotify_get("artists", token, params={"ids": ",".join(batch)})
        for artist in payload.get("artists", []):
            aid = str(artist.get("id", ""))
            if not aid:
                continue
            out[aid] = artist
    return out


def _parse_day(value: str, default: dt.date) -> dt.date:
    value = (value or "").strip()
    if not value:
        return default
    try:
        if len(value) == 4:
            return dt.date(int(value), 1, 1)
        if len(value) == 7:
            year, month = value.split("-")
            return dt.date(int(year), int(month), 1)
        return dt.date.fromisoformat(value)
    except Exception:
        return default


def _tracked_spotify_rows(
    cfg: dict[str, Any],
    token: str,
    *,
    today: dt.date,
    tracking_targets: dict[str, list[str]] | None,
    existing_track_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    targets = tracking_targets or {}
    requested = targets.get("spotify_track_ids", [])
    if not isinstance(requested, list) or not requested:
        return [], {"enabled": True, "reason": "no tracked spotify tracks", "requested": 0}

    max_tracks = int(cfg.get("tracking_max_tracks_per_run", 150))
    market = str(cfg.get("market", "US"))
    selected = _dedupe([str(track_id) for track_id in requested])[: max(1, max_tracks)]
    if not selected:
        return [], {"enabled": True, "reason": "no valid tracked spotify ids", "requested": 0}

    track_objs: list[dict[str, Any]] = []
    errors: list[str] = []
    for batch in _chunks(selected, 50):
        try:
            payload = _spotify_get("tracks", token, params={"ids": ",".join(batch), "market": market})
        except Exception:
            errors.append(f"track batch failed ({len(batch)})")
            continue
        tracks = payload.get("tracks", [])
        if not isinstance(tracks, list):
            continue
        for track in tracks:
            if not isinstance(track, dict):
                continue
            track_id = str(track.get("id", "")).strip()
            if not track_id or f"sp_{track_id}" in existing_track_ids:
                continue
            track_objs.append(track)

    if not track_objs:
        return [], {
            "enabled": True,
            "requested": len(selected),
            "resolved": 0,
            "rows": 0,
            "errors": errors[:10],
        }

    artist_ids = [
        str(((track.get("artists") or [{}])[0].get("id", "")).strip())
        for track in track_objs
        if isinstance(track.get("artists"), list) and track.get("artists")
    ]
    artist_map = _artist_map([artist_id for artist_id in artist_ids if artist_id], token)

    rows: list[dict[str, Any]] = []
    for track in track_objs:
        track_id = str(track.get("id", "")).strip()
        if not track_id:
            continue
        artists_data = track.get("artists", [])
        primary_artist = artists_data[0] if isinstance(artists_data, list) and artists_data else {}
        artist_id = str(primary_artist.get("id", "")).strip() or f"sp_artist_{_slug(str(primary_artist.get('name', 'unknown')))}"
        artist_name = str(primary_artist.get("name", "Unknown Artist"))

        artist_meta = artist_map.get(artist_id, {})
        artist_followers = int(((artist_meta.get("followers") or {}).get("total", 0)) or 0)
        artist_genres = " ".join(str(value) for value in artist_meta.get("genres", []))

        popularity = int(track.get("popularity", 0) or 0)
        release_date = _parse_day(str((track.get("album") or {}).get("release_date", "")), today)
        recency_boost_days = max(0, 45 - (today - release_date).days)
        follower_factor = 0.85 + min(0.45, math.log10(artist_followers + 10.0) / 10.0)
        proxy_base = max(90, popularity * 110)
        proxy_streams = int(proxy_base * (1.0 + recency_boost_days / 150.0) * follower_factor)

        track_name = str(track.get("name", "Unknown Track"))
        metadata_text = " ".join(
            [
                track_name,
                str((track.get("album") or {}).get("name", "")),
                artist_name,
                artist_genres,
                "tracked follow",
            ]
        ).strip()

        rows.append(
            {
                "date": today.isoformat(),
                "platform": "spotify",
                "track_id": f"sp_{track_id}",
                "track_name": track_name,
                "artist_id": f"sp_{artist_id}",
                "artist_name": artist_name,
                "artist_followers": artist_followers,
                "release_date": release_date.isoformat(),
                "genre_hint": artist_genres,
                "views": int(proxy_streams * 1.04),
                "likes": int(proxy_streams * 0.06),
                "comments": int(proxy_streams * 0.0018),
                "shares": int(proxy_streams * 0.005),
                "streams": proxy_streams,
                "listeners": int(proxy_streams * 0.58),
                "playlist_adds": 0,
                "creator_reuse": 0,
                "region_metrics": {market: max(1, int(proxy_streams * 0.8))} if market else {},
                "tastemaker_id": "sp_tracking_pool",
                "tastemaker_name": "Tracking Pool",
                "event_type": "track_follow",
                "source": "spotify_track_follow",
                "comments_text": [],
                "collaborators": [],
                "manual_seeded": False,
                "metadata_text": metadata_text,
            }
        )

    return rows, {
        "enabled": True,
        "requested": len(selected),
        "resolved": len(track_objs),
        "rows": len(rows),
        "errors": errors[:10],
    }


def collect_spotify_snapshots(
    config: dict[str, Any],
    registry: dict[str, Any],
    *,
    today: dt.date,
    seed_urls: list[str] | None = None,
    tracking_targets: dict[str, list[str]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = config.get("ingest", {}).get("auto", {}).get("spotify", {})
    if not bool(cfg.get("enabled", True)):
        return [], {"enabled": False, "reason": "spotify adapter disabled"}

    playlists = registry.get("spotify_playlists", []) if isinstance(registry, dict) else []
    if not isinstance(playlists, list) or not playlists:
        return [], {"enabled": True, "reason": "no spotify playlists configured", "playlists": 0}

    client_id = os.getenv(str(cfg.get("client_id_env", "SPOTIFY_CLIENT_ID")), "")
    client_secret = os.getenv(str(cfg.get("client_secret_env", "SPOTIFY_CLIENT_SECRET")), "")
    if not client_id or not client_secret:
        return [], {
            "enabled": True,
            "reason": "missing Spotify credentials",
            "required_env": [str(cfg.get("client_id_env", "SPOTIFY_CLIENT_ID")), str(cfg.get("client_secret_env", "SPOTIFY_CLIENT_SECRET"))],
            "playlists": len(playlists),
        }

    token = _spotify_token(client_id, client_secret)
    if not token:
        return [], {"enabled": True, "reason": "token fetch failed", "playlists": len(playlists)}

    max_tracks = int(cfg.get("max_tracks_per_playlist", 80))
    seed_tokens = _extract_seed_tokens(seed_urls or [])

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    playlists_ok = 0

    for playlist in playlists:
        if not isinstance(playlist, dict):
            continue
        playlist_id = str(playlist.get("playlist_id", "")).strip()
        if not playlist_id:
            continue

        try:
            payload = _spotify_get(
                f"playlists/{playlist_id}/tracks",
                token,
                params={"limit": max_tracks, "market": str(cfg.get("market", "US")), "fields": "items(added_at,track(id,name,popularity,duration_ms,artists(id,name),album(release_date,name))),next"},
            )
        except urllib.error.HTTPError as exc:
            errors.append(f"playlist {playlist_id}: http {exc.code}")
            continue
        except Exception:
            errors.append(f"playlist {playlist_id}: request failed")
            continue

        items = payload.get("items", [])
        if not isinstance(items, list):
            continue
        playlists_ok += 1

        artist_ids: list[str] = []
        for item in items:
            track = item.get("track") or {}
            for artist in track.get("artists", []):
                artist_id = str(artist.get("id", "")).strip()
                if artist_id:
                    artist_ids.append(artist_id)
        artists = _artist_map(artist_ids, token)

        playlist_name = str(playlist.get("name", playlist_id))
        genre_hint = " ".join(str(tag) for tag in playlist.get("genre_tags", []))
        region = str(playlist.get("region", "") or "")

        for item in items:
            track = item.get("track") or {}
            track_id = str(track.get("id", "")).strip()
            if not track_id:
                continue

            artists_data = track.get("artists", [])
            primary_artist = artists_data[0] if artists_data else {}
            artist_id = str(primary_artist.get("id", "")).strip() or f"sp_artist_{_slug(str(primary_artist.get('name', 'unknown')))}"
            artist_name = str(primary_artist.get("name", "Unknown Artist"))

            artist_meta = artists.get(artist_id, {})
            artist_followers = int(((artist_meta.get("followers") or {}).get("total", 0)) or 0)
            artist_genres = " ".join(str(g) for g in artist_meta.get("genres", []))

            popularity = int(track.get("popularity", 0) or 0)
            proxy_base = max(120, popularity * 120)
            release_date = _parse_day(str((track.get("album") or {}).get("release_date", "")), today)
            recency_boost_days = max(0, 45 - (today - release_date).days)
            proxy_streams = int(proxy_base * (1.0 + recency_boost_days / 120.0))

            track_name = str(track.get("name", "Unknown Track"))
            metadata_text = " ".join(
                [
                    track_name,
                    str((track.get("album") or {}).get("name", "")),
                    artist_name,
                    artist_genres,
                    playlist_name,
                    genre_hint,
                ]
            ).strip()

            region_metrics = {}
            if region:
                region_metrics[region] = max(1, int(proxy_streams * 0.8))

            rows.append(
                {
                    "date": today.isoformat(),
                    "platform": "spotify",
                    "track_id": f"sp_{track_id}",
                    "track_name": track_name,
                    "artist_id": f"sp_{artist_id}",
                    "artist_name": artist_name,
                    "artist_followers": artist_followers,
                    "release_date": release_date.isoformat(),
                    "genre_hint": f"{genre_hint} {artist_genres}".strip(),
                    "views": int(proxy_streams * 1.04),
                    "likes": int(proxy_streams * 0.06),
                    "comments": int(proxy_streams * 0.0018),
                    "shares": int(proxy_streams * 0.005),
                    "streams": proxy_streams,
                    "listeners": int(proxy_streams * 0.58),
                    "playlist_adds": 1,
                    "creator_reuse": 0,
                    "region_metrics": region_metrics,
                    "tastemaker_id": f"sp_pl_{playlist_id}",
                    "tastemaker_name": playlist_name,
                    "event_type": "playlist_add",
                    "source": "spotify_api",
                    "comments_text": [],
                    "collaborators": [],
                    "manual_seeded": _is_seeded(seed_tokens, track_id, track_name, artist_name, playlist_name),
                    "metadata_text": metadata_text,
                }
            )

    tracked_rows, tracked_meta = _tracked_spotify_rows(
        cfg,
        token,
        today=today,
        tracking_targets=tracking_targets,
        existing_track_ids={str(row.get("track_id", "")) for row in rows if str(row.get("platform", "")) == "spotify"},
    )
    rows.extend(tracked_rows)

    meta = {
        "enabled": True,
        "playlists": len(playlists),
        "playlists_ok": playlists_ok,
        "rows": len(rows),
        "errors": errors[:10],
        "tracking_follow": tracked_meta,
    }
    return rows, meta
