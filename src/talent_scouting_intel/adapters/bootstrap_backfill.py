from __future__ import annotations

import csv
import datetime as dt
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from talent_scouting_intel.utils.io import read_csv, resolve_path

SPOTIFY_TRACK_RE = re.compile(r"spotify\.com/track/([A-Za-z0-9]+)")
DATE_RE = re.compile(r"(20\d{2}-\d{2}-\d{2})")
SLUG_RE = re.compile(r"[^a-z0-9]+")


def _normalize_key(text: str) -> str:
    return SLUG_RE.sub("-", str(text).strip().lower()).strip("-") or "unknown"


def _parse_date(value: str) -> dt.date | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1]
    if "T" in raw:
        raw = raw.split("T", 1)[0]
    try:
        return dt.date.fromisoformat(raw)
    except Exception:
        return None


def _request_json(url: str, *, timeout: int = 4) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "tsi-bootstrap/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = response.read().decode("utf-8", errors="ignore")
    obj = json.loads(payload)
    return obj if isinstance(obj, dict) else {}


def _row_value(row: dict[str, Any], *keys: str) -> str:
    lowered = {str(key).strip().lower(): str(value) for key, value in row.items()}
    for key in keys:
        value = lowered.get(key.lower(), "")
        if str(value).strip():
            return str(value).strip()
    return ""


def _load_dataset_manifest(path: Path) -> set[str]:
    if not path.exists():
        return set()
    rows = read_csv(path)
    out: set[str] = set()
    for row in rows:
        name = str(row.get("dataset", "")).strip().lower()
        if name:
            out.add(name)
    return out


def _spotify_chart_rows(
    charts_dir: Path,
    *,
    today: dt.date,
    max_files_per_run: int,
    lookback_days: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not charts_dir.exists():
        return [], {"enabled": True, "rows": 0, "reason": "spotify charts directory not found"}

    files = sorted([path for path in charts_dir.glob("*.csv") if path.is_file()])
    if not files:
        return [], {"enabled": True, "rows": 0, "reason": "no spotify chart csv files found"}

    files = files[-max(1, max_files_per_run) :]
    rows_out: list[dict[str, Any]] = []
    files_used = 0
    skipped_out_of_range = 0

    for file_path in files:
        with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                continue
            files_used += 1
            file_date_match = DATE_RE.search(file_path.name)
            file_date = _parse_date(file_date_match.group(1)) if file_date_match else None
            file_region = "global"
            for token in file_path.stem.split("-"):
                tok = token.strip().lower()
                if tok in {"us", "uk", "au", "se", "global"}:
                    file_region = tok.upper() if tok != "global" else "global"

            for row in reader:
                date_value = _row_value(row, "date", "day", "chart_date")
                day = _parse_date(date_value) or file_date
                if not day:
                    continue
                if lookback_days > 0 and (today - day).days > lookback_days:
                    skipped_out_of_range += 1
                    continue

                track_name = _row_value(row, "track_name", "track", "title", "song", "name")
                artist_name = _row_value(row, "artist_name", "artist", "artists")
                if not track_name or not artist_name:
                    continue

                url = _row_value(row, "url", "track_url", "spotify_url")
                match = SPOTIFY_TRACK_RE.search(url)
                track_suffix = match.group(1) if match else _normalize_key(f"{artist_name}-{track_name}")
                streams_raw = _row_value(row, "streams", "plays")
                rank_raw = _row_value(row, "rank", "position")
                try:
                    streams = int(float(streams_raw.replace(",", ""))) if streams_raw else 0
                except Exception:
                    streams = 0
                try:
                    rank = int(float(rank_raw)) if rank_raw else 0
                except Exception:
                    rank = 0

                if streams <= 0:
                    # Rank-based fallback if stream value absent in archive.
                    streams = max(1, 500000 - (rank * 2500) if rank > 0 else 10000)

                artist_suffix = _normalize_key(artist_name)
                metadata_text = " ".join([track_name, artist_name, "spotify charts backfill"]).strip()
                rows_out.append(
                    {
                        "date": day.isoformat(),
                        "platform": "spotify",
                        "track_id": f"sp_{track_suffix}",
                        "track_name": track_name,
                        "artist_id": f"sp_{artist_suffix}",
                        "artist_name": artist_name,
                        "artist_followers": max(0, int(streams * 0.12)),
                        "release_date": day.isoformat(),
                        "genre_hint": "",
                        "views": int(streams * 1.04),
                        "likes": int(streams * 0.06),
                        "comments": int(streams * 0.0014),
                        "shares": int(streams * 0.004),
                        "streams": int(streams),
                        "listeners": int(streams * 0.58),
                        "playlist_adds": 0,
                        "creator_reuse": 0,
                        "region_metrics": {file_region: max(1, int(streams * 0.7))},
                        "tastemaker_id": f"sp_chart_{file_region}",
                        "tastemaker_name": f"Spotify Charts {file_region}",
                        "event_type": "chart_entry",
                        "source": "spotify_charts_backfill",
                        "comments_text": [f"rank={rank}"] if rank > 0 else [],
                        "collaborators": [],
                        "manual_seeded": False,
                        "metadata_text": metadata_text,
                    }
                )

    return rows_out, {
        "enabled": True,
        "rows": len(rows_out),
        "files_used": files_used,
        "skipped_out_of_range": skipped_out_of_range,
        "charts_dir": str(charts_dir),
    }


def _load_backfill_artists(config: dict[str, Any], project_root: Path, max_artists: int) -> list[str]:
    paths = config.get("paths", {})
    affinity_path = resolve_path(str(paths.get("affinity_artists", "")), project_root)
    breakout_path = resolve_path(str(paths.get("breakout_templates", "")), project_root)
    names: list[str] = []
    for path in [affinity_path, breakout_path]:
        if not path.exists():
            continue
        for row in read_csv(path):
            artist = str(row.get("artist_name", "")).strip()
            if artist:
                names.append(artist)

    deduped: list[str] = []
    seen: set[str] = set()
    for artist in names:
        key = artist.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(artist)
    return deduped[: max(1, max_artists)]


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
        req = urllib.request.Request(url, headers={"User-Agent": "tsi-bootstrap/0.1"})
        with urllib.request.urlopen(req, timeout=4) as response:
            payload = json.loads(response.read().decode("utf-8", errors="ignore"))
    except Exception:
        return None

    if isinstance(payload, list) and len(payload) >= 2 and isinstance(payload[1], list) and payload[1]:
        title = str(payload[1][0]).strip()
        return title if title else None
    return None


def _wikipedia_pageviews(title: str, start: dt.date, end: dt.date) -> list[dict[str, Any]]:
    safe = urllib.parse.quote(title.replace(" ", "_"), safe="")
    if not safe:
        return []
    url = (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"en.wikipedia.org/all-access/user/{safe}/daily/{start.strftime('%Y%m%d')}/{end.strftime('%Y%m%d')}"
    )
    try:
        payload = _request_json(url, timeout=5)
    except Exception:
        return []
    items = payload.get("items", [])
    return items if isinstance(items, list) else []


def _wikipedia_backfill_rows(
    config: dict[str, Any],
    project_root: Path,
    *,
    today: dt.date,
    lookback_days: int,
    max_artists: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    artist_names = _load_backfill_artists(config, project_root, max_artists)
    if not artist_names:
        return [], {"enabled": True, "rows": 0, "reason": "no artists available from affinity/breakout datasets"}

    start = today - dt.timedelta(days=max(7, lookback_days))
    end = today - dt.timedelta(days=1)

    rows: list[dict[str, Any]] = []
    resolved = 0
    errors = 0
    for artist in artist_names:
        title = _resolve_wikipedia_title(artist)
        if not title:
            errors += 1
            continue
        resolved += 1
        series = _wikipedia_pageviews(title, start, end)
        if not series:
            continue

        artist_slug = _normalize_key(artist)
        for item in series:
            timestamp = str(item.get("timestamp", "")).strip()
            day = _parse_date(timestamp[:8] if len(timestamp) >= 8 else "")
            if day is None:
                continue
            views = int(float(item.get("views", 0) or 0))
            if views <= 0:
                continue
            rows.append(
                {
                    "date": day.isoformat(),
                    "platform": "wikipedia",
                    "track_id": f"wiki_{artist_slug}",
                    "track_name": f"{artist} (wiki attention)",
                    "artist_id": f"wiki_{artist_slug}",
                    "artist_name": artist,
                    "artist_followers": max(50, int(views * 0.08)),
                    "release_date": day.isoformat(),
                    "genre_hint": "",
                    "views": views,
                    "likes": max(0, int(views * 0.01)),
                    "comments": max(0, int(views * 0.0008)),
                    "shares": max(0, int(views * 0.0018)),
                    "streams": max(1, int(views * 0.35)),
                    "listeners": max(1, int(views * 0.2)),
                    "playlist_adds": 0,
                    "creator_reuse": 0,
                    "region_metrics": {},
                    "tastemaker_id": "wiki_backfill",
                    "tastemaker_name": "Wikimedia Backfill",
                    "event_type": "knowledge_attention_backfill",
                    "source": "wikimedia_backfill",
                    "comments_text": [f"wiki_title={title}"],
                    "collaborators": [],
                    "manual_seeded": False,
                    "metadata_text": f"{artist} wikipedia historical pageviews",
                }
            )

    return rows, {
        "enabled": True,
        "rows": len(rows),
        "artists_considered": len(artist_names),
        "artists_resolved": resolved,
        "errors": errors,
        "lookback_days": lookback_days,
    }


def collect_bootstrap_backfill(
    config: dict[str, Any],
    *,
    project_root: Path,
    today: dt.date,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = config.get("ingest", {}).get("bootstrap_backfill", {})
    if not bool(cfg.get("enabled", True)):
        return [], {"enabled": False}

    manifest_path = resolve_path(str(config.get("paths", {}).get("bootstrap_datasets", "")), project_root)
    available = _load_dataset_manifest(manifest_path)
    rows: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}

    if "spotify_charts_top200_viral50" in available:
        charts_dir = resolve_path(str(cfg.get("spotify_charts_local_dir", "data/bootstrap/spotify_charts")), project_root)
        chart_rows, chart_meta = _spotify_chart_rows(
            charts_dir,
            today=today,
            max_files_per_run=int(cfg.get("max_files_per_run", 80)),
            lookback_days=int(cfg.get("lookback_days", 365)),
        )
        rows.extend(chart_rows)
        sources["spotify_charts_top200_viral50"] = chart_meta
    else:
        sources["spotify_charts_top200_viral50"] = {
            "enabled": False,
            "reason": "dataset not present in bootstrap manifest",
        }

    if "wikipedia_pageviews_api" in available:
        wiki_rows, wiki_meta = _wikipedia_backfill_rows(
            config,
            project_root,
            today=today,
            lookback_days=int(cfg.get("wiki_lookback_days", 180)),
            max_artists=int(cfg.get("wiki_max_artists", 30)),
        )
        rows.extend(wiki_rows)
        sources["wikipedia_pageviews_api"] = wiki_meta
    else:
        sources["wikipedia_pageviews_api"] = {
            "enabled": False,
            "reason": "dataset not present in bootstrap manifest",
        }

    # Datasets that are manifest-declared but currently not historical-backfill capable
    # in this adapter are reported explicitly to keep calibration expectations clear.
    for dataset_name in sorted(available):
        if dataset_name in sources:
            continue
        sources[dataset_name] = {
            "enabled": False,
            "reason": "declared dataset currently used in forward ingest or optional adapters, not historical bootstrap here",
        }

    return rows, {
        "enabled": True,
        "manifest_path": str(manifest_path),
        "datasets_declared": sorted(available),
        "rows": len(rows),
        "sources": sources,
    }
