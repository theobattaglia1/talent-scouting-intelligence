from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from talent_scouting_intel.utils.io import ensure_parent, resolve_path, write_csv

NORM_RE = re.compile(r"[^a-z0-9]+")


def _norm(value: Any) -> str:
    return NORM_RE.sub("", str(value or "").strip().lower())


def _prefix_value(value: str, prefix: str) -> str:
    text = str(value or "").strip()
    if text.startswith(prefix):
        return text[len(prefix) :].strip()
    return ""


def _artist_canonical_id(row: dict[str, Any]) -> tuple[str, int]:
    artist_id = str(row.get("artist_id", ""))
    spotify = _prefix_value(artist_id, "sp_")
    if spotify:
        return f"artist:spotify:{spotify}", 5
    mb = _prefix_value(artist_id, "mb_")
    if mb:
        return f"artist:musicbrainz:{mb}", 4
    youtube = _prefix_value(artist_id, "ytc_")
    if youtube:
        return f"artist:youtube:{youtube}", 3
    lastfm = _prefix_value(artist_id, "lfm_")
    if lastfm:
        return f"artist:lastfm:{lastfm}", 2
    return f"artist:name:{_norm(row.get('artist_name', ''))}", 1


def _track_canonical_id(row: dict[str, Any]) -> tuple[str, int]:
    track_id = str(row.get("track_id", ""))
    spotify = _prefix_value(track_id, "sp_")
    if spotify:
        return f"track:spotify:{spotify}", 5
    mb = _prefix_value(track_id, "mb_")
    if mb:
        return f"track:musicbrainz:{mb}", 4
    youtube = _prefix_value(track_id, "yt_")
    if youtube:
        return f"track:youtube:{youtube}", 3
    lastfm = _prefix_value(track_id, "lfm_")
    if lastfm:
        return f"track:lastfm:{lastfm}", 2
    key = f"{_norm(row.get('artist_name', ''))}:{_norm(row.get('track_name', ''))}"
    return f"track:name:{key}", 1


def _prefer(existing: tuple[str, int], candidate: tuple[str, int]) -> tuple[str, int]:
    if candidate[1] > existing[1]:
        return candidate
    if candidate[1] == existing[1] and candidate[0] < existing[0]:
        return candidate
    return existing


def _entity_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    artist_ids = {str(row.get("canonical_artist_id", "")) for row in rows if str(row.get("canonical_artist_id", ""))}
    track_ids = {str(row.get("canonical_track_id", "")) for row in rows if str(row.get("canonical_track_id", ""))}
    return {
        "rows": len(rows),
        "artists": len(artist_ids),
        "tracks": len(track_ids),
    }


def enrich_snapshots_with_entities(
    config: dict[str, Any],
    root: Path,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        return rows, {"enabled": True, "rows": 0}

    artist_name_best: dict[str, tuple[str, int]] = {}
    track_name_best: dict[str, tuple[str, int]] = {}

    for row in rows:
        artist_name_key = _norm(row.get("artist_name", ""))
        track_name_key = f"{_norm(row.get('artist_name', ''))}:{_norm(row.get('track_name', ''))}"
        artist_canon = _artist_canonical_id(row)
        track_canon = _track_canonical_id(row)
        if artist_name_key:
            artist_name_best[artist_name_key] = _prefer(artist_name_best.get(artist_name_key, artist_canon), artist_canon)
        if track_name_key and not track_name_key.endswith(":"):
            track_name_best[track_name_key] = _prefer(track_name_best.get(track_name_key, track_canon), track_canon)

    enriched: list[dict[str, Any]] = []
    alias_rows: list[dict[str, Any]] = []
    artist_alias_counts: dict[tuple[str, str], int] = defaultdict(int)
    track_alias_counts: dict[tuple[str, str], int] = defaultdict(int)

    for row in rows:
        out = dict(row)
        artist_name_key = _norm(row.get("artist_name", ""))
        track_name_key = f"{_norm(row.get('artist_name', ''))}:{_norm(row.get('track_name', ''))}"
        artist_canon = _artist_canonical_id(row)[0]
        track_canon = _track_canonical_id(row)[0]
        if artist_name_key and artist_name_key in artist_name_best:
            artist_canon = artist_name_best[artist_name_key][0]
        if track_name_key in track_name_best:
            track_canon = track_name_best[track_name_key][0]

        out["canonical_artist_id"] = artist_canon
        out["canonical_track_id"] = track_canon
        out["artist_alias_key"] = artist_name_key
        out["track_alias_key"] = track_name_key
        enriched.append(out)

        artist_alias_counts[(artist_canon, str(row.get("artist_id", "")))] += 1
        track_alias_counts[(track_canon, str(row.get("track_id", "")))] += 1

    latest_by_artist: dict[str, str] = {}
    latest_by_track: dict[str, str] = {}
    for row in enriched:
        day = str(row.get("date", ""))
        artist_canon = str(row.get("canonical_artist_id", ""))
        track_canon = str(row.get("canonical_track_id", ""))
        if day and artist_canon and day > latest_by_artist.get(artist_canon, ""):
            latest_by_artist[artist_canon] = day
        if day and track_canon and day > latest_by_track.get(track_canon, ""):
            latest_by_track[track_canon] = day

    for (canon, alias), count in sorted(artist_alias_counts.items()):
        alias_rows.append(
            {
                "entity_type": "artist",
                "canonical_id": canon,
                "alias_id": alias,
                "rows": count,
                "last_seen": latest_by_artist.get(canon, ""),
            }
        )
    for (canon, alias), count in sorted(track_alias_counts.items()):
        alias_rows.append(
            {
                "entity_type": "track",
                "canonical_id": canon,
                "alias_id": alias,
                "rows": count,
                "last_seen": latest_by_track.get(canon, ""),
            }
        )

    csv_path = resolve_path(str(config.get("paths", {}).get("entity_resolution_csv", "outputs/entity_resolution.csv")), root)
    json_path = resolve_path(str(config.get("paths", {}).get("entity_resolution_json", "outputs/state/entity_resolution.json")), root)
    write_csv(csv_path, alias_rows)
    ensure_parent(json_path)
    json_path.write_text(
        json.dumps(
            {
                "version": 1,
                "artists": sorted({row["canonical_artist_id"] for row in enriched}),
                "tracks": sorted({row["canonical_track_id"] for row in enriched}),
                "aliases": alias_rows,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    stats = _entity_stats(enriched)
    stats.update(
        {
            "enabled": True,
            "csv_path": str(csv_path),
            "json_path": str(json_path),
            "alias_rows": len(alias_rows),
        }
    )
    return enriched, stats
