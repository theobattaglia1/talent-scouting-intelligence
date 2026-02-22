from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {
    "date",
    "platform",
    "track_id",
    "track_name",
    "artist_id",
    "artist_name",
}


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    parsed = dict(row)
    parsed.setdefault("artist_followers", 0)
    parsed.setdefault("release_date", parsed.get("date", ""))
    parsed.setdefault("genre_hint", "unknown")
    parsed.setdefault("views", 0)
    parsed.setdefault("likes", 0)
    parsed.setdefault("comments", 0)
    parsed.setdefault("shares", 0)
    parsed.setdefault("streams", parsed.get("views", 0))
    parsed.setdefault("listeners", 0)
    parsed.setdefault("playlist_adds", 0)
    parsed.setdefault("creator_reuse", 0)
    parsed.setdefault("region_metrics", {})
    parsed.setdefault("tastemaker_id", None)
    parsed.setdefault("tastemaker_name", None)
    parsed.setdefault("event_type", "none")
    parsed.setdefault("source", "import")
    parsed.setdefault("comments_text", [])
    parsed.setdefault("collaborators", [])
    parsed.setdefault("manual_seeded", False)
    parsed.setdefault("metadata_text", "")

    for int_field in [
        "artist_followers",
        "views",
        "likes",
        "comments",
        "shares",
        "streams",
        "listeners",
        "playlist_adds",
        "creator_reuse",
    ]:
        try:
            parsed[int_field] = int(float(parsed.get(int_field, 0)))
        except Exception:
            parsed[int_field] = 0

    if isinstance(parsed.get("region_metrics"), str):
        try:
            parsed["region_metrics"] = json.loads(parsed["region_metrics"])
        except Exception:
            parsed["region_metrics"] = {}

    if isinstance(parsed.get("comments_text"), str):
        value = parsed["comments_text"].strip()
        if value.startswith("["):
            try:
                parsed["comments_text"] = json.loads(value)
            except Exception:
                parsed["comments_text"] = [value]
        else:
            parsed["comments_text"] = [value] if value else []

    if isinstance(parsed.get("collaborators"), str):
        value = parsed["collaborators"]
        parsed["collaborators"] = [v.strip() for v in value.split(";") if v.strip()]

    return parsed


def load_manual_snapshots(path: str) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Manual snapshot import not found: {source}")

    rows: list[dict[str, Any]]
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = payload.get("rows", [])
        elif isinstance(payload, list):
            rows = payload
        else:
            rows = []
    elif source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [dict(row) for row in reader]
    else:
        raise ValueError("Manual import supports only .csv and .json")

    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        missing = [col for col in REQUIRED_COLUMNS if col not in row or row[col] in (None, "")]
        if missing:
            raise ValueError(f"Row {idx} missing required columns: {missing}")
        normalized.append(_normalize_row(row))
    return normalized
