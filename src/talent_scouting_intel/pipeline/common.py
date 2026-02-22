from __future__ import annotations

import datetime as dt
from collections import defaultdict
from typing import Any


def parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def filter_as_of(rows: list[dict[str, Any]], as_of: str | None) -> list[dict[str, Any]]:
    if as_of is None:
        return rows
    cutoff = parse_date(as_of)
    return [row for row in rows if parse_date(row["date"]) <= cutoff]


def group_by_track(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["track_id"]].append(row)
    for track_rows in grouped.values():
        track_rows.sort(key=lambda row: (row["date"], row["platform"]))
    return dict(grouped)


def group_by_track_platform(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["track_id"], row["platform"])].append(row)
    for key in grouped:
        grouped[key].sort(key=lambda row: row["date"])
    return dict(grouped)


def rolling_sum(values: list[float], window: int) -> list[float]:
    if window <= 0:
        return []
    out: list[float] = []
    total = 0.0
    for idx, value in enumerate(values):
        total += value
        if idx >= window:
            total -= values[idx - window]
        if idx >= window - 1:
            out.append(total)
    return out


def normalize_dict(values: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, v) for v in values.values())
    if total == 0:
        if not values:
            return {}
        uniform = 1.0 / len(values)
        return {k: uniform for k in values}
    return {k: max(0.0, v) / total for k, v in values.items()}
