from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from talent_scouting_intel.adapters.autonomous_collector import collect_autonomous_snapshots
from talent_scouting_intel.adapters.bootstrap_backfill import collect_bootstrap_backfill
from talent_scouting_intel.adapters.manual_import import load_manual_snapshots
from talent_scouting_intel.adapters.mock_adapter import generate_mock_snapshots
from talent_scouting_intel.pipeline.entity_resolution import enrich_snapshots_with_entities
from talent_scouting_intel.pipeline.source_registry import merge_starter_source_additions
from talent_scouting_intel.pipeline.tracking_pool import build_tracking_targets, refresh_tracking_pool
from talent_scouting_intel.utils.io import load_config, read_jsonl, resolve_path, write_jsonl


def _load_seed_urls_from_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        urls = payload.get("seed_urls", [])
    else:
        urls = payload
    if not isinstance(urls, list):
        return []
    return [str(url) for url in urls]


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("date", "")),
        str(row.get("platform", "")),
        str(row.get("track_id", "")),
        str(row.get("tastemaker_id", "")),
        str(row.get("event_type", "")),
        str(row.get("source", "")),
    )


def _merge_history_rows(existing: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for row in existing:
        merged[_row_key(row)] = row
    for row in new_rows:
        merged[_row_key(row)] = row
    return list(merged.values())


def _retention_filter(rows: list[dict[str, Any]], retention_days: int) -> list[dict[str, Any]]:
    if retention_days <= 0 or not rows:
        return rows
    parsed: list[tuple[dict[str, Any], date]] = []
    for row in rows:
        try:
            parsed.append((row, date.fromisoformat(str(row.get("date", "")))))
        except Exception:
            continue
    if not parsed:
        return rows
    latest = max(day for _, day in parsed)
    return [row for row, day in parsed if (latest - day).days <= retention_days]


def run_ingest(
    config_path: str,
    *,
    mode: str = "auto",
    manual_import_path: str | None = None,
    seed_urls: list[str] | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    root = project_root or Path.cwd()
    output_path = resolve_path(config["paths"]["snapshots"], root)

    file_seed_urls = _load_seed_urls_from_file(root / "data" / "mock" / "manual_seeds.json")
    merged_seed_urls = list(seed_urls or []) + file_seed_urls

    rows: list[dict[str, Any]] = []
    source_stats: dict[str, Any] = {}
    tracking_targets: dict[str, list[str]] = {}

    if mode in {"auto", "hybrid"}:
        registry_bootstrap = merge_starter_source_additions(config, root)
        source_stats["source_registry_bootstrap"] = registry_bootstrap
        tracking_refresh = refresh_tracking_pool(config, root)
        tracking_targets, tracking_target_stats = build_tracking_targets(config, root)
        source_stats["tracking_pool"] = {
            "refresh": tracking_refresh,
            "target_stats": tracking_target_stats,
        }

    if mode in {"auto", "hybrid"}:
        auto_rows, auto_stats = collect_autonomous_snapshots(
            config,
            project_root=root,
            seed_urls=merged_seed_urls,
            tracking_targets=tracking_targets,
        )
        rows.extend(auto_rows)
        source_stats["autonomous"] = auto_stats

        backfill_rows, backfill_stats = collect_bootstrap_backfill(config, project_root=root, today=date.today())
        rows.extend(backfill_rows)
        source_stats["bootstrap_backfill"] = backfill_stats

    if mode in {"mock", "hybrid"}:
        mock_days = int(config.get("project", {}).get("mock_days", 112))
        rows.extend(generate_mock_snapshots(config, days=mock_days, seed_urls=merged_seed_urls))
        source_stats["mock_days"] = mock_days

    if mode == "manual":
        if not manual_import_path:
            raise ValueError("--manual-import-path is required for manual mode.")
        rows.extend(load_manual_snapshots(manual_import_path))
        source_stats["manual_import_path"] = manual_import_path

    if mode not in {"auto", "mock", "manual", "hybrid"}:
        raise ValueError("mode must be one of: auto, mock, manual, hybrid")

    if mode == "auto" and not rows and bool(config.get("ingest", {}).get("auto", {}).get("fallback_to_mock_if_empty", True)):
        mock_days = int(config.get("project", {}).get("mock_days", 112))
        rows.extend(generate_mock_snapshots(config, days=mock_days, seed_urls=merged_seed_urls))
        source_stats["auto_fallback_to_mock"] = True

    append_history = bool(config.get("ingest", {}).get("append_history", mode == "auto"))
    existing_rows = read_jsonl(output_path) if append_history else []
    if append_history:
        rows = _merge_history_rows(existing_rows, rows)
        retention_days = int(config.get("ingest", {}).get("history_retention_days", 365))
        rows = _retention_filter(rows, retention_days)
        source_stats["history"] = {
            "enabled": True,
            "existing_rows": len(existing_rows),
            "retention_days": retention_days,
        }

    rows, entity_stats = enrich_snapshots_with_entities(config, root, rows)
    source_stats["entity_resolution"] = entity_stats

    rows.sort(key=lambda r: (r["date"], r["track_id"], r["platform"]))
    write_jsonl(output_path, rows)

    return {
        "rows": len(rows),
        "tracks": len({row["track_id"] for row in rows}),
        "artists": len({row["artist_id"] for row in rows}),
        "output_path": str(output_path),
        "mode": mode,
        "source_stats": source_stats,
    }
