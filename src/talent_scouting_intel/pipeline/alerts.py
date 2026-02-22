from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from talent_scouting_intel.utils.io import ensure_parent, load_config, read_csv, resolve_path, write_jsonl


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _stage_rank(stage: str) -> int:
    mapping = {"early": 0, "emerging": 1, "breaking": 2}
    return mapping.get(stage, -1)


def _load_state(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            out[str(key)] = value
    return out


def _write_state(path: Path, state: dict[str, dict[str, Any]]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")


def build_alerts(
    scored_rows: list[dict[str, Any]],
    prev_state: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    alert_cfg = config.get("alerts", {})
    min_watch_score = float(alert_cfg.get("min_watch_score", 0.45))
    min_score_delta = float(alert_cfg.get("min_score_delta", 0.06))
    include_risk = bool(alert_cfg.get("include_risk_alerts", True))
    max_alerts = int(alert_cfg.get("max_alerts", 25))

    generated_at = dt.datetime.now().replace(microsecond=0).isoformat()
    alerts: list[dict[str, Any]] = []
    new_state: dict[str, dict[str, Any]] = {}

    sorted_rows = sorted(scored_rows, key=lambda row: _as_float(row.get("final_score")), reverse=True)

    for row in sorted_rows:
        track_id = str(row.get("track_id", ""))
        if not track_id:
            continue

        score = _as_float(row.get("final_score"))
        stage = str(row.get("stage", "early"))
        inflection = _as_bool(row.get("inflection_detected"))
        suspicious = _as_bool(row.get("suspicious"))
        spike_only = _as_bool(row.get("spike_only"))
        playlist_dependent = _as_bool(row.get("playlist_dependent"))

        prior = prev_state.get(track_id, {})
        prev_score = _as_float(prior.get("score"))
        prev_stage = str(prior.get("stage", "early"))
        prev_inflection = _as_bool(prior.get("inflection_detected"))
        prev_risky = _as_bool(prior.get("risky"))

        risky = suspicious or spike_only or playlist_dependent
        stage_upgraded = _stage_rank(stage) > _stage_rank(prev_stage)
        score_delta = score - prev_score

        if inflection and not prev_inflection:
            alerts.append(
                {
                    "generated_at": generated_at,
                    "type": "inflection",
                    "priority": 0,
                    "track_id": track_id,
                    "track_name": row.get("track_name", ""),
                    "artist_name": row.get("artist_name", ""),
                    "genre": row.get("genre", "unknown"),
                    "score": round(score, 6),
                    "stage": stage,
                    "reason": "Inflection trigger fired with multi-bucket corroboration.",
                    "explanation": row.get("explanation", ""),
                }
            )

        if stage_upgraded:
            alerts.append(
                {
                    "generated_at": generated_at,
                    "type": "stage_upgrade",
                    "priority": 1,
                    "track_id": track_id,
                    "track_name": row.get("track_name", ""),
                    "artist_name": row.get("artist_name", ""),
                    "genre": row.get("genre", "unknown"),
                    "score": round(score, 6),
                    "stage": stage,
                    "reason": f"Stage changed from {prev_stage} to {stage}.",
                    "explanation": row.get("explanation", ""),
                }
            )

        if score >= min_watch_score and score_delta >= min_score_delta:
            alerts.append(
                {
                    "generated_at": generated_at,
                    "type": "momentum_surge",
                    "priority": 1,
                    "track_id": track_id,
                    "track_name": row.get("track_name", ""),
                    "artist_name": row.get("artist_name", ""),
                    "genre": row.get("genre", "unknown"),
                    "score": round(score, 6),
                    "stage": stage,
                    "reason": f"Score increased by {score_delta:.3f} since last run.",
                    "explanation": row.get("explanation", ""),
                }
            )

        if not prior and score >= min_watch_score:
            alerts.append(
                {
                    "generated_at": generated_at,
                    "type": "new_watchlist",
                    "priority": 2,
                    "track_id": track_id,
                    "track_name": row.get("track_name", ""),
                    "artist_name": row.get("artist_name", ""),
                    "genre": row.get("genre", "unknown"),
                    "score": round(score, 6),
                    "stage": stage,
                    "reason": "New high-priority candidate entered watchlist threshold.",
                    "explanation": row.get("explanation", ""),
                }
            )

        if include_risk and risky and not prev_risky:
            risk_parts = []
            if suspicious:
                risk_parts.append("suspicious")
            if spike_only:
                risk_parts.append("spike_only")
            if playlist_dependent:
                risk_parts.append("playlist_dependent")
            alerts.append(
                {
                    "generated_at": generated_at,
                    "type": "risk_flag",
                    "priority": 0,
                    "track_id": track_id,
                    "track_name": row.get("track_name", ""),
                    "artist_name": row.get("artist_name", ""),
                    "genre": row.get("genre", "unknown"),
                    "score": round(score, 6),
                    "stage": stage,
                    "reason": f"Risk signature detected: {', '.join(risk_parts)}.",
                    "explanation": row.get("explanation", ""),
                }
            )

        new_state[track_id] = {
            "score": round(score, 6),
            "stage": stage,
            "inflection_detected": inflection,
            "risky": risky,
            "updated_at": generated_at,
        }

    alerts.sort(key=lambda item: (int(item.get("priority", 9)), -_as_float(item.get("score"))))
    return alerts[:max_alerts], new_state


def _render_markdown(alerts: list[dict[str, Any]], generated_at: str) -> str:
    lines = [
        "# Scout Alerts",
        "",
        f"Generated at: {generated_at}",
        "",
    ]

    if not alerts:
        lines.extend(["No new alerts this run.", ""])
        return "\n".join(lines)

    lines.extend(
        [
            "| Priority | Type | Track | Artist | Genre | Stage | Score | Reason |",
            "|---:|---|---|---|---|---|---:|---|",
        ]
    )
    for item in alerts:
        lines.append(
            "| {priority} | {type} | {track} | {artist} | {genre} | {stage} | {score:.3f} | {reason} |".format(
                priority=item.get("priority", 9),
                type=item.get("type", ""),
                track=str(item.get("track_name", "")).replace("|", "/"),
                artist=str(item.get("artist_name", "")).replace("|", "/"),
                genre=str(item.get("genre", "")).replace("|", "/"),
                stage=str(item.get("stage", "")).replace("|", "/"),
                score=_as_float(item.get("score", 0.0)),
                reason=str(item.get("reason", "")).replace("|", "/"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def run_alerts(
    config_path: str,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    root = project_root or Path.cwd()

    scored = read_csv(resolve_path(config["paths"]["scored"], root))
    alert_path = resolve_path(config["paths"]["alerts_jsonl"], root)
    alert_md = resolve_path(config["paths"]["alerts_md"], root)
    state_path = resolve_path(config["paths"]["alert_state"], root)

    previous_state = _load_state(state_path)
    alerts, new_state = build_alerts(scored, previous_state, config)

    write_jsonl(alert_path, alerts)
    generated_at = dt.datetime.now().replace(microsecond=0).isoformat()
    ensure_parent(alert_md)
    alert_md.write_text(_render_markdown(alerts, generated_at), encoding="utf-8")
    _write_state(state_path, new_state)

    counts: dict[str, int] = {}
    for item in alerts:
        key = str(item.get("type", "other"))
        counts[key] = counts.get(key, 0) + 1

    return {
        "alerts": len(alerts),
        "by_type": counts,
        "alerts_jsonl": str(alert_path),
        "alerts_md": str(alert_md),
        "state_path": str(state_path),
    }
