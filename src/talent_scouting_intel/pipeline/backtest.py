from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from talent_scouting_intel.pipeline.candidates import build_candidates
from talent_scouting_intel.pipeline.common import group_by_track, parse_date
from talent_scouting_intel.pipeline.features import build_features
from talent_scouting_intel.pipeline.scoring import build_scores
from talent_scouting_intel.utils.io import load_config, read_csv, read_jsonl, resolve_path, write_json

NORM_RE = re.compile(r"[^a-z0-9]+")
YEAR_MONTH_RE = re.compile(r"(20\d{2})[-/](\d{2})")
YEAR_Q_RE = re.compile(r"(20\d{2})\s*[-/]?\s*q([1-4])", flags=re.IGNORECASE)
YEAR_H_RE = re.compile(r"(20\d{2})\s*[-/]?\s*h([12])", flags=re.IGNORECASE)
YEAR_RE = re.compile(r"(20\d{2})")


@dataclass
class TemplateWindow:
    artist_name: str
    artist_norm: str
    track_name: str
    approx_window: str
    start: dt.date
    end: dt.date


def _normalize(text: Any) -> str:
    return NORM_RE.sub("", str(text or "").strip().lower())


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _end_of_month(year: int, month: int) -> dt.date:
    if month == 12:
        return dt.date(year, 12, 31)
    return dt.date(year, month + 1, 1) - dt.timedelta(days=1)


def _window_from_text(raw: str) -> tuple[dt.date | None, dt.date | None]:
    text = str(raw or "").strip().lower()
    if not text:
        return None, None

    match_ym = YEAR_MONTH_RE.search(text)
    if match_ym:
        year = int(match_ym.group(1))
        month = max(1, min(12, int(match_ym.group(2))))
        return dt.date(year, month, 1), _end_of_month(year, month)

    match_q = YEAR_Q_RE.search(text)
    if match_q:
        year = int(match_q.group(1))
        quarter = int(match_q.group(2))
        start_month = (quarter - 1) * 3 + 1
        end_month = start_month + 2
        return dt.date(year, start_month, 1), _end_of_month(year, end_month)

    match_h = YEAR_H_RE.search(text)
    if match_h:
        year = int(match_h.group(1))
        half = int(match_h.group(2))
        if half == 1:
            return dt.date(year, 1, 1), dt.date(year, 6, 30)
        return dt.date(year, 7, 1), dt.date(year, 12, 31)

    match_year = YEAR_RE.search(text)
    if match_year:
        year = int(match_year.group(1))
        return dt.date(year, 1, 1), dt.date(year, 12, 31)
    return None, None


def _parse_template_window(row: dict[str, Any]) -> TemplateWindow | None:
    artist_name = str(row.get("artist_name", "")).strip()
    if not artist_name:
        return None
    artist_norm = _normalize(artist_name)
    if not artist_norm:
        return None

    iso_start = str(row.get("iso_start", "")).strip()
    iso_end = str(row.get("iso_end", "")).strip()
    start = None
    end = None
    if iso_start:
        try:
            start = dt.date.fromisoformat(iso_start)
        except Exception:
            start = None
    if iso_end:
        try:
            end = dt.date.fromisoformat(iso_end)
        except Exception:
            end = None

    approx_window = str(row.get("approx_window", "")).strip()
    if start is None:
        start, end_from_text = _window_from_text(approx_window)
        if end is None:
            end = end_from_text
    if end is None and start is not None:
        end = start + dt.timedelta(days=30)
    if start is None or end is None:
        return None

    if end < start:
        start, end = end, start

    track_name = str(row.get("track_name", "")).strip()
    return TemplateWindow(
        artist_name=artist_name,
        artist_norm=artist_norm,
        track_name=track_name,
        approx_window=approx_window,
        start=start,
        end=end,
    )


def _load_template_windows(config: dict[str, Any], root: Path) -> dict[str, list[TemplateWindow]]:
    breakout_path = resolve_path(str(config.get("paths", {}).get("breakout_templates", "")), root)
    rows = read_csv(breakout_path)
    out: dict[str, list[TemplateWindow]] = {}
    for row in rows:
        parsed = _parse_template_window(row)
        if parsed is None:
            continue
        out.setdefault(parsed.artist_norm, []).append(parsed)
    for windows in out.values():
        windows.sort(key=lambda item: item.start)
    return out


def _future_broke_date(
    track_rows: list[dict[str, Any]],
    cutoff: dt.date,
    horizon_days: int,
    broke_cfg: dict[str, Any],
) -> dt.date | None:
    horizon_date = cutoff + dt.timedelta(days=horizon_days)
    future = [
        row
        for row in track_rows
        if row["platform"] == "spotify" and cutoff < parse_date(row["date"]) <= horizon_date
    ]
    if len(future) < 21:
        return None

    streams = [float(row.get("streams", 0.0)) for row in future]
    if max(streams, default=0.0) < float(broke_cfg["min_streams"]):
        return None

    min_weekly_growth = float(broke_cfg["min_weekly_growth"])
    sustain_weeks = int(broke_cfg["sustain_weeks"])

    run = 0
    for idx in range(7, len(streams)):
        weekly_growth = (streams[idx] - streams[idx - 7]) / (streams[idx - 7] + 1.0)
        if weekly_growth >= min_weekly_growth:
            run += 1
            if run >= sustain_weeks:
                return parse_date(future[idx]["date"])
        else:
            run = 0
    return None


def _template_future_dates(
    scored_rows: list[dict[str, Any]],
    templates: dict[str, list[TemplateWindow]],
    cutoff: dt.date,
    horizon_days: int,
) -> dict[str, dt.date]:
    horizon_date = cutoff + dt.timedelta(days=horizon_days)
    out: dict[str, dt.date] = {}
    for row in scored_rows:
        track_id = str(row.get("track_id", "")).strip()
        artist_norm = _normalize(row.get("artist_name", ""))
        if not track_id or not artist_norm:
            continue
        windows = templates.get(artist_norm, [])
        for window in windows:
            if window.end <= cutoff:
                continue
            if window.start > horizon_date:
                continue
            event_date = window.start if window.start > cutoff else cutoff + dt.timedelta(days=1)
            existing = out.get(track_id)
            if existing is None or event_date < existing:
                out[track_id] = event_date
    return out


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _frange(start: float, end: float, step: float) -> list[float]:
    if step <= 0:
        return [round(start, 6)]
    out: list[float] = []
    value = start
    while value <= end + 1e-9:
        out.append(round(value, 6))
        value += step
    return out


def _evaluate_thresholds(
    eval_rows: list[dict[str, Any]],
    *,
    final_score_threshold: float,
    trust_score_threshold: float,
) -> dict[str, Any]:
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    lead_times: list[float] = []
    predicted_rows: list[dict[str, Any]] = []
    false_positive_rows: list[dict[str, Any]] = []
    false_negative_rows: list[dict[str, Any]] = []

    for row in eval_rows:
        label = int(row.get("label", 0))
        final_score = _safe_float(row.get("final_score", 0.0))
        trust_score = _safe_float(row.get("trust_score", 0.0))
        stage = str(row.get("stage", "")).strip().lower()
        inflection = _safe_bool(row.get("inflection_detected"))
        predicted = bool(
            final_score >= final_score_threshold
            and trust_score >= trust_score_threshold
            and (inflection or stage in {"emerging", "breaking"})
        )
        if predicted:
            predicted_rows.append(row)

        if predicted and label == 1:
            tp += 1
            lead_time = row.get("lead_time_days")
            if lead_time is not None and str(lead_time).strip() != "":
                lead_times.append(_safe_float(lead_time))
        elif predicted and label == 0:
            fp += 1
            false_positive_rows.append(row)
        elif (not predicted) and label == 1:
            fn += 1
            false_negative_rows.append(row)
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "false_positive_rate": round(fpr, 6),
        "lead_time_days": round(_mean(lead_times), 3),
        "support_positive": int(tp + fn),
        "support_negative": int(fp + tn),
        "predicted_positive": int(tp + fp),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "predicted_rows": predicted_rows,
        "false_positive_rows": false_positive_rows,
        "false_negative_rows": false_negative_rows,
    }


def _failure_reason(row: dict[str, Any], *, failure_type: str) -> str:
    if _safe_bool(row.get("spike_only")):
        return "spike signature still passed thresholds"
    if _safe_bool(row.get("suspicious")):
        return "suspicious ratio pattern still passed thresholds"
    if _safe_bool(row.get("playlist_dependent")):
        return "playlist-dependent pattern"
    if _safe_bool(row.get("established_artist")):
        return "established-artist leakage"
    if _safe_float(row.get("prior_boost", 0.0)) > 0.04 and _safe_float(row.get("acceleration_score", 0.0)) < 0.3:
        return "prior-heavy score without sustained acceleration"
    if failure_type == "fn" and _safe_float(row.get("trust_score", 0.0)) < 0.4:
        return "candidate suppressed by low trust confidence"
    return "threshold mismatch vs realized outcome"


def _compact_failure_rows(rows: list[dict[str, Any]], *, failure_type: str, limit: int = 12) -> list[dict[str, Any]]:
    if failure_type == "fp":
        sorted_rows = sorted(
            rows,
            key=lambda item: (_safe_float(item.get("final_score", 0.0)), _safe_float(item.get("trust_score", 0.0))),
            reverse=True,
        )
    else:
        sorted_rows = sorted(
            rows,
            key=lambda item: (
                -9999.0 if item.get("lead_time_days") is None else -_safe_float(item.get("lead_time_days")),
                _safe_float(item.get("final_score", 0.0)),
            ),
            reverse=True,
        )

    out: list[dict[str, Any]] = []
    for row in sorted_rows[:limit]:
        out.append(
            {
                "cutoff": str(row.get("cutoff", "")),
                "track_id": str(row.get("track_id", "")),
                "track_name": str(row.get("track_name", "")),
                "artist_name": str(row.get("artist_name", "")),
                "final_score": round(_safe_float(row.get("final_score", 0.0)), 6),
                "trust_score": round(_safe_float(row.get("trust_score", 0.0)), 6),
                "stage": str(row.get("stage", "")),
                "lead_time_days": None if row.get("lead_time_days") is None else round(_safe_float(row.get("lead_time_days")), 3),
                "reason": _failure_reason(row, failure_type=failure_type),
            }
        )
    return out


def _calibrate_thresholds(
    eval_rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    windows_count: int,
) -> dict[str, Any]:
    calibration_cfg = config.get("calibration", {})
    enabled = bool(calibration_cfg.get("enabled", True))
    default_thresholds = calibration_cfg.get("default_thresholds", {})
    default_final = _safe_float(default_thresholds.get("final_score", 0.52))
    default_trust = _safe_float(default_thresholds.get("trust_score", 0.45))
    min_windows = int(config.get("backtest", {}).get("min_windows_for_calibration", 4))

    if (not enabled) or (windows_count < max(1, min_windows)) or (not eval_rows):
        metrics = _evaluate_thresholds(
            eval_rows,
            final_score_threshold=default_final,
            trust_score_threshold=default_trust,
        )
        return {
            "enabled": enabled,
            "used_default_thresholds": True,
            "reason": "insufficient_windows_or_disabled",
            "recommended_thresholds": {
                "final_score": round(default_final, 6),
                "trust_score": round(default_trust, 6),
            },
            "metrics": {
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "false_positive_rate": metrics["false_positive_rate"],
                "lead_time_days": metrics["lead_time_days"],
                "predicted_positive": metrics["predicted_positive"],
                "support_positive": metrics["support_positive"],
            },
            "grid_points": 0,
            "top_false_positives": _compact_failure_rows(metrics["false_positive_rows"], failure_type="fp", limit=8),
            "top_false_negatives": _compact_failure_rows(metrics["false_negative_rows"], failure_type="fn", limit=8),
        }

    grid_cfg = calibration_cfg.get("threshold_grid", {})
    score_grid_cfg = grid_cfg.get("final_score", {})
    trust_grid_cfg = grid_cfg.get("trust_score", {})
    score_values = _frange(
        _safe_float(score_grid_cfg.get("min", 0.05)),
        _safe_float(score_grid_cfg.get("max", 0.95)),
        _safe_float(score_grid_cfg.get("step", 0.02)),
    )
    trust_values = _frange(
        _safe_float(trust_grid_cfg.get("min", 0.05)),
        _safe_float(trust_grid_cfg.get("max", 0.95)),
        _safe_float(trust_grid_cfg.get("step", 0.02)),
    )
    objective_cfg = calibration_cfg.get("objective", {})
    objective_name = str(objective_cfg.get("maximize", "f1")).strip().lower()
    if objective_name not in {"f1", "precision", "recall", "lead_time_days"}:
        objective_name = "f1"
    min_precision = _safe_float(objective_cfg.get("min_precision", 0.0))

    best_thresholds = (default_final, default_trust)
    best_metrics: dict[str, Any] | None = None
    best_objective = -1.0
    grid_points = 0

    for score_thr in score_values:
        for trust_thr in trust_values:
            grid_points += 1
            metrics = _evaluate_thresholds(
                eval_rows,
                final_score_threshold=score_thr,
                trust_score_threshold=trust_thr,
            )
            precision = _safe_float(metrics["precision"])
            if precision < min_precision:
                continue
            objective_value = _safe_float(metrics.get(objective_name, 0.0))
            if best_metrics is None:
                best_thresholds = (score_thr, trust_thr)
                best_metrics = metrics
                best_objective = objective_value
                continue
            if objective_value > best_objective + 1e-9:
                best_thresholds = (score_thr, trust_thr)
                best_metrics = metrics
                best_objective = objective_value
                continue
            if abs(objective_value - best_objective) <= 1e-9:
                # Tie-break: prefer higher precision, then lower FPR, then higher lead time.
                if (
                    _safe_float(metrics["precision"]) > _safe_float(best_metrics["precision"]) + 1e-9
                    or (
                        abs(_safe_float(metrics["precision"]) - _safe_float(best_metrics["precision"])) <= 1e-9
                        and (
                            _safe_float(metrics["false_positive_rate"])
                            < _safe_float(best_metrics["false_positive_rate"]) - 1e-9
                        )
                    )
                    or (
                        abs(_safe_float(metrics["precision"]) - _safe_float(best_metrics["precision"])) <= 1e-9
                        and abs(
                            _safe_float(metrics["false_positive_rate"]) - _safe_float(best_metrics["false_positive_rate"])
                        )
                        <= 1e-9
                        and _safe_float(metrics["lead_time_days"]) > _safe_float(best_metrics["lead_time_days"]) + 1e-9
                    )
                ):
                    best_thresholds = (score_thr, trust_thr)
                    best_metrics = metrics
                    best_objective = objective_value

    if best_metrics is None:
        best_metrics = _evaluate_thresholds(
            eval_rows,
            final_score_threshold=default_final,
            trust_score_threshold=default_trust,
        )
        best_thresholds = (default_final, default_trust)
        used_defaults = True
    else:
        used_defaults = False

    return {
        "enabled": True,
        "used_default_thresholds": used_defaults,
        "objective": objective_name,
        "objective_min_precision": min_precision,
        "recommended_thresholds": {
            "final_score": round(best_thresholds[0], 6),
            "trust_score": round(best_thresholds[1], 6),
        },
        "metrics": {
            "precision": best_metrics["precision"],
            "recall": best_metrics["recall"],
            "f1": best_metrics["f1"],
            "false_positive_rate": best_metrics["false_positive_rate"],
            "lead_time_days": best_metrics["lead_time_days"],
            "predicted_positive": best_metrics["predicted_positive"],
            "support_positive": best_metrics["support_positive"],
        },
        "grid_points": grid_points,
        "top_false_positives": _compact_failure_rows(best_metrics["false_positive_rows"], failure_type="fp", limit=12),
        "top_false_negatives": _compact_failure_rows(best_metrics["false_negative_rows"], failure_type="fn", limit=12),
    }


def run_backtest(
    config_path: str,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    root = project_root or Path.cwd()
    snapshots = read_jsonl(resolve_path(config["paths"]["snapshots"], root))
    grouped = group_by_track(snapshots)
    template_windows = _load_template_windows(config, root)

    bt_cfg = config["backtest"]
    broke_cfg = config["thresholds"]["broke_proxy"]

    top_k = int(bt_cfg["top_k"])
    min_history = int(bt_cfg["min_history_days"])
    step_days = int(bt_cfg["replay_step_days"])
    horizon_days = int(bt_cfg["prediction_horizon_days"])

    unique_dates = sorted({parse_date(row["date"]) for row in snapshots})
    if len(unique_dates) < min_history + horizon_days + 7:
        raise ValueError("Not enough history for configured backtest windows.")

    start_idx = min_history
    end_date = unique_dates[-1]

    windows: list[dict[str, Any]] = []
    previous_top: set[str] | None = None
    eval_rows: list[dict[str, Any]] = []

    for idx in range(start_idx, len(unique_dates), step_days):
        cutoff = unique_dates[idx]
        if cutoff + dt.timedelta(days=horizon_days) > end_date:
            break

        historical = [row for row in snapshots if parse_date(row["date"]) <= cutoff]
        candidates = build_candidates(historical, config)
        features = build_features(historical, candidates, config)
        scored, _ = build_scores(features, config, project_root=root)

        top_tracks = [row["track_id"] for row in scored[:top_k]]
        top_set = set(top_tracks)

        universe = {row["track_id"] for row in scored}
        proxy_dates: dict[str, dt.date] = {}
        for track_id in universe:
            broke_date = _future_broke_date(grouped.get(track_id, []), cutoff, horizon_days, broke_cfg)
            if broke_date is not None:
                proxy_dates[track_id] = broke_date

        template_dates = _template_future_dates(scored, template_windows, cutoff, horizon_days)
        positive_dates: dict[str, dt.date] = dict(template_dates)
        for track_id, day in proxy_dates.items():
            previous = positive_dates.get(track_id)
            if previous is None or day < previous:
                positive_dates[track_id] = day

        actual_positive = set(positive_dates.keys())
        tp = len(top_set & actual_positive)
        fp = len(top_set - actual_positive)
        fn = len(actual_positive - top_set)
        tn = len(universe - (top_set | actual_positive))

        precision = tp / top_k if top_k else 0.0
        recall = tp / len(actual_positive) if actual_positive else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        lead_times = [
            (positive_dates[track_id] - cutoff).days
            for track_id in (top_set & actual_positive)
            if track_id in positive_dates
        ]
        avg_lead_time = _mean([float(value) for value in lead_times]) if lead_times else 0.0

        stability = 0.0
        if previous_top is not None:
            union = previous_top | top_set
            inter = previous_top & top_set
            stability = len(inter) / len(union) if union else 0.0
        previous_top = top_set

        windows.append(
            {
                "cutoff": cutoff.isoformat(),
                "precision_at_k": round(precision, 6),
                "recall_at_k": round(recall, 6),
                "false_positive_rate": round(fpr, 6),
                "avg_lead_time_days": round(avg_lead_time, 3),
                "top_k": top_tracks,
                "actual_broke": sorted(actual_positive),
                "stability_vs_prev": round(stability, 6),
                "labels_proxy_only": len([track_id for track_id in actual_positive if track_id in proxy_dates]),
                "labels_template_only": len([track_id for track_id in actual_positive if track_id in template_dates and track_id not in proxy_dates]),
            }
        )

        for row in scored:
            track_id = str(row.get("track_id", "")).strip()
            label = 1 if track_id in actual_positive else 0
            lead = None
            if label == 1:
                lead = (positive_dates[track_id] - cutoff).days
            eval_rows.append(
                {
                    "cutoff": cutoff.isoformat(),
                    "track_id": track_id,
                    "track_name": str(row.get("track_name", "")),
                    "artist_name": str(row.get("artist_name", "")),
                    "final_score": _safe_float(row.get("final_score", 0.0)),
                    "trust_score": _safe_float(row.get("trust_score", 0.0)),
                    "stage": str(row.get("stage", "")),
                    "inflection_detected": _safe_bool(row.get("inflection_detected")),
                    "label": label,
                    "lead_time_days": lead,
                    "spike_only": _safe_bool(row.get("spike_only")),
                    "suspicious": _safe_bool(row.get("suspicious")),
                    "playlist_dependent": _safe_bool(row.get("playlist_dependent")),
                    "established_artist": _safe_bool(row.get("established_artist")),
                    "prior_boost": _safe_float(row.get("prior_boost", 0.0)),
                    "acceleration_score": _safe_float(row.get("acceleration_score", 0.0)),
                }
            )

    summary = {
        "windows": len(windows),
        "precision_at_k": round(_mean([w["precision_at_k"] for w in windows]), 6),
        "recall_at_k": round(_mean([w["recall_at_k"] for w in windows]), 6),
        "false_positive_rate": round(_mean([w["false_positive_rate"] for w in windows]), 6),
        "lead_time_days": round(_mean([w["avg_lead_time_days"] for w in windows]), 3),
        "week_to_week_stability": round(_mean([w["stability_vs_prev"] for w in windows[1:]]), 6)
        if len(windows) > 1
        else 0.0,
        "label_mix": {
            "proxy_only": int(sum(int(window.get("labels_proxy_only", 0)) for window in windows)),
            "template_only": int(sum(int(window.get("labels_template_only", 0)) for window in windows)),
        },
    }

    calibration = _calibrate_thresholds(eval_rows, config, windows_count=len(windows))

    payload = {
        "summary": summary,
        "windows": windows,
        "calibration": calibration,
    }
    write_json(resolve_path(config["paths"]["backtest_json"], root), payload)
    write_json(resolve_path(config["paths"]["calibration_json"], root), calibration)

    backtest_md = resolve_path(config["paths"]["backtest_md"], root)
    lines = [
        "# Backtest Report",
        "",
        "## Summary",
        "",
        f"- Windows: {summary['windows']}",
        f"- Precision@{top_k}: {summary['precision_at_k']}",
        f"- Recall@{top_k}: {summary['recall_at_k']}",
        f"- False Positive Rate: {summary['false_positive_rate']}",
        f"- Average Lead Time (days): {summary['lead_time_days']}",
        f"- Week-to-week Stability: {summary['week_to_week_stability']}",
        f"- Label mix (proxy/template-only): {summary['label_mix']['proxy_only']}/{summary['label_mix']['template_only']}",
        "",
        "## Replay Windows",
        "",
    ]
    for window in windows:
        lines.append(
            "- {cutoff}: precision={precision_at_k}, recall={recall_at_k}, fpr={false_positive_rate}, lead_time={avg_lead_time_days}".format(
                **window
            )
        )
    backtest_md.write_text("\n".join(lines), encoding="utf-8")

    calibration_md = resolve_path(config["paths"]["calibration_md"], root)
    cal_lines = [
        "# Threshold Calibration",
        "",
        f"- Objective: `{calibration.get('objective', 'f1')}`",
        f"- Min precision constraint: `{calibration.get('objective_min_precision', 0.0)}`",
        f"- Grid points searched: `{calibration.get('grid_points', 0)}`",
        "",
        "## Recommended Thresholds",
        "",
        f"- Final score >= `{_safe_float(((calibration.get('recommended_thresholds') or {}).get('final_score', 0.0))):.3f}`",
        f"- Trust score >= `{_safe_float(((calibration.get('recommended_thresholds') or {}).get('trust_score', 0.0))):.3f}`",
        "",
        "## Metrics At Recommended Thresholds",
        "",
        f"- Precision: `{_safe_float(((calibration.get('metrics') or {}).get('precision', 0.0))):.3f}`",
        f"- Recall: `{_safe_float(((calibration.get('metrics') or {}).get('recall', 0.0))):.3f}`",
        f"- F1: `{_safe_float(((calibration.get('metrics') or {}).get('f1', 0.0))):.3f}`",
        f"- False positive rate: `{_safe_float(((calibration.get('metrics') or {}).get('false_positive_rate', 0.0))):.3f}`",
        f"- Lead time days: `{_safe_float(((calibration.get('metrics') or {}).get('lead_time_days', 0.0))):.2f}`",
        "",
    ]
    calibration_md.write_text("\n".join(cal_lines), encoding="utf-8")

    calibrated_md = resolve_path(config["paths"]["calibration_backtest_md"], root)
    failure_lines = [
        "# Calibrated Backtest (Failure Analysis)",
        "",
        "## Top False Positives",
        "",
    ]
    top_fps = calibration.get("top_false_positives", [])
    if isinstance(top_fps, list) and top_fps:
        for row in top_fps:
            failure_lines.append(
                "- {artist_name} - {track_name} | score={final_score:.3f} trust={trust_score:.3f} | {reason}".format(
                    **{
                        "artist_name": str(row.get("artist_name", "")),
                        "track_name": str(row.get("track_name", "")),
                        "final_score": _safe_float(row.get("final_score", 0.0)),
                        "trust_score": _safe_float(row.get("trust_score", 0.0)),
                        "reason": str(row.get("reason", "")),
                    }
                )
            )
    else:
        failure_lines.append("- No false positives at calibrated thresholds.")
    failure_lines.extend(["", "## Top False Negatives", ""])
    top_fns = calibration.get("top_false_negatives", [])
    if isinstance(top_fns, list) and top_fns:
        for row in top_fns:
            failure_lines.append(
                "- {artist_name} - {track_name} | score={final_score:.3f} trust={trust_score:.3f} lead={lead_time_days}d | {reason}".format(
                    **{
                        "artist_name": str(row.get("artist_name", "")),
                        "track_name": str(row.get("track_name", "")),
                        "final_score": _safe_float(row.get("final_score", 0.0)),
                        "trust_score": _safe_float(row.get("trust_score", 0.0)),
                        "lead_time_days": row.get("lead_time_days"),
                        "reason": str(row.get("reason", "")),
                    }
                )
            )
    else:
        failure_lines.append("- No false negatives at calibrated thresholds.")
    calibrated_md.write_text("\n".join(failure_lines), encoding="utf-8")

    return {
        "summary": summary,
        "calibration": {
            "recommended_thresholds": calibration.get("recommended_thresholds", {}),
            "metrics": calibration.get("metrics", {}),
        },
        "backtest_json": str(resolve_path(config["paths"]["backtest_json"], root)),
        "backtest_md": str(backtest_md),
        "calibration_json": str(resolve_path(config["paths"]["calibration_json"], root)),
        "calibration_md": str(calibration_md),
        "calibration_backtest_md": str(calibrated_md),
    }
