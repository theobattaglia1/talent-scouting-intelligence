from __future__ import annotations

import datetime as dt
import html
from pathlib import Path
from typing import Any

from talent_scouting_intel.utils.io import ensure_parent, load_config, read_csv, read_jsonl, resolve_path


try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _render_markdown(
    top_rows: list[dict[str, Any]],
    generated_at: str,
    *,
    alerts: list[dict[str, Any]],
    tastemakers: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("# Talent Scouting Intelligence Report")
    lines.append("")
    lines.append(f"Generated at: {generated_at}")
    lines.append("")
    lines.append("## Top Candidates")
    lines.append("")
    lines.append("| Rank | Track | Artist | Genre | Stage | Score | Trust | Affinity | Path Sim | Flags | Why now |")
    lines.append("|---:|---|---|---|---|---:|---:|---:|---:|---|---|")

    for idx, row in enumerate(top_rows, start=1):
        flags = []
        if _bool(row.get("spike_only")):
            flags.append("spike-only")
        if _bool(row.get("suspicious")):
            flags.append("suspicious")
        if _bool(row.get("playlist_dependent")):
            flags.append("playlist-dependent")
        if not flags:
            flags.append("clean")

        why = str(row.get("explanation", ""))
        lines.append(
            (
                "| {rank} | {track} | {artist} | {genre} | {stage} | {score:.3f} | {trust:.3f} | {affinity:.3f} | {path:.3f} | {flags} | {why} |".format(
                rank=idx,
                track=row.get("track_name", ""),
                artist=row.get("artist_name", ""),
                genre=row.get("genre", "unknown"),
                stage=row.get("stage", "early"),
                score=_float(row.get("final_score")),
                trust=_float(row.get("trust_score")),
                affinity=_float(row.get("affinity_score")),
                path=_float(row.get("path_similarity_score")),
                flags=", ".join(flags),
                why=why.replace("|", "/"),
            )
            )
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Inflection requires sustained size-normalized acceleration and at least two corroborating signal buckets.")
    lines.append("- Penalties applied for spike-only, suspicious ratio, and playlist-dependent signatures.")

    lines.append("")
    lines.append("## Scout Inbox")
    lines.append("")
    if alerts:
        lines.append("| Priority | Type | Track | Artist | Score | Reason |")
        lines.append("|---:|---|---|---|---:|---|")
        for item in alerts[:12]:
            lines.append(
                "| {priority} | {type} | {track} | {artist} | {score:.3f} | {reason} |".format(
                    priority=item.get("priority", 9),
                    type=item.get("type", ""),
                    track=str(item.get("track_name", "")).replace("|", "/"),
                    artist=str(item.get("artist_name", "")).replace("|", "/"),
                    score=_float(item.get("score", 0.0)),
                    reason=str(item.get("reason", "")).replace("|", "/"),
                )
            )
    else:
        lines.append("- No new alerts this run.")

    lines.append("")
    lines.append("## Top Tastemakers")
    lines.append("")
    if tastemakers:
        lines.append("| Rank | Tastemaker | Status | Quant Score | Bayes Precision | Avg Lead Days |")
        lines.append("|---:|---|---|---:|---:|---:|")
        for idx, row in enumerate(tastemakers[:10], start=1):
            lines.append(
                "| {rank} | {name} | {status} | {score:.3f} | {precision:.3f} | {lead:.1f} |".format(
                    rank=idx,
                    name=str(row.get("tastemaker_name", "")),
                    status=str(row.get("status", "")),
                    score=_float(row.get("quant_score", 0.0)),
                    precision=_float(row.get("bayes_precision", 0.0)),
                    lead=_float(row.get("avg_lead_days", 0.0)),
                )
            )
    else:
        lines.append("- No tastemaker profiles available yet.")
    return "\n".join(lines)


def _render_html(top_rows: list[dict[str, Any]], generated_at: str, chart_path: str | None) -> str:
    rows_html: list[str] = []
    for idx, row in enumerate(top_rows, start=1):
        flags = []
        if _bool(row.get("spike_only")):
            flags.append("spike-only")
        if _bool(row.get("suspicious")):
            flags.append("suspicious")
        if _bool(row.get("playlist_dependent")):
            flags.append("playlist-dependent")
        flags_text = ", ".join(flags) if flags else "clean"

        rows_html.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td>{html.escape(str(row.get('track_name', '')))}</td>"
            f"<td>{html.escape(str(row.get('artist_name', '')))}</td>"
            f"<td>{html.escape(str(row.get('genre', 'unknown')))}</td>"
            f"<td>{html.escape(str(row.get('stage', 'early')))}</td>"
            f"<td>{_float(row.get('final_score')):.3f}</td>"
            f"<td>{_float(row.get('trust_score')):.3f}</td>"
            f"<td>{_float(row.get('affinity_score')):.3f}</td>"
            f"<td>{_float(row.get('path_similarity_score')):.3f}</td>"
            f"<td>{html.escape(flags_text)}</td>"
            f"<td>{html.escape(str(row.get('explanation', '')))}</td>"
            "</tr>"
        )

    chart_html = ""
    if chart_path:
        chart_html = f"<h2>Score Chart</h2><img src='{html.escape(chart_path)}' style='max-width:100%;height:auto;' />"

    return f"""<!doctype html>
<html>
<head>
  <meta charset='utf-8' />
  <title>Talent Scouting Intelligence Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; background: #f7f8fa; color: #111; }}
    table {{ border-collapse: collapse; width: 100%; background: #fff; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f0f2f6; }}
    .meta {{ margin-bottom: 16px; color: #444; }}
  </style>
</head>
<body>
  <h1>Talent Scouting Intelligence Report</h1>
  <div class='meta'>Generated at: {html.escape(generated_at)}</div>
  {chart_html}
  <h2>Top Candidates</h2>
  <table>
    <thead>
      <tr>
        <th>Rank</th><th>Track</th><th>Artist</th><th>Genre</th><th>Stage</th><th>Score</th><th>Trust</th><th>Affinity</th><th>Path Sim</th><th>Flags</th><th>Why now</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
</body>
</html>"""


def _plot_chart(rows: list[dict[str, Any]], chart_path: Path) -> bool:
    if plt is None:
        return False
    if not rows:
        return False

    labels = [f"{row.get('artist_name', '')} - {row.get('track_name', '')}" for row in rows[:10]]
    scores = [_float(row.get("final_score")) for row in rows[:10]]

    ensure_parent(chart_path)
    plt.figure(figsize=(12, 6))
    plt.barh(list(reversed(labels)), list(reversed(scores)), color="#2f6db5")
    plt.xlabel("Final Score")
    plt.title("Top Candidate Scores")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    return True


def run_report(
    config_path: str,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    root = project_root or Path.cwd()

    scored_path = resolve_path(config["paths"]["scored"], root)
    rows = read_csv(scored_path)
    rows.sort(key=lambda row: _float(row.get("final_score")), reverse=True)

    top_n = int(config["report"]["top_n"])
    top_rows = rows[:top_n]
    generated_at = dt.datetime.now().replace(microsecond=0).isoformat()

    md_path = resolve_path(config["paths"]["report_md"], root)
    html_path = resolve_path(config["paths"]["report_html"], root)
    chart_path = resolve_path(config["paths"]["report_chart"], root)

    alerts_path = resolve_path(config["paths"]["alerts_jsonl"], root)
    tastemakers_path = resolve_path(config["paths"]["tastemakers_csv"], root)
    alerts = read_jsonl(alerts_path)
    tastemakers = read_csv(tastemakers_path)
    tastemakers.sort(key=lambda row: _float(row.get("quant_score")), reverse=True)

    chart_written = False
    if bool(config["report"].get("include_chart", True)):
        chart_written = _plot_chart(top_rows, chart_path)

    markdown = _render_markdown(top_rows, generated_at, alerts=alerts, tastemakers=tastemakers)
    ensure_parent(md_path)
    md_path.write_text(markdown, encoding="utf-8")

    chart_ref = str(chart_path.name) if chart_written else None
    html_doc = _render_html(top_rows, generated_at, chart_ref)
    ensure_parent(html_path)
    html_path.write_text(html_doc, encoding="utf-8")

    return {
        "rows": len(rows),
        "top_rows": len(top_rows),
        "markdown": str(md_path),
        "html": str(html_path),
        "chart": str(chart_path) if chart_written else "not_generated",
    }
