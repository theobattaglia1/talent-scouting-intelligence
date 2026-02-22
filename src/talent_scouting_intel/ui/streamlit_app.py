from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from talent_scouting_intel.pipeline.alerts import run_alerts
from talent_scouting_intel.pipeline.backtest import run_backtest
from talent_scouting_intel.pipeline.candidates import run_candidates
from talent_scouting_intel.pipeline.features import run_features
from talent_scouting_intel.pipeline.ingest import run_ingest
from talent_scouting_intel.pipeline.report import run_report
from talent_scouting_intel.pipeline.scoring import run_score
from talent_scouting_intel.pipeline.tastemakers import run_tastemakers
from talent_scouting_intel.ui.data import (
    artist_timeseries,
    find_track_row,
    load_snapshots,
    load_outputs,
    track_platform_map,
    track_region_map,
    track_sparkline_map,
    track_timeseries,
)
from talent_scouting_intel.ui.state import DEFAULT_GENRES, load_ui_state, merge_source_registry, save_ui_state
from talent_scouting_intel.ui.styles import APP_CSS, stage_chip
from talent_scouting_intel.utils.io import ensure_parent, load_config, resolve_path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
RUNTIME_CONFIG_PATH = PROJECT_ROOT / "outputs" / "state" / "ui_runtime_config.json"
RUNTIME_SOURCE_REGISTRY_PATH = PROJECT_ROOT / "outputs" / "state" / "source_registry.runtime.json"

PAGES = [
    "Home / Watchlist",
    "Track Detail",
    "Artist Detail",
    "Tastemakers",
    "Alerts",
    "Backtest / History",
    "Settings",
]

FLAG_LABELS = {
    "spike_only": "Spike-only (jump then reversion, weak depth)",
    "suspicious": "Suspicious ratios (possible inorganic pattern)",
    "playlist_dependent": "Playlist-dependent (single-source dependence)",
    "established_artist": "Established artist (size beyond early-breakout focus)",
    "insufficient_history": "Insufficient history depth (collecting baseline)",
    "weak_evidence": "Weak supporting momentum evidence",
    "low_trust": "Low trust confidence (insufficient corroboration depth)",
}

SOURCE_TYPE_OPTIONS = {
    "YouTube channel": "youtube_channels",
    "Spotify playlist": "spotify_playlists",
    "Reddit subreddit": "reddit_subreddits",
    "Last.fm tag": "lastfm_tags",
    "RSS feed": "rss_feeds",
}


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _inject_styles() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)


def _set_defaults(ui_state: dict[str, Any]) -> None:
    st.session_state.setdefault("page", "Home / Watchlist")
    st.session_state.setdefault("explain_track_id", "")
    st.session_state.setdefault("selected_track_id", "")
    st.session_state.setdefault("selected_artist_id", "")
    st.session_state.setdefault("load_snapshots", False)
    st.session_state.setdefault("onboarding_step", int(ui_state.get("onboarding_step", 1)))


def _runtime_config(ui_state: dict[str, Any], base_config_path: Path) -> Path:
    config = load_config(str(base_config_path))
    config["genres"]["priority"] = list(ui_state.get("genre_focus", DEFAULT_GENRES))

    source_mode = str(ui_state.get("discovery_source_mode", "starter_pack"))
    if source_mode == "custom":
        starter_registry_path = resolve_path(config["ingest"]["auto"]["source_registry"], PROJECT_ROOT)
        starter_registry: dict[str, Any] = {}
        if starter_registry_path.exists():
            try:
                starter_registry = json.loads(starter_registry_path.read_text(encoding="utf-8"))
            except Exception:
                starter_registry = {}

        merged_registry = merge_source_registry(starter_registry, ui_state.get("custom_sources", {}))
        ensure_parent(RUNTIME_SOURCE_REGISTRY_PATH)
        RUNTIME_SOURCE_REGISTRY_PATH.write_text(
            json.dumps(merged_registry, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        config["ingest"]["auto"]["source_registry"] = str(RUNTIME_SOURCE_REGISTRY_PATH)

    ensure_parent(RUNTIME_CONFIG_PATH)
    RUNTIME_CONFIG_PATH.write_text(json.dumps(config, indent=2, ensure_ascii=True), encoding="utf-8")
    return RUNTIME_CONFIG_PATH


def _pipeline_stage_messages() -> list[tuple[str, str]]:
    return [
        ("ingest", "Collecting fresh source snapshots and daily deltas."),
        ("candidates", "Generating low-obviousness candidates with tastemaker + anomaly logic."),
        ("features", "Computing acceleration, depth, resonance, network, and anti-gaming features."),
        ("score", "Applying interpretable weighted scoring and inflection detection."),
        ("tastemakers", "Re-estimating tastemaker quality with Bayesian hit-rate and lead-time."),
        ("alerts", "Creating scout inbox alerts for what changed now."),
        ("report", "Updating markdown/html scouting report artifacts."),
    ]


def _run_scan(ui_state: dict[str, Any], mode_override: str | None = None, include_backtest: bool | None = None) -> tuple[bool, dict[str, Any]]:
    runtime_config_path = _runtime_config(ui_state, DEFAULT_CONFIG_PATH)
    run_mode = mode_override or str(ui_state.get("run_mode", "auto"))
    run_with_backtest = bool(ui_state.get("run_with_backtest", False) if include_backtest is None else include_backtest)

    stage_messages = _pipeline_stage_messages()
    total_stages = len(stage_messages) + (1 if run_with_backtest else 0)

    progress = st.progress(0)
    stage_text = st.empty()
    log_box = st.container()
    logs: list[str] = []
    outputs: dict[str, Any] = {}

    runners = {
        "ingest": lambda: run_ingest(str(runtime_config_path), mode=run_mode, project_root=PROJECT_ROOT),
        "candidates": lambda: run_candidates(str(runtime_config_path), project_root=PROJECT_ROOT),
        "features": lambda: run_features(str(runtime_config_path), project_root=PROJECT_ROOT),
        "score": lambda: run_score(str(runtime_config_path), project_root=PROJECT_ROOT),
        "tastemakers": lambda: run_tastemakers(str(runtime_config_path), project_root=PROJECT_ROOT),
        "alerts": lambda: run_alerts(str(runtime_config_path), project_root=PROJECT_ROOT),
        "report": lambda: run_report(str(runtime_config_path), project_root=PROJECT_ROOT),
    }

    try:
        for index, (stage_key, stage_message) in enumerate(stage_messages, start=1):
            stage_text.markdown(f"**{index}/{total_stages}** {stage_message}")
            output = runners[stage_key]()
            outputs[stage_key] = output
            logs.append(f"- `{stage_key}` complete: {json.dumps(output, ensure_ascii=True)}")
            with log_box:
                st.markdown("\n".join(logs))
            progress.progress(index / total_stages)

        if run_with_backtest:
            idx = len(stage_messages) + 1
            stage_text.markdown(f"**{idx}/{total_stages}** Running historical replay backtest.")
            output = run_backtest(str(runtime_config_path), project_root=PROJECT_ROOT)
            outputs["backtest"] = output
            logs.append(f"- `backtest` complete: {json.dumps(output, ensure_ascii=True)}")
            with log_box:
                st.markdown("\n".join(logs))
            progress.progress(1.0)

        stage_text.markdown("**Scan complete.** Candidates and alerts are now refreshed.")
        st.cache_data.clear()
        return True, outputs
    except Exception as exc:
        logs.append(f"- error: {exc}")
        with log_box:
            st.markdown("\n".join(logs))
        stage_text.markdown("**Scan failed.** Review error details and try again.")
        return False, {"error": str(exc), "partial": outputs}


def _metric_card(label: str, value: str, subtitle: str = "") -> None:
    st.markdown(
        (
            "<div class='metric-card'>"
            f"<div class='metric-label'>{label}</div>"
            f"<div class='metric-value'>{value}</div>"
            f"<div class='small-note'>{subtitle}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _flag_list(row: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    for key in [
        "spike_only",
        "suspicious",
        "playlist_dependent",
        "established_artist",
        "insufficient_history",
        "weak_evidence",
        "low_trust",
    ]:
        if _bool(row.get(key)):
            flags.append(key)
    return flags


def _render_empty_state(title: str, body: str, cta_label: str, cta_key: str) -> bool:
    st.markdown(
        (
            "<div class='empty-state'>"
            f"<h3>{title}</h3>"
            f"<p class='help-copy'>{body}</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    return st.button(cta_label, key=cta_key, type="primary")


def _save_feedback(ui_state: dict[str, Any], track_ids: list[str], action: str) -> None:
    tracked = set(ui_state.get("tracked_track_ids", []))
    ignored = set(ui_state.get("ignored_track_ids", []))

    if action == "track":
        tracked.update(track_ids)
        ignored -= set(track_ids)
    elif action == "ignore":
        ignored.update(track_ids)
        tracked -= set(track_ids)

    ui_state["tracked_track_ids"] = sorted(tracked)
    ui_state["ignored_track_ids"] = sorted(ignored)
    save_ui_state(PROJECT_ROOT, ui_state)


def _set_thumb(ui_state: dict[str, Any], track_id: str, value: int) -> None:
    thumbs = dict(ui_state.get("thumbs", {}))
    thumbs[str(track_id)] = 1 if value > 0 else -1
    ui_state["thumbs"] = thumbs
    save_ui_state(PROJECT_ROOT, ui_state)


def _search_matches(outputs: dict[str, Any], query: str) -> list[dict[str, str]]:
    q = query.strip().lower()
    if not q:
        return []

    scored = outputs["scored"]
    tastemakers = outputs["tastemakers"]
    matches: list[dict[str, str]] = []

    if not scored.empty:
        subset = scored[
            scored["track_name"].str.lower().str.contains(q, na=False)
            | scored["artist_name"].str.lower().str.contains(q, na=False)
        ].head(8)
        for _, row in subset.iterrows():
            matches.append(
                {
                    "label": f"Track: {row['track_name']} - {row['artist_name']}",
                    "kind": "track",
                    "track_id": str(row["track_id"]),
                    "artist_id": str(row["artist_id"]),
                }
            )

    if not tastemakers.empty:
        subset = tastemakers[tastemakers["tastemaker_name"].str.lower().str.contains(q, na=False)].head(5)
        for _, row in subset.iterrows():
            matches.append(
                {
                    "label": f"Tastemaker: {row['tastemaker_name']}",
                    "kind": "tastemaker",
                    "track_id": "",
                    "artist_id": "",
                }
            )

    return matches


def _pick_default_track(outputs: dict[str, Any]) -> str:
    scored = outputs["scored"]
    if scored.empty:
        return ""
    return str(scored.iloc[0]["track_id"])


def _render_explainability_drawer(outputs: dict[str, Any], ui_state: dict[str, Any]) -> None:
    track_id = str(st.session_state.get("explain_track_id", "")).strip()
    if not track_id:
        return

    scored = outputs["scored"]
    inflections = outputs["inflections"]
    row = find_track_row(scored, track_id)
    if row is None:
        return

    with st.sidebar:
        st.markdown("### Explainability Drawer")
        st.markdown("Why this surfaced now, with the exact score components.")

        stage_html = stage_chip(str(row.get("stage", "early")))
        st.markdown(stage_html, unsafe_allow_html=True)
        st.caption(f"{row.get('track_name', '')} - {row.get('artist_name', '')}")

        contribution_rows = [
            {
                "Component": "Momentum engine composite",
                "Score": round(_float(row.get("weighted_score", 0.0)), 3),
                "Weight": round(_float(row.get("weight_momentum_engine", 1.0)), 3),
                "Contribution": round(
                    _float(row.get("weighted_score", 0.0)) * _float(row.get("weight_momentum_engine", 1.0)),
                    3,
                ),
            },
            {
                "Component": "Creative fit composite (40/30/15/15)",
                "Score": round(_float(row.get("creative_model_score", 0.0)), 3),
                "Weight": round(_float(row.get("weight_creative_fit", 0.0)), 3),
                "Contribution": round(
                    _float(row.get("creative_model_score", 0.0)) * _float(row.get("weight_creative_fit", 0.0)),
                    3,
                ),
            },
            {
                "Component": "Affinity prior",
                "Score": round(_float(row.get("affinity_score", 0.0)), 3),
                "Weight": round(_float(row.get("weight_affinity", 0.0)), 3),
                "Contribution": round(_float(row.get("affinity_boost", 0.0)), 3),
            },
            {
                "Component": "Path similarity prior",
                "Score": round(_float(row.get("path_similarity_score", 0.0)), 3),
                "Weight": round(_float(row.get("weight_path_similarity", 0.0)), 3),
                "Contribution": round(_float(row.get("path_similarity_boost", 0.0)), 3),
            },
        ]

        st.markdown("#### Score Breakdown")
        st.dataframe(pd.DataFrame(contribution_rows), use_container_width=True, hide_index=True)
        st.markdown(
            (
                f"`Weighted score: {float(row.get('weighted_score', 0.0)):.3f}`  "
                f"`Creative model: {float(row.get('creative_model_score', 0.0)):.3f}`  "
                f"`Blended model: {float(row.get('blended_model_score', row.get('weighted_score', 0.0))):.3f}`  "
                f"`Penalty: {float(row.get('penalty', 0.0)):.3f}`  "
                f"`Base final: {float(row.get('base_final_score', row.get('final_score', 0.0))):.3f}`  "
                f"`Prior boost: {float(row.get('prior_boost', 0.0)):.3f}`  "
                f"`Final: {float(row.get('final_score', 0.0)):.3f}`"
            )
        )
        if _bool(row.get("prior_boost_capped")):
            st.caption("Prior boost is capped until corroborated acceleration/depth strengthens.")

        st.markdown("#### Trust Score")
        st.markdown(
            (
                f"`Trust: {float(row.get('trust_score', 0.0)):.3f}`  "
                f"`Tier: {str(row.get('trust_tier', 'low'))}`  "
                f"`Base: {float(row.get('trust_base', 0.0)):.3f}`  "
                f"`Penalty: {float(row.get('trust_penalty', 0.0)):.3f}`"
            )
        )
        trust_rows = [
            {"Component": "History depth", "Score": round(_float(row.get("trust_history")), 3)},
            {"Component": "Evidence strength", "Score": round(_float(row.get("trust_evidence")), 3)},
            {"Component": "Corroboration", "Score": round(_float(row.get("trust_corroboration")), 3)},
            {"Component": "Data quality", "Score": round(_float(row.get("trust_data_quality")), 3)},
        ]
        st.dataframe(pd.DataFrame(trust_rows), use_container_width=True, hide_index=True)

        st.markdown("#### Why Now")
        st.markdown(str(row.get("explanation", "No explanation available.")))

        st.markdown("#### Affinity + Path Priors")
        if _float(row.get("prior_gate")) < 0.15:
            st.markdown(
                "- Prior impact is currently **downweighted** because evidence depth is still weak "
                "(insufficient history and/or weak acceleration/depth/resonance)."
            )
        st.markdown(
            (
                f"- Affinity score: `{_float(row.get('affinity_score')):.3f}`  "
                f"(match: `{str(row.get('affinity_match_artist', '') or 'none')}`)"
            )
        )
        st.markdown(
            (
                f"- Path similarity score: `{_float(row.get('path_similarity_score')):.3f}`  "
                f"(template: `{str(row.get('path_template_artist', '') or 'none')}`)"
            )
        )
        if str(row.get("path_template_path", "")).strip():
            st.markdown(f"- Template path: `{str(row.get('path_template_path', '')).strip()}`")
        if str(row.get("affinity_match_reason", "")).strip():
            st.markdown(f"- Affinity rationale: {str(row.get('affinity_match_reason', '')).strip()}")

        inflection_date = "Not detected"
        if not inflections.empty:
            selected = inflections[inflections["track_id"].astype(str) == track_id]
            if not selected.empty:
                inflection_date = str(selected.iloc[0].get("date", "Not detected"))
        elif _bool(row.get("inflection_detected")):
            inflection_date = str(row.get("latest_date", "Unknown"))

        st.markdown("#### Trigger Context")
        st.markdown(f"- Inflection date: `{inflection_date}`")
        st.markdown(f"- Cross-platform echo: `{_float(row.get('echo_score')):.3f}`")
        st.markdown(f"- Standardized resonance: `{_float(row.get('cross_platform_score')):.3f}`")
        st.markdown(f"- Short-form proxy lift: `{_float(row.get('shortform_proxy_score')):.3f}`")
        st.markdown(f"- Prior gate: `{_float(row.get('prior_gate')):.3f}`")
        st.markdown(f"- Affinity boost: `{_float(row.get('affinity_boost')):.3f}`")
        st.markdown(f"- Path similarity boost: `{_float(row.get('path_similarity_boost')):.3f}`")
        st.markdown(f"- Popping decision: `{str(_bool(row.get('popping_decision'))).lower()}`")
        st.markdown(f"- Popping confidence: `{_float(row.get('popping_confidence')):.3f}`")

        flags = _flag_list(row)
        st.markdown("#### Flags")
        if not flags:
            st.markdown("No risk flags on this candidate.")
        else:
            for flag in flags:
                st.markdown(f"- `{flag}`: {FLAG_LABELS[flag]}")

        st.markdown("#### Quick Feedback")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Thumbs Up", key=f"thumb_up_{track_id}"):
                _set_thumb(ui_state, track_id, 1)
                st.success("Saved positive feedback.")
        with c2:
            if st.button("Thumbs Down", key=f"thumb_down_{track_id}"):
                _set_thumb(ui_state, track_id, -1)
                st.success("Saved negative feedback.")

        if st.button("Close Drawer", key="close_explainability"):
            st.session_state["explain_track_id"] = ""
            st.rerun()


def _render_guided_tour(ui_state: dict[str, Any]) -> None:
    if _bool(ui_state.get("tour_dismissed")):
        return

    st.info(
        "First scan guided tour: 1) Filter by genre/stage, 2) open Explainability Drawer on top candidates, "
        "3) use Track/Ignore actions to train your watchlist."
    )
    if st.button("Dismiss guided tour", key="dismiss_tour"):
        ui_state["tour_dismissed"] = True
        save_ui_state(PROJECT_ROOT, ui_state)
        st.rerun()


def _render_home(outputs: dict[str, Any], ui_state: dict[str, Any]) -> None:
    scored = outputs["scored"].copy()
    snapshots = outputs["snapshots"]

    left, right = st.columns([0.72, 0.28])
    with left:
        st.title("Watchlist")
        st.caption("Before everyone else: low-base acceleration with explainable conviction.")
    with right:
        if st.button("Run New Scan", type="primary", key="home_run_scan"):
            ok, payload = _run_scan(ui_state)
            if ok:
                ui_state["onboarding_complete"] = True
                ui_state["onboarding_step"] = 3
                save_ui_state(PROJECT_ROOT, ui_state)
                st.success("Scan completed. Watchlist refreshed.")
                st.rerun()
            st.error(f"Scan failed: {payload.get('error', 'Unknown error')}")

    _render_guided_tour(ui_state)

    if scored.empty:
        should_scan = _render_empty_state(
            "No candidates yet.",
            "Run your first scan to generate a watchlist from starter tastemakers and open-source momentum signals.",
            "Run first scan",
            "empty_scan_home",
        )
        if should_scan:
            ok, payload = _run_scan(ui_state)
            if ok:
                st.success("Scan completed. Reloading watchlist.")
                st.rerun()
            st.error(f"Scan failed: {payload.get('error', 'Unknown error')}")
        return

    platform_map = track_platform_map(snapshots)
    region_map = track_region_map(snapshots)
    spark_map = track_sparkline_map(snapshots)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _metric_card("Candidates", f"{len(scored):,}", "Scored this run")
    with c2:
        _metric_card("Breaking", f"{len(scored[scored['stage'] == 'breaking']):,}", "High-confidence inflections")
    with c3:
        inflections = int(scored["inflection_detected"].map(_bool).sum()) if "inflection_detected" in scored else 0
        _metric_card("Inflections", f"{inflections:,}", "Trigger fired")
    with c4:
        tracked = len(ui_state.get("tracked_track_ids", []))
        _metric_card("Tracked", f"{tracked:,}", "In active scout queue")

    all_genres = sorted({str(x) for x in scored.get("genre", pd.Series(dtype=str)).dropna().tolist()})
    stage_order = ["baseline", "early", "emerging", "breaking"]
    all_platforms = sorted({platform for items in platform_map.values() for platform in items})
    all_regions = sorted({region for items in region_map.values() for region in items})

    default_genres = [g for g in ui_state.get("genre_focus", DEFAULT_GENRES) if g in all_genres] or all_genres

    f1, f2, f3, f4 = st.columns(4)
    selected_genres = f1.multiselect("Genre", all_genres, default=default_genres)
    selected_stages = f2.multiselect("Stage", stage_order, default=["early", "emerging", "breaking"])
    selected_platforms = f3.multiselect("Platform", all_platforms, default=all_platforms)
    selected_regions = f4.multiselect("Region", all_regions, default=[])

    hide_defaults = ui_state.get("hide_flags_default", {})
    flag1, flag2, flag3, flag4, flag5, flag6 = st.columns(6)
    hide_spike = flag1.checkbox("Hide spike-only", value=bool(hide_defaults.get("spike_only", True)))
    hide_suspicious = flag2.checkbox("Hide suspicious", value=bool(hide_defaults.get("suspicious", True)))
    hide_playlist = flag3.checkbox(
        "Hide playlist-dependent",
        value=bool(hide_defaults.get("playlist_dependent", False)),
    )
    hide_established = flag4.checkbox("Hide established artists", value=bool(hide_defaults.get("established_artist", True)))
    hide_baseline = flag5.checkbox("Hide baseline-only", value=bool(hide_defaults.get("insufficient_history", True)))
    hide_low_trust = flag6.checkbox("Hide low-trust", value=bool(hide_defaults.get("low_trust", True)))

    s1, s2, s3 = st.columns(3)
    min_final_score = s1.slider("Min final score", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    min_trust_score = s2.slider("Min trust score", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    min_prior_gate = s3.slider("Min prior gate", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

    prior1, prior2 = st.columns(2)
    min_affinity = prior1.slider("Min affinity prior", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    min_path_similarity = prior2.slider("Min path similarity", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

    filtered_rows: list[dict[str, Any]] = []
    for row in scored.to_dict(orient="records"):
        track_id = str(row.get("track_id", ""))
        if selected_genres and str(row.get("genre", "")) not in selected_genres:
            continue
        if selected_stages and str(row.get("stage", "")) not in selected_stages:
            continue

        track_platforms = set(platform_map.get(track_id, []))
        if selected_platforms and not track_platforms.intersection(set(selected_platforms)):
            continue

        track_regions = set(region_map.get(track_id, []))
        if selected_regions and not track_regions.intersection(set(selected_regions)):
            continue

        if hide_spike and _bool(row.get("spike_only")):
            continue
        if hide_suspicious and _bool(row.get("suspicious")):
            continue
        if hide_playlist and _bool(row.get("playlist_dependent")):
            continue
        if hide_established and _bool(row.get("established_artist")):
            continue
        if hide_baseline and _bool(row.get("insufficient_history")):
            continue
        if hide_low_trust and _bool(row.get("low_trust")):
            continue
        if _float(row.get("affinity_score", 0.0)) < min_affinity:
            continue
        if _float(row.get("path_similarity_score", 0.0)) < min_path_similarity:
            continue
        if _float(row.get("final_score", 0.0)) < min_final_score:
            continue
        if _float(row.get("trust_score", 0.0)) < min_trust_score:
            continue
        if _float(row.get("prior_gate", 0.0)) < min_prior_gate:
            continue

        filtered_rows.append(row)

    filtered = pd.DataFrame(filtered_rows)
    if filtered.empty:
        baseline_only = bool(
            not scored.empty
            and "insufficient_history" in scored
            and scored["insufficient_history"].map(_bool).all()
        )
        if baseline_only:
            st.warning(
                "Baseline collection mode: not enough history yet for reliable inflection scoring. "
                "Run daily scans for several days, then disable 'Hide baseline-only'."
            )
        else:
            st.warning("No candidates match your filters. Reset one filter group to expand the watchlist.")
        return

    filtered = filtered.sort_values("final_score", ascending=False)
    filtered["spark"] = filtered["track_id"].map(lambda tid: spark_map.get(str(tid), ""))
    filtered["flags"] = filtered.apply(
        lambda row: ", ".join(_flag_list(row.to_dict())) if _flag_list(row.to_dict()) else "clean",
        axis=1,
    )
    filtered["why_now"] = filtered["explanation"].astype(str).str.slice(0, 120)

    display_columns = [
        "track_id",
        "track_name",
        "artist_name",
        "genre",
        "stage",
        "final_score",
        "trust_score",
        "affinity_score",
        "path_similarity_score",
        "spark",
        "flags",
        "why_now",
    ]
    display = filtered[display_columns].copy()
    display["select"] = display["track_id"].isin(set(ui_state.get("tracked_track_ids", [])))

    st.markdown("### Scout Queue")
    edited = st.data_editor(
        display,
        hide_index=True,
        use_container_width=True,
        disabled=[
            "track_id",
            "track_name",
            "artist_name",
            "genre",
            "stage",
            "final_score",
            "trust_score",
            "spark",
            "flags",
            "why_now",
        ],
        column_config={
            "select": st.column_config.CheckboxColumn("Select"),
            "final_score": st.column_config.NumberColumn("score", format="%.3f"),
            "trust_score": st.column_config.NumberColumn("trust", format="%.3f"),
            "affinity_score": st.column_config.NumberColumn("affinity", format="%.3f"),
            "path_similarity_score": st.column_config.NumberColumn("path_sim", format="%.3f"),
            "spark": st.column_config.TextColumn("sparkline"),
        },
    )

    selected_track_ids = edited[edited["select"]]["track_id"].astype(str).tolist()

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Track Selected", key="bulk_track", type="primary", use_container_width=True):
            _save_feedback(ui_state, selected_track_ids, "track")
            st.success(f"Tracking {len(selected_track_ids)} track(s).")
            st.rerun()
    with b2:
        if st.button("Ignore Selected", key="bulk_ignore", use_container_width=True):
            _save_feedback(ui_state, selected_track_ids, "ignore")
            st.success(f"Ignored {len(selected_track_ids)} track(s).")
            st.rerun()
    with b3:
        csv_data = filtered[filtered["track_id"].isin(selected_track_ids)].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export Selected",
            data=csv_data,
            file_name="watchlist_selection.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("### Candidate Explainability")
    options = ["Select a candidate"] + [
        f"{row['track_name']} - {row['artist_name']}" for _, row in filtered.head(50).iterrows()
    ]
    choice = st.selectbox("Open explainability drawer", options)
    if choice != "Select a candidate" and st.button("Open Explainability", key="open_explainability", type="primary"):
        selected = filtered[filtered.apply(lambda r: f"{r['track_name']} - {r['artist_name']}" == choice, axis=1)].iloc[0]
        st.session_state["explain_track_id"] = str(selected["track_id"])
        st.session_state["selected_track_id"] = str(selected["track_id"])
        st.session_state["selected_artist_id"] = str(selected.get("artist_id", ""))
        st.rerun()

    st.markdown("### What To Do Next")
    st.markdown(
        "1. Track the top 3 emerging candidates with clean flags.  "
        "2. Open Explainability Drawer to validate corroborated buckets.  "
        "3. Ignore spike-only candidates to tighten future prioritization."
    )


def _render_track_detail(outputs: dict[str, Any], ui_state: dict[str, Any]) -> None:
    scored = outputs["scored"]
    snapshots = outputs["snapshots"]

    st.title("Track Detail")
    st.caption("Single-track decision view with depth, resonance, and risk context.")

    if scored.empty:
        st.warning("No tracks available yet. Run a scan from Home.")
        return

    track_options = scored.sort_values("final_score", ascending=False)
    labels = [f"{row['track_name']} - {row['artist_name']}" for _, row in track_options.iterrows()]
    mapping = {label: str(track_options.iloc[idx]["track_id"]) for idx, label in enumerate(labels)}

    default_track = st.session_state.get("selected_track_id") or _pick_default_track(outputs)
    default_label = labels[0]
    for label, track_id in mapping.items():
        if str(track_id) == str(default_track):
            default_label = label
            break

    selected_label = st.selectbox("Track", labels, index=labels.index(default_label) if default_label in labels else 0)
    track_id = mapping[selected_label]
    st.session_state["selected_track_id"] = track_id

    row = find_track_row(scored, track_id)
    if row is None:
        st.warning("Unable to locate track row.")
        return

    stage_html = stage_chip(str(row.get("stage", "early")))
    st.markdown(stage_html, unsafe_allow_html=True)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        _metric_card("Final Score", f"{_float(row.get('final_score')):.3f}")
    with m2:
        _metric_card("Acceleration", f"{_float(row.get('size_norm_accel')):.3f}")
    with m3:
        _metric_card("Depth", f"{_float(row.get('depth_score')):.3f}")
    with m4:
        _metric_card("Echo", f"{_float(row.get('cross_platform_score')):.3f}")
    with m5:
        _metric_card("Prior Boost", f"{_float(row.get('prior_boost')):.3f}")
    with m6:
        _metric_card("Trust", f"{_float(row.get('trust_score')):.3f}", str(row.get("trust_tier", "low")))

    a1, a2, a3 = st.columns(3)
    with a1:
        if st.button("Track This", key=f"track_btn_{track_id}", type="primary"):
            _save_feedback(ui_state, [track_id], "track")
            st.success("Added to tracked watchlist.")
    with a2:
        if st.button("Ignore", key=f"ignore_btn_{track_id}"):
            _save_feedback(ui_state, [track_id], "ignore")
            st.success("Marked as ignored.")
    with a3:
        if st.button("Open Explainability Drawer", key=f"drawer_btn_{track_id}"):
            st.session_state["explain_track_id"] = track_id
            st.rerun()

    st.markdown("### Momentum Timeline")
    series = track_timeseries(snapshots, track_id)
    if not series.empty:
        pivot = series.pivot(index="date", columns="platform", values="value").fillna(0.0)
        st.line_chart(pivot)
    else:
        st.info("No timeline available for this track yet.")

    st.markdown("### Top Features")
    feature_rows = [
        ("Momentum", _float(row.get("momentum_score"))),
        ("Acceleration", _float(row.get("acceleration_score"))),
        ("Depth", _float(row.get("depth_score"))),
        ("Cross-platform", _float(row.get("cross_platform_score"))),
        ("Network", _float(row.get("network_score"))),
        ("Consistency", _float(row.get("consistency_score"))),
        ("Geo", _float(row.get("geo_score"))),
        ("Sonic palette", _float(row.get("sonic_palette_score"))),
        ("Persona/brand", _float(row.get("persona_brand_score"))),
        ("Writing", _float(row.get("writing_score"))),
        ("Market position", _float(row.get("market_position_score"))),
        ("Creative model", _float(row.get("creative_model_score"))),
        ("Trust", _float(row.get("trust_score"))),
        ("Affinity prior", _float(row.get("affinity_score"))),
        ("Path similarity prior", _float(row.get("path_similarity_score"))),
        ("Comment specificity", _float(row.get("comment_specificity"))),
        ("Short-form proxy", _float(row.get("shortform_proxy_score"))),
    ]
    feature_df = pd.DataFrame(feature_rows, columns=["Feature", "Score"]).sort_values("Score", ascending=False)
    st.dataframe(feature_df, hide_index=True, use_container_width=True)

    st.markdown("### Affinity + Path Priors")
    st.markdown(
        f"- Affinity match artist: `{str(row.get('affinity_match_artist', '') or 'none')}` (score `{_float(row.get('affinity_score')):.3f}`)"
    )
    st.markdown(
        f"- Path template: `{str(row.get('path_template_artist', '') or 'none')}` (score `{_float(row.get('path_similarity_score')):.3f}`)"
    )
    if str(row.get("path_template_path", "")).strip():
        st.markdown(f"- Template path summary: `{str(row.get('path_template_path', '')).strip()}`")
    if str(row.get("affinity_match_reason", "")).strip():
        st.markdown(f"- Affinity rationale: {str(row.get('affinity_match_reason', '')).strip()}")

    st.markdown("### Trust Breakdown")
    st.markdown(
        (
            f"- Trust score: `{_float(row.get('trust_score')):.3f}` (`{str(row.get('trust_tier', 'low'))}`)\n"
            f"- History depth: `{_float(row.get('trust_history')):.3f}`\n"
            f"- Evidence strength: `{_float(row.get('trust_evidence')):.3f}`\n"
            f"- Corroboration: `{_float(row.get('trust_corroboration')):.3f}`\n"
            f"- Data quality: `{_float(row.get('trust_data_quality')):.3f}`\n"
            f"- Trust penalty: `{_float(row.get('trust_penalty')):.3f}`"
        )
    )

    st.markdown("### Comment Specificity Evidence")
    comments: list[str] = []
    if not snapshots.empty and "comments_text" in snapshots.columns:
        track_rows = snapshots[snapshots["track_id"].astype(str) == track_id]
        for raw in track_rows.get("comments_text", []):
            if isinstance(raw, list):
                comments.extend([str(item) for item in raw if str(item).strip()])

    if comments:
        comments = comments[:30]
        terms = outputs["config"]["features"]["depth"]["comment_specificity_terms"]
        comment_rows: list[dict[str, Any]] = []
        for text in comments:
            lowered = text.lower()
            matched = [term for term in terms if term in lowered]
            comment_rows.append(
                {
                    "comment": text,
                    "matched_terms": ", ".join(matched) if matched else "none",
                    "specific": "yes" if matched else "no",
                }
            )

        st.dataframe(pd.DataFrame(comment_rows), use_container_width=True, hide_index=True)
        st.caption(
            "Specificity score is the share of comments containing lyrical/emotional repeat-listening terms from config."
        )
    else:
        st.info("No comment samples available in current snapshots.")

    st.markdown("### What To Do Next")
    st.markdown("Use this page to decide: track for daily monitoring, or ignore to reduce noisy recall.")


def _render_artist_detail(outputs: dict[str, Any], ui_state: dict[str, Any]) -> None:
    scored = outputs["scored"]
    snapshots = outputs["snapshots"]

    st.title("Artist Detail")
    st.caption("Artist-level view across tracks, channels, and tastemaker touchpoints.")

    if scored.empty:
        st.warning("No artists available yet. Run a scan first.")
        return

    artists = scored[["artist_id", "artist_name"]].drop_duplicates().sort_values("artist_name")
    labels = [f"{row['artist_name']} ({row['artist_id']})" for _, row in artists.iterrows()]
    selected_artist_id = st.session_state.get("selected_artist_id")

    default_label = labels[0]
    if selected_artist_id:
        for label in labels:
            if label.endswith(f"({selected_artist_id})"):
                default_label = label
                break

    selected_label = st.selectbox("Artist", labels, index=labels.index(default_label) if default_label in labels else 0)
    artist_id = selected_label.split("(")[-1].replace(")", "").strip()
    st.session_state["selected_artist_id"] = artist_id

    artist_tracks = scored[scored["artist_id"].astype(str) == artist_id].sort_values("final_score", ascending=False)
    if artist_tracks.empty:
        st.info("No tracks found for this artist.")
        return

    best_score = artist_tracks["final_score"].max()
    best_stage = artist_tracks.iloc[0]["stage"]
    tracked_count = sum(1 for tid in artist_tracks["track_id"].astype(str) if tid in set(ui_state.get("tracked_track_ids", [])))

    c1, c2, c3 = st.columns(3)
    with c1:
        _metric_card("Top Track Score", f"{best_score:.3f}")
    with c2:
        _metric_card("Current Stage", str(best_stage))
    with c3:
        _metric_card("Tracked Tracks", str(tracked_count))

    st.markdown("### Artist Track Stack")
    show = artist_tracks[
        [
            "track_id",
            "track_name",
            "genre",
            "stage",
            "final_score",
            "trust_score",
            "affinity_score",
            "path_similarity_score",
            "inflection_detected",
            "explanation",
        ]
    ].copy()
    show["final_score"] = show["final_score"].map(lambda value: round(float(value), 3))
    st.dataframe(show, use_container_width=True, hide_index=True)

    st.markdown("### Platform Momentum")
    series = artist_timeseries(snapshots, artist_id)
    if not series.empty:
        pivot = series.pivot(index="date", columns="platform", values="value").fillna(0.0)
        st.line_chart(pivot)
    else:
        st.info("No platform history available for this artist.")

    st.markdown("### Tastemaker Touchpoints")
    if snapshots.empty or "tastemaker_id" not in snapshots.columns:
        st.info("No tastemaker touchpoints found.")
    else:
        rows = snapshots[
            (snapshots["artist_id"].astype(str) == artist_id)
            & snapshots["tastemaker_id"].notna()
            & (snapshots["tastemaker_id"].astype(str) != "")
        ]
        if rows.empty:
            st.info("No tastemaker touchpoints found.")
        else:
            grouped = (
                rows.groupby(["tastemaker_id", "tastemaker_name"], as_index=False)
                .agg(mentions=("track_id", "count"), first_seen=("date", "min"), last_seen=("date", "max"))
                .sort_values("mentions", ascending=False)
            )
            st.dataframe(grouped, use_container_width=True, hide_index=True)

    st.markdown("### What To Do Next")
    st.markdown("Open Track Detail for the highest-score song and validate whether momentum is broad or source-concentrated.")


def _render_tastemakers(outputs: dict[str, Any], ui_state: dict[str, Any]) -> None:
    st.title("Tastemakers")
    st.caption("Manage sources and evaluate tastemaker quality using quantified hit-rate + lead-time.")

    tastemakers = outputs["tastemakers"].copy()
    if tastemakers.empty:
        st.info("No tastemaker profiles yet. Run a scan to populate this page.")
    else:
        tastemakers = tastemakers.sort_values("quant_score", ascending=False)
        st.dataframe(
            tastemakers[
                [
                    "tastemaker_name",
                    "status",
                    "quant_score",
                    "bayes_precision",
                    "avg_lead_days",
                    "trials",
                    "successes",
                    "genre_alignment",
                    "reliability",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    source_registry = outputs.get("source_registry", {})
    source_counts = {
        "YouTube": len(source_registry.get("youtube_channels", [])),
        "Spotify": len(source_registry.get("spotify_playlists", [])),
        "Reddit": len(source_registry.get("reddit_subreddits", [])),
        "Last.fm": len(source_registry.get("lastfm_tags", [])),
        "RSS": len(source_registry.get("rss_feeds", [])),
    }

    st.markdown("### Starter Pack Coverage")
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, (label, count) in zip([c1, c2, c3, c4, c5], source_counts.items()):
        with col:
            _metric_card(label, str(count), "starter sources")

    st.markdown("### Add Tastemaker (Custom Layer)")
    st.caption("Primary action: add one source. Custom sources are merged with starter pack in Custom mode.")

    with st.form("add_source_form"):
        source_type_label = st.selectbox("Source type", list(SOURCE_TYPE_OPTIONS.keys()))
        source_type = SOURCE_TYPE_OPTIONS[source_type_label]
        name = st.text_input("Display name")
        identifier = st.text_input("Identifier (channel id / playlist id / subreddit / tag / feed id)")
        url = st.text_input("URL (required for RSS only)")
        genre_tags = st.multiselect("Genre tags", DEFAULT_GENRES, default=ui_state.get("genre_focus", DEFAULT_GENRES))
        submitted = st.form_submit_button("Add Tastemaker", type="primary")

    if submitted:
        custom = copy.deepcopy(ui_state.get("custom_sources", {}))
        custom.setdefault(source_type, [])

        entry: dict[str, Any] = {"genre_tags": genre_tags}
        if source_type == "youtube_channels":
            entry.update({"channel_id": identifier.strip(), "name": name.strip() or identifier.strip(), "estimated_followers": 0})
        elif source_type == "spotify_playlists":
            entry.update({"playlist_id": identifier.strip(), "name": name.strip() or identifier.strip(), "region": "United States"})
        elif source_type == "reddit_subreddits":
            entry.update({"name": identifier.strip()})
        elif source_type == "lastfm_tags":
            entry.update({"name": identifier.strip()})
        elif source_type == "rss_feeds":
            entry.update({"id": identifier.strip(), "name": name.strip() or identifier.strip(), "url": url.strip()})

        if source_type == "rss_feeds" and not url.strip():
            st.error("RSS feeds require a URL.")
        elif not identifier.strip():
            st.error("Identifier is required.")
        else:
            custom[source_type].append(entry)
            ui_state["custom_sources"] = custom
            save_ui_state(PROJECT_ROOT, ui_state)
            st.success("Custom tastemaker added.")
            st.rerun()

    st.markdown("### Custom Source Layer")
    custom_sources = ui_state.get("custom_sources", {})
    present_rows: list[dict[str, str]] = []
    for key, entries in custom_sources.items():
        for entry in entries:
            identifier = ""
            for candidate in ["channel_id", "playlist_id", "id", "name"]:
                if entry.get(candidate):
                    identifier = str(entry.get(candidate))
                    break
            present_rows.append({"type": key, "identifier": identifier, "name": str(entry.get("name", identifier))})

    if present_rows:
        custom_df = pd.DataFrame(present_rows)
        st.dataframe(custom_df, use_container_width=True, hide_index=True)
        remove_options = [f"{row['type']}::{row['identifier']}" for row in present_rows]
        remove_target = st.selectbox("Remove source", ["None"] + remove_options)
        if remove_target != "None" and st.button("Remove selected source", key="remove_source"):
            source_type, identifier = remove_target.split("::", 1)
            rows = []
            for item in custom_sources.get(source_type, []):
                item_identifier = (
                    item.get("channel_id")
                    or item.get("playlist_id")
                    or item.get("id")
                    or item.get("name")
                    or ""
                )
                if str(item_identifier) != identifier:
                    rows.append(item)
            custom_sources[source_type] = rows
            ui_state["custom_sources"] = custom_sources
            save_ui_state(PROJECT_ROOT, ui_state)
            st.success("Custom source removed.")
            st.rerun()
    else:
        st.info("No custom sources yet.")

    st.markdown("### What To Do Next")
    st.markdown("Switch to Custom mode in Settings, add 5-10 niche sources, then run a new scan to score their early-hit quality.")


def _render_alerts(outputs: dict[str, Any], ui_state: dict[str, Any]) -> None:
    st.title("Alerts")
    st.caption("What changed since the last run and why it matters now.")

    alerts = outputs["alerts"].copy()
    if alerts.empty:
        should_run = _render_empty_state(
            "No alerts yet.",
            "Generate alerts from current scored outputs in one click.",
            "Generate alerts now",
            "generate_alerts_btn",
        )
        if should_run:
            runtime_config_path = _runtime_config(ui_state, DEFAULT_CONFIG_PATH)
            try:
                payload = run_alerts(str(runtime_config_path), project_root=PROJECT_ROOT)
                st.success(f"Generated {payload.get('alerts', 0)} alerts.")
                st.cache_data.clear()
                st.rerun()
            except Exception as exc:
                st.error(f"Alert generation failed: {exc}")
        return

    alerts = alerts.sort_values(["priority", "score"], ascending=[True, False])
    st.dataframe(
        alerts[
            [
                "generated_at",
                "priority",
                "type",
                "track_name",
                "artist_name",
                "genre",
                "stage",
                "score",
                "reason",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### What To Do Next")
    st.markdown("Prioritize priority-0 alerts, then open Track Detail to validate depth and anti-gaming flags before outreach.")


def _render_backtest(outputs: dict[str, Any], ui_state: dict[str, Any]) -> None:
    st.title("Backtest / History")
    st.caption("Historical replay metrics for precision, recall, lead time, and list stability.")

    backtest = outputs.get("backtest", {})
    summary = backtest.get("summary") if isinstance(backtest, dict) else None

    if not isinstance(summary, dict):
        run_now = _render_empty_state(
            "No backtest output yet.",
            "Run a replay against stored snapshots to benchmark hit quality and lead time.",
            "Run backtest",
            "run_backtest_btn",
        )
        if run_now:
            runtime_config_path = _runtime_config(ui_state, DEFAULT_CONFIG_PATH)
            try:
                payload = run_backtest(str(runtime_config_path), project_root=PROJECT_ROOT)
                st.success(f"Backtest complete across {payload['summary']['windows']} windows.")
                st.cache_data.clear()
                st.rerun()
            except Exception as exc:
                st.error(f"Backtest failed: {exc}")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        _metric_card("Precision@k", f"{float(summary.get('precision_at_k', 0.0)):.3f}")
    with c2:
        _metric_card("Recall@k", f"{float(summary.get('recall_at_k', 0.0)):.3f}")
    with c3:
        _metric_card("Lead Time", f"{float(summary.get('lead_time_days', 0.0)):.1f}d")
    with c4:
        _metric_card("False Pos Rate", f"{float(summary.get('false_positive_rate', 0.0)):.3f}")
    with c5:
        _metric_card("Stability", f"{float(summary.get('week_to_week_stability', 0.0)):.3f}")

    windows = backtest.get("windows", []) if isinstance(backtest, dict) else []
    if windows:
        st.markdown("### Replay Windows")
        frame = pd.DataFrame(windows)
        st.dataframe(frame, use_container_width=True, hide_index=True)

    st.markdown("### What To Do Next")
    st.markdown("Tune thresholds only after reviewing precision vs lead-time tradeoff in at least 8 replay windows.")


def _render_settings(outputs: dict[str, Any], ui_state: dict[str, Any]) -> None:
    st.title("Settings")
    st.caption("Advanced controls. Defaults are already tuned for first-use success.")

    with st.form("settings_form"):
        run_mode = st.selectbox("Run mode", ["auto", "mock", "hybrid"], index=["auto", "mock", "hybrid"].index(ui_state.get("run_mode", "auto")))
        source_mode = st.radio(
            "Discovery sources",
            ["starter_pack", "custom"],
            index=0 if ui_state.get("discovery_source_mode", "starter_pack") == "starter_pack" else 1,
            format_func=lambda value: "Starter Pack (Recommended)" if value == "starter_pack" else "Custom",
        )
        run_with_backtest = st.checkbox("Run backtest on each scan", value=bool(ui_state.get("run_with_backtest", False)))
        genre_focus = st.multiselect("Genre focus", DEFAULT_GENRES, default=ui_state.get("genre_focus", DEFAULT_GENRES))

        st.markdown("#### Default Flag Filters")
        defaults = ui_state.get("hide_flags_default", {})
        hide_spike = st.checkbox("Hide spike-only by default", value=bool(defaults.get("spike_only", True)))
        hide_suspicious = st.checkbox("Hide suspicious by default", value=bool(defaults.get("suspicious", True)))
        hide_playlist = st.checkbox(
            "Hide playlist-dependent by default",
            value=bool(defaults.get("playlist_dependent", False)),
        )
        hide_established = st.checkbox(
            "Hide established artists by default",
            value=bool(defaults.get("established_artist", True)),
        )
        hide_baseline = st.checkbox(
            "Hide baseline-only by default",
            value=bool(defaults.get("insufficient_history", True)),
        )
        hide_low_trust = st.checkbox(
            "Hide low-trust by default",
            value=bool(defaults.get("low_trust", True)),
        )

        save = st.form_submit_button("Save settings", type="primary")

    if save:
        ui_state["run_mode"] = run_mode
        ui_state["discovery_source_mode"] = source_mode
        ui_state["run_with_backtest"] = run_with_backtest
        ui_state["genre_focus"] = genre_focus or list(DEFAULT_GENRES)
        ui_state["hide_flags_default"] = {
            "spike_only": hide_spike,
            "suspicious": hide_suspicious,
            "playlist_dependent": hide_playlist,
            "established_artist": hide_established,
            "insufficient_history": hide_baseline,
            "low_trust": hide_low_trust,
        }
        save_ui_state(PROJECT_ROOT, ui_state)
        st.success("Settings saved.")

    st.markdown("### Runtime Files")
    st.markdown(f"- Project root: `{PROJECT_ROOT}`")
    st.markdown(f"- Base config: `{DEFAULT_CONFIG_PATH}`")
    st.markdown(f"- Runtime config: `{RUNTIME_CONFIG_PATH}`")

    st.markdown("### What To Do Next")
    st.markdown("If you switched to Custom mode, add tastemakers in Tastemakers page and run a new scan from Home.")


def _run_onboarding_scan(ui_state: dict[str, Any], mode: str) -> bool:
    st.markdown("### First Scan Progress")
    ok, payload = _run_scan(ui_state, mode_override=mode, include_backtest=False)
    if ok:
        ui_state["onboarding_complete"] = True
        ui_state["onboarding_step"] = 3
        save_ui_state(PROJECT_ROOT, ui_state)
        st.success("First scan complete. Opening your watchlist.")
        return True
    st.error(f"Scan failed: {payload.get('error', 'Unknown error')}.")
    return False


def _render_onboarding(ui_state: dict[str, Any]) -> bool:
    step = int(st.session_state.get("onboarding_step", ui_state.get("onboarding_step", 1)))
    step = max(1, min(3, step))

    st.title("Welcome to Talent Scouting Intelligence")
    st.caption("Treat this like a consumer-grade product: premium polish with scout-grade rigor.")

    if step == 1:
        st.markdown("### Step 1 of 3: Choose genre focus")
        st.markdown("**Screen copy:** `Pick the lanes you care about. We preloaded your core scouting genres.`")

        genres = st.multiselect(
            "Genre focus",
            DEFAULT_GENRES,
            default=ui_state.get("genre_focus", DEFAULT_GENRES),
        )

        c1, c2 = st.columns([0.6, 0.4])
        with c1:
            if st.button("Continue", type="primary"):
                ui_state["genre_focus"] = genres or list(DEFAULT_GENRES)
                ui_state["onboarding_step"] = 2
                save_ui_state(PROJECT_ROOT, ui_state)
                st.session_state["onboarding_step"] = 2
                st.rerun()
        with c2:
            if st.button("Skip and use defaults"):
                ui_state["genre_focus"] = list(DEFAULT_GENRES)
                ui_state["discovery_source_mode"] = "starter_pack"
                ui_state["onboarding_complete"] = True
                ui_state["onboarding_step"] = 3
                save_ui_state(PROJECT_ROOT, ui_state)
                return True

    elif step == 2:
        st.markdown("### Step 2 of 3: Pick discovery sources")
        st.markdown("**Screen copy:** `Choose how to source candidates. Starter Pack is curated and works instantly.`")

        source_mode = st.radio(
            "Discovery sources",
            ["starter_pack", "custom"],
            index=0 if ui_state.get("discovery_source_mode", "starter_pack") == "starter_pack" else 1,
            format_func=lambda v: "Starter Pack (recommended)" if v == "starter_pack" else "Custom",
        )

        if source_mode == "custom":
            st.info("You can add custom tastemakers later in the Tastemakers page. No setup required right now.")

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Back"):
                ui_state["onboarding_step"] = 1
                save_ui_state(PROJECT_ROOT, ui_state)
                st.session_state["onboarding_step"] = 1
                st.rerun()
        with c2:
            if st.button("Continue", type="primary"):
                ui_state["discovery_source_mode"] = source_mode
                ui_state["onboarding_step"] = 3
                save_ui_state(PROJECT_ROOT, ui_state)
                st.session_state["onboarding_step"] = 3
                st.rerun()
        with c3:
            if st.button("Skip and use Starter Pack"):
                ui_state["discovery_source_mode"] = "starter_pack"
                ui_state["onboarding_step"] = 3
                save_ui_state(PROJECT_ROOT, ui_state)
                st.session_state["onboarding_step"] = 3
                st.rerun()

    elif step == 3:
        st.markdown("### Step 3 of 3: Run first scan")
        st.markdown("**Screen copy:** `We’re about to scan, score, and explain your first scout queue.`")

        mode = st.selectbox(
            "Scan mode",
            ["auto", "mock", "hybrid"],
            index=["auto", "mock", "hybrid"].index(ui_state.get("run_mode", "auto")),
            help="Auto uses free live adapters + fallback. Mock is instant demo data.",
        )

        st.caption(
            "If you skip here, the app still opens with any existing demo outputs and remains fully usable."
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Run First Scan", type="primary"):
                ui_state["run_mode"] = mode
                save_ui_state(PROJECT_ROOT, ui_state)
                if _run_onboarding_scan(ui_state, mode):
                    st.rerun()
        with c2:
            if st.button("Back"):
                ui_state["onboarding_step"] = 2
                save_ui_state(PROJECT_ROOT, ui_state)
                st.session_state["onboarding_step"] = 2
                st.rerun()
        with c3:
            if st.button("Skip and open app"):
                ui_state["onboarding_complete"] = True
                ui_state["onboarding_step"] = 3
                save_ui_state(PROJECT_ROOT, ui_state)
                return True

    return bool(ui_state.get("onboarding_complete", False))


def _render_sidebar(outputs: dict[str, Any], ui_state: dict[str, Any]) -> None:
    st.sidebar.markdown("## Scout Navigator")
    page = st.sidebar.radio("Page", PAGES, index=PAGES.index(st.session_state.get("page", PAGES[0])))
    st.session_state["page"] = page

    query = st.sidebar.text_input("Search artist / track / tastemaker")
    matches = _search_matches(outputs, query)
    if matches:
        labels = [entry["label"] for entry in matches]
        picked = st.sidebar.selectbox("Jump to result", ["Select result"] + labels)
        if picked != "Select result":
            target = next(item for item in matches if item["label"] == picked)
            if target["kind"] == "track":
                st.session_state["selected_track_id"] = target["track_id"]
                st.session_state["selected_artist_id"] = target["artist_id"]
                st.session_state["page"] = "Track Detail"
                st.session_state["explain_track_id"] = target["track_id"]
                st.rerun()
            else:
                st.session_state["page"] = "Tastemakers"
                st.rerun()

    st.sidebar.markdown("---")
    st.session_state["load_snapshots"] = st.sidebar.checkbox(
        "Load timeline history",
        value=bool(st.session_state.get("load_snapshots", False)),
        help="Off by default for fast startup on cloud-synced folders.",
    )
    st.sidebar.caption("Every recommendation is explainable: open the Explainability Drawer from Home or Track Detail.")


def main() -> None:
    st.set_page_config(
        page_title="Talent Scouting Intelligence",
        page_icon="\U0001F3A7",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()

    ui_state = load_ui_state(PROJECT_ROOT)
    _set_defaults(ui_state)

    if not _bool(ui_state.get("onboarding_complete", False)):
        completed = _render_onboarding(ui_state)
        if completed:
            ui_state["onboarding_complete"] = True
            save_ui_state(PROJECT_ROOT, ui_state)
            st.rerun()
        return

    @st.cache_data(ttl=20, show_spinner=False)
    def _load_cached_outputs(config_path: str, root_path: str) -> dict[str, Any]:
        # Keep first paint fast and robust, especially on cloud-synced folders.
        # Snapshot history can be loaded separately without blocking app boot.
        return load_outputs(config_path, Path(root_path), load_snapshots=False)

    @st.cache_data(ttl=20, show_spinner=False)
    def _load_cached_snapshots(config_path: str, root_path: str) -> pd.DataFrame:
        return load_snapshots(config_path, Path(root_path))

    outputs = _load_cached_outputs(str(DEFAULT_CONFIG_PATH), str(PROJECT_ROOT))

    _render_sidebar(outputs, ui_state)
    if bool(st.session_state.get("load_snapshots", False)):
        with st.spinner("Loading timeline history..."):
            outputs["snapshots"] = _load_cached_snapshots(str(DEFAULT_CONFIG_PATH), str(PROJECT_ROOT))

    _render_explainability_drawer(outputs, ui_state)

    page = st.session_state.get("page", "Home / Watchlist")
    if page == "Home / Watchlist":
        _render_home(outputs, ui_state)
    elif page == "Track Detail":
        _render_track_detail(outputs, ui_state)
    elif page == "Artist Detail":
        _render_artist_detail(outputs, ui_state)
    elif page == "Tastemakers":
        _render_tastemakers(outputs, ui_state)
    elif page == "Alerts":
        _render_alerts(outputs, ui_state)
    elif page == "Backtest / History":
        _render_backtest(outputs, ui_state)
    elif page == "Settings":
        _render_settings(outputs, ui_state)


if __name__ == "__main__":
    main()
