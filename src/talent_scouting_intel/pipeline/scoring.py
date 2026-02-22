from __future__ import annotations

from pathlib import Path
from typing import Any

from talent_scouting_intel.pipeline.priors import build_prior_context, score_priors
from talent_scouting_intel.pipeline.tracking_pool import refresh_tracking_pool
from talent_scouting_intel.utils.genre import classify_genre
from talent_scouting_intel.utils.io import load_config, read_csv, resolve_path, write_csv, write_jsonl
from talent_scouting_intel.utils.math_utils import clamp01


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def _as_int(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, value) for value in weights.values())
    if total <= 0:
        return {key: 1.0 / len(weights) for key in weights}
    return {key: max(0.0, value) / total for key, value in weights.items()}


def genre_weight_profile(genre: str, config: dict[str, Any]) -> dict[str, float]:
    base = {key: float(value) for key, value in config["weights"]["base"].items()}
    adjustments = config["weights"]["genre_adjustments"].get(genre, {})
    for key, value in adjustments.items():
        base[key] = base.get(key, 0.0) + float(value)
    return _normalize_weights(base)


def detect_inflection(feature: dict[str, Any], config: dict[str, Any]) -> tuple[bool, list[str]]:
    cfg = config["thresholds"]["inflection"]
    bucket_cfg = cfg["bucket_thresholds"]

    size_norm_accel = _as_float(feature.get("size_norm_accel"))
    accel_windows = _as_int(feature.get("consecutive_positive_accel"))

    if size_norm_accel < float(cfg["size_norm_accel_min"]):
        return False, []
    if accel_windows < int(cfg["min_consecutive_windows"]):
        return False, []

    buckets = {
        "acceleration": _as_float(feature.get("acceleration_score")),
        "depth": _as_float(feature.get("depth_score")),
        "cross_platform": _as_float(feature.get("cross_platform_score")),
        "network": _as_float(feature.get("network_score")),
        "geo": _as_float(feature.get("geo_score")),
        "consistency": _as_float(feature.get("consistency_score")),
    }

    corroborated = [name for name, score in buckets.items() if score >= float(bucket_cfg[name])]
    if len(corroborated) < int(cfg["min_corrob_buckets"]):
        return False, corroborated
    return True, corroborated


def _trust_score(
    row: dict[str, Any],
    corroborated: list[str],
    config: dict[str, Any],
    *,
    insufficient_history: bool,
    established_artist: bool,
    weak_evidence: bool,
    inflection: bool,
) -> dict[str, Any]:
    cfg = config.get("thresholds", {}).get("trust", {})
    weights = cfg.get("weights", {})
    penalties = cfg.get("penalties", {})

    history_full_points = max(1, int(cfg.get("history_full_points", 21)))
    corrob_full_buckets = max(1, int(cfg.get("corrob_full_buckets", 4)))
    min_views_quality = max(1.0, float(cfg.get("min_views_for_quality", 5000)))
    min_comments_quality = max(1.0, float(cfg.get("min_comments_for_quality", 20)))
    min_history_quality = max(1.0, float(cfg.get("min_history_days_for_quality", 7)))

    history_score = clamp01(_as_float(row.get("spotify_points")) / history_full_points)
    evidence_score = clamp01(
        0.45 * _as_float(row.get("acceleration_score"))
        + 0.3 * _as_float(row.get("depth_score"))
        + 0.2 * _as_float(row.get("cross_platform_score"))
        + 0.05 * _as_float(row.get("consistency_score"))
    )
    corroboration_score = clamp01(len(corroborated) / corrob_full_buckets)
    data_quality_score = clamp01(
        0.45 * min(1.0, _as_float(row.get("views_recent")) / min_views_quality)
        + 0.35 * min(1.0, _as_float(row.get("comments_recent")) / min_comments_quality)
        + 0.2 * min(1.0, _as_float(row.get("history_days")) / min_history_quality)
    )

    trust_base = clamp01(
        _as_float(weights.get("history", 0.35)) * history_score
        + _as_float(weights.get("evidence", 0.35)) * evidence_score
        + _as_float(weights.get("corroboration", 0.2)) * corroboration_score
        + _as_float(weights.get("data_quality", 0.1)) * data_quality_score
    )

    trust_penalty = 0.0
    if _as_bool(row.get("spike_only")):
        trust_penalty += _as_float(penalties.get("spike_only", 0.2))
    if _as_bool(row.get("suspicious")):
        trust_penalty += _as_float(penalties.get("suspicious", 0.25))
    if _as_bool(row.get("playlist_dependent")):
        trust_penalty += _as_float(penalties.get("playlist_dependent", 0.15))
    if insufficient_history:
        trust_penalty += _as_float(penalties.get("insufficient_history", 0.35))
    if weak_evidence:
        trust_penalty += _as_float(penalties.get("weak_evidence", 0.15))
    if established_artist and not inflection and not _as_bool(row.get("manual_seeded")):
        trust_penalty += _as_float(penalties.get("established_artist", 0.15))

    trust_score = clamp01(trust_base - trust_penalty)
    if trust_score >= 0.75:
        trust_tier = "high"
    elif trust_score >= 0.45:
        trust_tier = "medium"
    else:
        trust_tier = "low"

    return {
        "trust_score": round(trust_score, 6),
        "trust_tier": trust_tier,
        "trust_base": round(trust_base, 6),
        "trust_penalty": round(trust_penalty, 6),
        "trust_history": round(history_score, 6),
        "trust_evidence": round(evidence_score, 6),
        "trust_corroboration": round(corroboration_score, 6),
        "trust_data_quality": round(data_quality_score, 6),
        "low_trust": trust_score < _as_float(cfg.get("low_threshold", 0.4)),
    }


def _creative_fit_scores(
    row: dict[str, Any],
    priors: dict[str, Any],
) -> dict[str, float]:
    # Sonic score is a proxy blend until audio embeddings are available per-track.
    sonic = clamp01(
        0.5 * _as_float(priors.get("affinity_direct_score"))
        + 0.3 * _as_float(priors.get("path_similarity_score"))
        + 0.2 * _as_float(row.get("genre_confidence"))
    )
    persona = clamp01(
        0.35 * _as_float(row.get("network_score"))
        + 0.2 * _as_float(row.get("cross_platform_score"))
        + 0.2 * _as_float(row.get("knowledge_graph_score"))
        + 0.15 * minmax(  # follower conversion normalized
            _as_float(row.get("follower_conversion")),
            0.02,
            2.0,
        )
        + 0.1 * minmax(_as_float(row.get("tastemaker_weighted_hits", row.get("tastemaker_score"))), 0.0, 10.0)
    )
    writing = clamp01(
        0.45 * _as_float(row.get("comment_specificity_score", row.get("comment_specificity")))
        + 0.35 * _as_float(row.get("depth_score"))
        + 0.2 * _as_float(priors.get("affinity_text_score"))
    )
    market = clamp01(
        0.35 * _as_float(row.get("momentum_score"))
        + 0.25 * _as_float(row.get("acceleration_score"))
        + 0.2 * _as_float(row.get("cross_platform_score"))
        + 0.1 * minmax(_as_float(row.get("tastemaker_weighted_hits", row.get("tastemaker_score"))), 0.0, 10.0)
        + 0.1 * _as_float(row.get("geo_score"))
    )
    # Fixed weights requested by user:
    # sonic 40%, persona/brand 30%, writing 15%, market 15%.
    taste_fit = clamp01(0.4 * sonic + 0.3 * persona + 0.15 * writing + 0.15 * market)
    return {
        "sonic_palette_score": round(sonic, 6),
        "persona_brand_score": round(persona, 6),
        "writing_score": round(writing, 6),
        "market_position_score": round(market, 6),
        "taste_fit_score": round(taste_fit, 6),
        "creative_model_weight_sonic": 0.4,
        "creative_model_weight_persona": 0.3,
        "creative_model_weight_writing": 0.15,
        "creative_model_weight_market": 0.15,
    }


def minmax(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return clamp01((value - low) / (high - low))


def _load_calibration_thresholds(config: dict[str, Any], root: Path) -> dict[str, float]:
    cfg = config.get("calibration", {})
    default_score = _as_float(cfg.get("default_thresholds", {}).get("final_score", 0.52))
    default_trust = _as_float(cfg.get("default_thresholds", {}).get("trust_score", 0.45))
    out = {"final_score": default_score, "trust_score": default_trust}

    path_value = str(config.get("paths", {}).get("calibration_json", ""))
    if not path_value:
        return out
    path = resolve_path(path_value, root)
    rows = read_csv(path) if path.suffix.lower() == ".csv" else []
    if rows:
        # CSV calibration is not expected here but keep compatibility.
        row = rows[0]
        out["final_score"] = _as_float(row.get("recommended_final_score", default_score))
        out["trust_score"] = _as_float(row.get("recommended_trust_score", default_trust))
        return out
    if not path.exists():
        return out
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return out
    rec = payload.get("recommended_thresholds", {}) if isinstance(payload, dict) else {}
    out["final_score"] = _as_float(rec.get("final_score", out["final_score"]))
    out["trust_score"] = _as_float(rec.get("trust_score", out["trust_score"]))
    return out


def _popping_decision(
    *,
    final_score: float,
    trust_score: float,
    corroborated_count: int,
    stage: str,
    inflection: bool,
    thresholds: dict[str, float],
    config: dict[str, Any],
) -> dict[str, Any]:
    popping_cfg = config.get("popping", {})
    min_buckets = int(popping_cfg.get("min_corroborated_buckets", 2))
    score_w = _as_float(popping_cfg.get("score_weight", 0.45))
    trust_w = _as_float(popping_cfg.get("trust_weight", 0.35))
    corrob_w = _as_float(popping_cfg.get("corroboration_weight", 0.2))

    corrob_norm = clamp01(corroborated_count / max(1, min_buckets + 1))
    confidence = clamp01(score_w * final_score + trust_w * trust_score + corrob_w * corrob_norm)
    score_thr = _as_float(thresholds.get("final_score", 0.52))
    trust_thr = _as_float(thresholds.get("trust_score", 0.45))

    decision = bool(
        final_score >= score_thr
        and trust_score >= trust_thr
        and corroborated_count >= min_buckets
        and (inflection or stage in {"emerging", "breaking"})
    )

    reasons: list[str] = []
    if inflection:
        reasons.append("inflection_detected")
    if stage in {"emerging", "breaking"}:
        reasons.append(f"stage={stage}")
    if final_score >= score_thr:
        reasons.append(f"score>={score_thr:.2f}")
    if trust_score >= trust_thr:
        reasons.append(f"trust>={trust_thr:.2f}")
    if corroborated_count >= min_buckets:
        reasons.append(f"corroborated_buckets={corroborated_count}")
    if not reasons:
        reasons.append("below_popping_threshold")

    return {
        "popping_decision": decision,
        "popping_confidence": round(confidence, 6),
        "popping_reason": "; ".join(reasons),
        "popping_score_threshold": round(score_thr, 6),
        "popping_trust_threshold": round(trust_thr, 6),
    }


def _explanation(
    row: dict[str, Any],
    corroborated: list[str],
    priors: dict[str, Any],
    trust: dict[str, Any],
    creative: dict[str, Any],
    popping: dict[str, Any],
    *,
    inflection: bool,
) -> str:
    parts: list[str] = []
    if _as_bool(row.get("insufficient_history")):
        parts.append("insufficient history depth; collecting baseline before momentum conviction")
    if _as_bool(row.get("prior_boost_capped")):
        parts.append("priors were capped until stronger evidence confirms compounding")
    if _as_bool(row.get("established_artist")) and not inflection:
        parts.append("established artist size deprioritized until fresh inflection appears")
    if _as_float(row.get("acceleration_score")) >= 0.65:
        parts.append("size-normalized acceleration is sustained")
    if _as_float(row.get("depth_score")) >= 0.6:
        parts.append("engagement depth and comment specificity are strong")
    if _as_float(row.get("cross_platform_score")) >= 0.6:
        parts.append("cross-platform echo is aligned with lag structure")
    if _as_float(row.get("shortform_proxy_score")) >= 0.55:
        parts.append("short-form proxy momentum is rising (TikTok-like signal)")
    if _as_float(row.get("network_score")) >= 0.55:
        parts.append("high-quality tastemaker and collaborator network exposure")
    if _as_float(row.get("knowledge_graph_score")) >= 0.5:
        parts.append("knowledge-graph adjacency and off-platform attention are strengthening")
    if _as_float(row.get("geo_score")) >= 0.5:
        parts.append("micro-geo concentration is diffusing")
    if _as_bool(row.get("manual_seeded")):
        parts.append("artist was manually seeded for scout tracking")
    if _as_float(row.get("prior_gate")) >= 0.15 and _as_float(priors.get("affinity_score")) >= 0.58:
        matched = str(priors.get("affinity_match_artist", "")).strip()
        if matched:
            parts.append(f"strong taste-fit affinity with {matched}")
        else:
            parts.append("strong taste-fit affinity prior")
    if _as_float(creative.get("taste_fit_score")) >= 0.55:
        components = {
            "sonic": _as_float(creative.get("sonic_palette_score")),
            "persona": _as_float(creative.get("persona_brand_score")),
            "writing": _as_float(creative.get("writing_score")),
            "market": _as_float(creative.get("market_position_score")),
        }
        top_component = max(components.items(), key=lambda item: item[1])[0]
        parts.append(f"creative prior strong ({top_component}-led taste fit)")
    if _as_float(row.get("prior_gate")) >= 0.15 and _as_float(priors.get("path_similarity_score")) >= 0.58:
        template = str(priors.get("path_template_artist", "")).strip()
        if template:
            parts.append(f"trajectory resembles early breakout path seen in {template}")
        else:
            parts.append("trajectory aligns with historical breakout templates")
    trust_tier = str(trust.get("trust_tier", "low"))
    if trust_tier == "high":
        parts.append("high trust: sufficient history and corroborated signal coverage")
    elif trust_tier == "medium":
        parts.append("medium trust: partial corroboration, monitor next windows")
    else:
        parts.append("low trust: confidence limited until more history and signal confirmation")
    if not parts:
        parts.append("candidate shows emerging but incomplete momentum")
    if corroborated:
        parts.append(f"corroborated buckets: {', '.join(corroborated)}")
    if _as_bool(popping.get("popping_decision")):
        parts.append(f"popping now ({str(popping.get('popping_reason', '')).strip()})")
    return "; ".join(parts)


def build_scores(
    features: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    project_root: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scored: list[dict[str, Any]] = []
    inflections: list[dict[str, Any]] = []

    penalties = config["weights"]["penalties"]
    prototypes = config["genres"]["prototypes"]
    allowed = set(config["genres"]["priority"])
    prior_context = build_prior_context(config, project_root or Path.cwd(), features)
    priors_cfg = config.get("priors", {})
    affinity_weight = _as_float(priors_cfg.get("affinity", {}).get("weight"))
    path_weight = _as_float(priors_cfg.get("path_similarity", {}).get("weight"))
    prior_gating_cfg = priors_cfg.get("gating", {})
    blend_cfg = config.get("model_blend", {})
    engine_weight = max(0.0, _as_float(blend_cfg.get("momentum_engine_weight", 0.7)))
    creative_weight = max(0.0, _as_float(blend_cfg.get("creative_fit_weight", 0.3)))
    blend_total = engine_weight + creative_weight
    if blend_total <= 0:
        engine_weight = 1.0
        creative_weight = 0.0
    else:
        engine_weight /= blend_total
        creative_weight /= blend_total
    calibration_thresholds = _load_calibration_thresholds(config, project_root or Path.cwd())

    for row in features:
        raw_text = " ".join(
            [
                str(row.get("genre_hint", "")),
                str(row.get("metadata_text", "")),
                str(row.get("track_name", "")),
            ]
        )
        genre, confidence = classify_genre(raw_text, prototypes)
        if genre not in allowed:
            # Keep but deprioritize if outside preferred scouting lanes.
            confidence *= 0.7

        weights = genre_weight_profile(genre, config)
        priors = score_priors(row, genre, prior_context, config)
        creative = _creative_fit_scores({"genre_confidence": confidence, **row}, priors)

        momentum = _as_float(row.get("momentum_score"))
        acceleration = _as_float(row.get("acceleration_score"))
        depth = _as_float(row.get("depth_score"))
        cross_platform = _as_float(row.get("cross_platform_score"))
        network = _as_float(row.get("network_score"))
        consistency = _as_float(row.get("consistency_score"))
        geo = _as_float(row.get("geo_score"))
        spotify_points = _as_int(row.get("spotify_points"))
        artist_followers = _as_float(row.get("artist_followers"))

        weighted_score = (
            weights["momentum"] * momentum
            + weights["acceleration"] * acceleration
            + weights["depth"] * depth
            + weights["cross_platform"] * cross_platform
            + weights["network"] * network
            + weights["consistency"] * consistency
            + weights["geo"] * geo
        )
        creative_model_score = _as_float(creative.get("taste_fit_score"))
        blended_model_score = clamp01((engine_weight * weighted_score) + (creative_weight * creative_model_score))

        inflection, corroborated = detect_inflection(row, config)
        insufficient_history = spotify_points < int(prior_gating_cfg.get("min_spotify_points", 7))
        established_artist = (
            _as_bool(row.get("established_artist"))
            or artist_followers > _as_float(prior_gating_cfg.get("established_max_followers", 1_000_000))
        )
        sustained_accel_windows = _as_float(row.get("sustained_accel_windows", 0))
        min_sustained_windows = _as_float(prior_gating_cfg.get("min_sustained_accel_windows", 1))
        weak_evidence = (
            acceleration < _as_float(prior_gating_cfg.get("min_acceleration_score", 0.3))
            and depth < _as_float(prior_gating_cfg.get("min_depth_score", 0.18))
            and cross_platform < _as_float(prior_gating_cfg.get("min_cross_platform_score", 0.2))
            and sustained_accel_windows < min_sustained_windows
        )

        penalty = 0.0
        if _as_bool(row.get("spike_only")):
            penalty += float(penalties["spike_only"])
        if _as_bool(row.get("suspicious")):
            penalty += float(penalties["suspicious"])
        if _as_bool(row.get("playlist_dependent")):
            penalty += float(penalties["playlist_dependent"])
        if insufficient_history:
            penalty += _as_float(penalties.get("insufficient_history", 0.0))
        if established_artist and not inflection:
            penalty += _as_float(penalties.get("established_artist", 0.0))

        base_score = clamp01(blended_model_score - penalty)
        prior_gate = clamp01(
            0.35 * acceleration
            + 0.2 * depth
            + 0.2 * cross_platform
            + 0.15 * consistency
            + 0.1 * minmax(sustained_accel_windows, 0.0, 3.0)
        )
        if insufficient_history:
            prior_gate *= _as_float(prior_gating_cfg.get("insufficient_history_gate_multiplier", 0.05))
        if weak_evidence:
            prior_gate *= _as_float(prior_gating_cfg.get("weak_evidence_gate_multiplier", 0.25))
        if sustained_accel_windows < min_sustained_windows and acceleration < _as_float(
            prior_gating_cfg.get("min_acceleration_score", 0.3)
        ):
            prior_gate *= _as_float(prior_gating_cfg.get("low_signal_prior_gate_multiplier", 0.2))
        if _as_bool(row.get("spike_only")) or _as_bool(row.get("suspicious")) or _as_bool(row.get("playlist_dependent")):
            prior_gate *= _as_float(prior_gating_cfg.get("risk_flag_gate_multiplier", 0.1))
        if established_artist and not inflection:
            prior_gate *= _as_float(prior_gating_cfg.get("established_gate_multiplier", 0.05))
        effective_affinity_score = clamp01(
            0.65 * _as_float(creative.get("taste_fit_score"))
            + 0.35 * _as_float(priors.get("affinity_score"))
        )
        affinity_boost = affinity_weight * effective_affinity_score * prior_gate
        path_boost = path_weight * _as_float(priors.get("path_similarity_score")) * prior_gate
        raw_prior_boost = affinity_boost + path_boost
        max_prior_boost = _as_float(
            prior_gating_cfg.get(
                "max_prior_boost_inflection" if inflection else "max_prior_boost_non_inflection",
                0.18 if inflection else 0.05,
            )
        )
        prior_boost_capped = raw_prior_boost > max_prior_boost and max_prior_boost > 0
        if prior_boost_capped and raw_prior_boost > 0:
            cap_ratio = max_prior_boost / raw_prior_boost
            affinity_boost *= cap_ratio
            path_boost *= cap_ratio
        prior_boost = affinity_boost + path_boost
        score = clamp01(base_score + prior_boost)

        if inflection:
            inflections.append(
                {
                    "track_id": row["track_id"],
                    "track_name": row["track_name"],
                    "artist_name": row["artist_name"],
                    "date": row["latest_date"],
                    "size_norm_accel": _as_float(row.get("size_norm_accel")),
                    "corroborated_buckets": corroborated,
                    "score": round(score, 6),
                }
            )

        trust = _trust_score(
            row,
            corroborated,
            config,
            insufficient_history=insufficient_history,
            established_artist=established_artist,
            weak_evidence=weak_evidence,
            inflection=inflection,
        )

        if insufficient_history or (_as_bool(trust.get("low_trust")) and not inflection):
            stage = "baseline"
        elif inflection and score >= 0.72 and _as_float(trust.get("trust_score")) >= 0.55:
            stage = "breaking"
        elif score >= 0.52 and _as_float(trust.get("trust_score")) >= 0.45:
            stage = "emerging"
        else:
            stage = "early"
        popping = _popping_decision(
            final_score=score,
            trust_score=_as_float(trust.get("trust_score")),
            corroborated_count=len(corroborated),
            stage=stage,
            inflection=inflection,
            thresholds=calibration_thresholds,
            config=config,
        )

        temp_row = dict(row)
        temp_row.update(
            {
                "prior_gate": prior_gate,
                "insufficient_history": insufficient_history,
                "established_artist": established_artist,
                "inflection_detected": inflection,
                "prior_boost_capped": prior_boost_capped,
            }
        )
        explanation = _explanation(temp_row, corroborated, priors, trust, creative, popping, inflection=inflection)

        enriched = dict(row)
        enriched.update(
            {
                "genre": genre,
                "genre_confidence": round(confidence, 6),
                "weight_momentum": round(weights["momentum"], 6),
                "weight_acceleration": round(weights["acceleration"], 6),
                "weight_depth": round(weights["depth"], 6),
                "weight_cross_platform": round(weights["cross_platform"], 6),
                "weight_network": round(weights["network"], 6),
                "weight_consistency": round(weights["consistency"], 6),
                "weight_geo": round(weights["geo"], 6),
                "weight_affinity": round(affinity_weight * prior_gate, 6),
                "weight_path_similarity": round(path_weight * prior_gate, 6),
                "weight_momentum_engine": round(engine_weight, 6),
                "weight_creative_fit": round(creative_weight, 6),
                "weighted_score": round(weighted_score, 6),
                "creative_model_score": round(creative_model_score, 6),
                "blended_model_score": round(blended_model_score, 6),
                "penalty": round(penalty, 6),
                "base_final_score": round(base_score, 6),
                "prior_gate": round(prior_gate, 6),
                "affinity_score_effective": round(effective_affinity_score, 6),
                "affinity_boost": round(affinity_boost, 6),
                "path_similarity_boost": round(path_boost, 6),
                "prior_boost": round(prior_boost, 6),
                "prior_boost_capped": prior_boost_capped,
                "insufficient_history": insufficient_history,
                "established_artist": established_artist,
                "weak_evidence": weak_evidence,
                "final_score": round(score, 6),
                "inflection_detected": inflection,
                "stage": stage,
                "explanation": explanation,
            }
        )
        enriched.update(priors)
        enriched.update(creative)
        enriched.update(trust)
        enriched.update(popping)
        scored.append(enriched)

    scored.sort(key=lambda item: float(item["final_score"]), reverse=True)
    inflections.sort(key=lambda item: float(item["score"]), reverse=True)
    return scored, inflections


def run_score(
    config_path: str,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    root = project_root or Path.cwd()

    features = read_csv(resolve_path(config["paths"]["features"], root))
    scored, inflections = build_scores(features, config, project_root=root)

    scored_path = resolve_path(config["paths"]["scored"], root)
    inflections_path = resolve_path(config["paths"]["inflections"], root)

    write_csv(scored_path, scored)
    write_jsonl(inflections_path, inflections)
    tracking_pool = refresh_tracking_pool(config, root, scored_rows=scored)

    return {
        "scored": len(scored),
        "inflections": len(inflections),
        "scored_path": str(scored_path),
        "inflections_path": str(inflections_path),
        "tracking_pool": tracking_pool,
    }
