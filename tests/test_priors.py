from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from typing import Any

from talent_scouting_intel.pipeline.priors import build_prior_context, score_priors
from talent_scouting_intel.pipeline.scoring import build_scores
from talent_scouting_intel.utils.io import load_config


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class PriorTests(unittest.TestCase):
    def test_affinity_and_path_priors_attach_from_user_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            affinity_csv = root / "data" / "user" / "affinity_artists_updated.csv"
            breakout_csv = root / "data" / "user" / "historical_breakouts_updated.csv"

            _write_csv(
                affinity_csv,
                [
                    {
                        "artist_name": "Adele",
                        "primary_genre": "pop",
                        "region_hint": "US/Global",
                        "priority": 5,
                        "affinity_anchor": "Hello",
                        "reason": "Big hook chorus and repeat-listening emotional pull.",
                        "data_confidence": 0.9,
                        "source_url": "",
                    }
                ],
            )
            _write_csv(
                breakout_csv,
                [
                    {
                        "artist_name": "sombr",
                        "primary_genre": "pop",
                        "breakout_window_estimate": "2025-01",
                        "breakout_track_or_moment": "undressed breakout",
                        "platform_path": "Short-form to streaming acceleration",
                        "regions_momentum": "US/UK",
                        "evidence_url_primary": "",
                        "evidence_url_secondary": "",
                        "notes": "TikTok conversation then Spotify lift",
                    }
                ],
            )

            config = load_config("configs/default.yaml")
            config["paths"]["affinity_artists"] = str(affinity_csv)
            config["paths"]["breakout_templates"] = str(breakout_csv)
            config["paths"]["priors_identity_cache"] = str(root / "outputs" / "state" / "id_cache.json")
            config["paths"]["priors_resolution_csv"] = str(root / "outputs" / "user_priors_resolution.csv")
            config["priors"]["identity_resolution"]["network_lookups"] = False

            feature_rows = [
                {
                    "track_id": "sp_track_1",
                    "track_name": "Hello",
                    "artist_id": "sp_artist_1",
                    "artist_name": "Adele",
                    "metadata_text": "pop chorus emotional lyrics on repeat",
                    "momentum_score": 0.62,
                    "acceleration_score": 0.67,
                    "depth_score": 0.59,
                    "cross_platform_score": 0.61,
                    "network_score": 0.42,
                    "consistency_score": 0.56,
                    "geo_score": 0.32,
                    "echo_score": 0.57,
                    "shortform_proxy_score": 0.63,
                    "comment_specificity": 0.42,
                    "knowledge_graph_score": 0.21,
                }
            ]

            context = build_prior_context(config, root, feature_rows)
            priors = score_priors(feature_rows[0], "pop", context, config)

            self.assertGreater(priors["affinity_score"], 0.55)
            self.assertEqual(priors["affinity_match_artist"], "Adele")
            self.assertGreater(priors["path_similarity_score"], 0.35)
            self.assertEqual(priors["path_template_artist"], "sombr")

    def test_build_scores_applies_prior_boost_without_replacing_base_engine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            affinity_csv = root / "data" / "user" / "affinity_artists_updated.csv"
            breakout_csv = root / "data" / "user" / "historical_breakouts_updated.csv"

            _write_csv(
                affinity_csv,
                [
                    {
                        "artist_name": "Adele",
                        "primary_genre": "pop",
                        "region_hint": "US",
                        "priority": 5,
                        "affinity_anchor": "",
                        "reason": "Hook-focused pop with repeat behavior.",
                        "data_confidence": 0.9,
                        "source_url": "",
                    }
                ],
            )
            _write_csv(
                breakout_csv,
                [
                    {
                        "artist_name": "sombr",
                        "primary_genre": "pop",
                        "breakout_window_estimate": "2025-01",
                        "breakout_track_or_moment": "undressed",
                        "platform_path": "short-form to streaming",
                        "regions_momentum": "US",
                        "evidence_url_primary": "",
                        "evidence_url_secondary": "",
                        "notes": "viral sound then streaming validation",
                    }
                ],
            )

            config = load_config("configs/default.yaml")
            config["paths"]["affinity_artists"] = str(affinity_csv)
            config["paths"]["breakout_templates"] = str(breakout_csv)
            config["paths"]["priors_identity_cache"] = str(root / "outputs" / "state" / "id_cache.json")
            config["paths"]["priors_resolution_csv"] = str(root / "outputs" / "user_priors_resolution.csv")
            config["priors"]["identity_resolution"]["network_lookups"] = False
            config["priors"]["affinity"]["weight"] = 0.12
            config["priors"]["path_similarity"]["weight"] = 0.12

            features = [
                {
                    "track_id": "sp_track_1",
                    "track_name": "Skyline",
                    "artist_id": "sp_artist_1",
                    "artist_name": "Adele",
                    "genre_hint": "pop",
                    "metadata_text": "hook pop emotional chorus",
                    "artist_followers": 120000,
                    "latest_date": "2026-02-20",
                    "momentum_score": 0.54,
                    "acceleration_score": 0.63,
                    "size_norm_accel": 0.09,
                    "positive_accel_rate": 0.62,
                    "consecutive_positive_accel": 3,
                    "avg_growth": 0.11,
                    "depth_score": 0.58,
                    "cross_platform_score": 0.6,
                    "network_score": 0.44,
                    "consistency_score": 0.57,
                    "positive_growth_rate": 0.61,
                    "geo_score": 0.33,
                    "shortform_proxy_score": 0.59,
                    "knowledge_graph_score": 0.28,
                    "wikipedia_growth": 0.02,
                    "content_velocity": 0.72,
                    "creator_reuse_growth": 0.1,
                    "engagement_rate": 0.08,
                    "comments_per_view": 0.006,
                    "shares_per_view": 0.012,
                    "save_proxy": 0.03,
                    "follower_conversion": 0.6,
                    "comment_specificity": 0.45,
                    "candidate_priority": 0.66,
                    "tastemaker_score": 0.52,
                    "low_base_accel": 0.61,
                    "anomaly_score": 0.53,
                    "echo_score": 0.6,
                    "manual_seeded": False,
                    "spike_only": False,
                    "suspicious": False,
                    "playlist_dependent": False,
                    "spotify_growth_share": 0.42,
                    "playlist_ratio": 0.003,
                    "views_recent": 20000,
                    "likes_recent": 1300,
                    "comments_recent": 180,
                    "collaborator_count": 2,
                }
            ]

            scored, _ = build_scores(features, config, project_root=root)
            self.assertEqual(len(scored), 1)
            row = scored[0]

            self.assertGreater(float(row["prior_boost"]), 0.0)
            self.assertGreater(float(row["final_score"]), float(row["base_final_score"]))
            self.assertGreater(float(row["affinity_score"]), 0.0)
            self.assertGreater(float(row["path_similarity_score"]), 0.0)
            self.assertIn("baseline", str(row["explanation"]).lower())


if __name__ == "__main__":
    unittest.main()
