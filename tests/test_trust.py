from __future__ import annotations

import unittest

from talent_scouting_intel.pipeline.scoring import build_scores
from talent_scouting_intel.utils.io import load_config


class TrustScoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config("configs/default.yaml")
        # Isolate trust behavior from priors in this unit test.
        self.config["priors"]["enabled"] = False

    def test_trust_score_drops_with_insufficient_history(self) -> None:
        features = [
            {
                "track_id": "trk_low",
                "track_name": "Low History",
                "artist_id": "art_low",
                "artist_name": "Test Artist",
                "genre_hint": "pop",
                "metadata_text": "pop",
                "latest_date": "2026-02-22",
                "momentum_score": 0.35,
                "acceleration_score": 0.0,
                "size_norm_accel": 0.0,
                "consecutive_positive_accel": 0,
                "depth_score": 0.06,
                "cross_platform_score": 0.13,
                "network_score": 0.15,
                "consistency_score": 0.0,
                "geo_score": 0.0,
                "shortform_proxy_score": 0.0,
                "knowledge_graph_score": 0.1,
                "comment_specificity": 0.0,
                "echo_score": 0.12,
                "candidate_priority": 0.2,
                "tastemaker_score": 0.15,
                "anomaly_score": 0.2,
                "manual_seeded": False,
                "spike_only": False,
                "suspicious": False,
                "playlist_dependent": False,
                "artist_followers": 1500,
                "spotify_points": 1,
                "history_days": 1,
                "views_recent": 300,
                "comments_recent": 2,
                "likes_recent": 15,
                "follower_conversion": 0.1,
                "save_proxy": 0.002,
                "shares_per_view": 0.001,
                "comments_per_view": 0.001,
                "engagement_rate": 0.01,
            }
        ]
        scored, _ = build_scores(features, self.config)
        row = scored[0]
        self.assertEqual("baseline", row["stage"])
        self.assertLess(float(row["trust_score"]), 0.4)
        self.assertTrue(bool(row["low_trust"]))

    def test_trust_score_rises_with_history_and_signal_strength(self) -> None:
        features = [
            {
                "track_id": "trk_high",
                "track_name": "High Trust",
                "artist_id": "art_high",
                "artist_name": "Indie Breakout",
                "genre_hint": "indie pop",
                "metadata_text": "indie pop chorus story",
                "latest_date": "2026-02-22",
                "momentum_score": 0.74,
                "acceleration_score": 0.78,
                "size_norm_accel": 0.11,
                "consecutive_positive_accel": 4,
                "depth_score": 0.72,
                "cross_platform_score": 0.7,
                "network_score": 0.56,
                "consistency_score": 0.69,
                "geo_score": 0.41,
                "shortform_proxy_score": 0.63,
                "knowledge_graph_score": 0.52,
                "comment_specificity": 0.61,
                "echo_score": 0.67,
                "candidate_priority": 0.78,
                "tastemaker_score": 0.69,
                "anomaly_score": 0.71,
                "manual_seeded": False,
                "spike_only": False,
                "suspicious": False,
                "playlist_dependent": False,
                "artist_followers": 85000,
                "spotify_points": 24,
                "history_days": 24,
                "views_recent": 85000,
                "comments_recent": 560,
                "likes_recent": 6700,
                "follower_conversion": 1.2,
                "save_proxy": 0.05,
                "shares_per_view": 0.018,
                "comments_per_view": 0.007,
                "engagement_rate": 0.12,
            }
        ]
        scored, _ = build_scores(features, self.config)
        row = scored[0]
        self.assertNotEqual("baseline", row["stage"])
        self.assertGreater(float(row["trust_score"]), 0.7)
        self.assertEqual("high", str(row["trust_tier"]))
        self.assertFalse(bool(row["low_trust"]))


if __name__ == "__main__":
    unittest.main()

