from __future__ import annotations

import unittest

from talent_scouting_intel.pipeline.tastemakers import build_tastemaker_profiles
from talent_scouting_intel.utils.io import load_config


class TastemakerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config("configs/default.yaml")
        self.config["tastemaker_discovery"]["min_trials"] = 1

    def test_successful_lead_tastemaker_scores_higher(self) -> None:
        snapshots = [
            {
                "date": "2026-01-01",
                "platform": "spotify",
                "track_id": "trk_a",
                "track_name": "Track A",
                "artist_id": "art_1",
                "artist_name": "Artist 1",
                "artist_followers": 20000,
                "tastemaker_id": "tm_good",
                "tastemaker_name": "Good Curator",
            },
            {
                "date": "2026-01-03",
                "platform": "spotify",
                "track_id": "trk_b",
                "track_name": "Track B",
                "artist_id": "art_2",
                "artist_name": "Artist 2",
                "artist_followers": 30000,
                "tastemaker_id": "tm_bad",
                "tastemaker_name": "Bad Curator",
            },
        ]
        scored = [
            {
                "track_id": "trk_a",
                "latest_date": "2026-01-15",
                "final_score": 0.71,
                "inflection_detected": True,
                "genre": "indie pop",
                "spike_only": False,
                "suspicious": False,
                "playlist_dependent": False,
            },
            {
                "track_id": "trk_b",
                "latest_date": "2026-01-15",
                "final_score": 0.31,
                "inflection_detected": False,
                "genre": "pop",
                "spike_only": True,
                "suspicious": False,
                "playlist_dependent": False,
            },
        ]

        profiles = build_tastemaker_profiles(snapshots, scored, self.config)
        by_id = {row["tastemaker_id"]: row for row in profiles}

        self.assertGreater(by_id["tm_good"]["quant_score"], by_id["tm_bad"]["quant_score"])
        self.assertEqual(by_id["tm_good"]["status"], "qualified")


if __name__ == "__main__":
    unittest.main()
