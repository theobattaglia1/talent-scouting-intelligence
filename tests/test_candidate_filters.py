from __future__ import annotations

import unittest

from talent_scouting_intel.pipeline.candidates import build_candidates
from talent_scouting_intel.utils.io import load_config


class CandidateFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config("configs/default.yaml")
        self.config["candidate_generation"]["min_spotify_points"] = 3
        self.config["candidate_generation"]["established_artist_track_footprint_min"] = 4
        self.config["candidate_generation"]["established_artist_require_strong_signals"] = True
        self.config["candidate_generation"]["established_artist_min_low_base_accel"] = 0.68
        self.config["candidate_generation"]["established_artist_min_echo"] = 0.62
        self.config["candidate_generation"]["established_artist_min_anomaly"] = 0.58

    def test_established_catalog_is_filtered_without_strong_signals(self) -> None:
        rows: list[dict[str, object]] = []
        # Build a mature catalog footprint for one artist.
        for track_idx in range(1, 7):
            for day_idx in range(1, 6):
                rows.append(
                    {
                        "date": f"2026-01-0{day_idx}",
                        "platform": "spotify",
                        "track_id": f"sp_mega_{track_idx}",
                        "track_name": f"Mega Track {track_idx}",
                        "artist_id": "sp_mega_artist",
                        "artist_name": "Mega Artist",
                        "artist_followers": 60000,
                        "streams": 150000 + (day_idx * 250),
                        "views": 150000 + (day_idx * 300),
                        "likes": 4500,
                        "comments": 40,
                        "shares": 90,
                        "listeners": 50000,
                        "tastemaker_id": "tm_catalog",
                        "region_metrics": {"US": 80000, "UK": 20000},
                    }
                )

        # Add a genuinely early candidate with stronger acceleration profile.
        for day_idx, streams in enumerate([800, 980, 1300, 2050, 3300], start=1):
            rows.append(
                {
                    "date": f"2026-01-0{day_idx}",
                    "platform": "spotify",
                    "track_id": "sp_indie_breakout",
                    "track_name": "Indie Breakout",
                    "artist_id": "sp_indie_artist",
                    "artist_name": "Indie Artist",
                    "artist_followers": 8000,
                    "streams": streams,
                    "views": int(streams * 1.05),
                    "likes": int(streams * 0.12),
                    "comments": int(streams * 0.012),
                    "shares": int(streams * 0.02),
                    "listeners": int(streams * 0.65),
                    "tastemaker_id": "tm_indie",
                    "region_metrics": {"US": int(streams * 0.6), "AU": int(streams * 0.2)},
                }
            )

        candidates = build_candidates(rows, self.config)
        track_ids = {row["track_id"] for row in candidates}

        self.assertIn("sp_indie_breakout", track_ids)
        self.assertNotIn("sp_mega_1", track_ids)
        self.assertNotIn("sp_mega_2", track_ids)


if __name__ == "__main__":
    unittest.main()
