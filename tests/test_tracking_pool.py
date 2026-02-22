from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from talent_scouting_intel.pipeline.tracking_pool import build_tracking_targets, refresh_tracking_pool
from talent_scouting_intel.utils.io import ensure_parent, load_config, write_csv, write_jsonl


class TrackingPoolTests(unittest.TestCase):
    def _config_for_tmp(self) -> dict:
        config = load_config("configs/default.yaml")
        config["paths"]["scored"] = "outputs/scored.csv"
        config["paths"]["candidates"] = "outputs/candidates.jsonl"
        config["paths"]["tracking_pool_state"] = "outputs/state/tracking_pool.json"
        return config

    def test_refresh_adds_targets_and_respects_ignored(self) -> None:
        config = self._config_for_tmp()

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_csv(
                root / "outputs" / "scored.csv",
                [
                    {
                        "track_id": "sp_score1",
                        "artist_id": "sp_artist1",
                        "track_name": "Score Song",
                        "artist_name": "Score Artist",
                        "stage": "baseline",
                        "final_score": 0.42,
                        "prior_gate": 0.31,
                        "trust_score": 0.37,
                        "candidate_priority": 0.55,
                        "spike_only": False,
                        "suspicious": False,
                        "inflection_detected": False,
                    },
                    {
                        "track_id": "sp_flagged1",
                        "artist_id": "sp_artist2",
                        "track_name": "Flagged Song",
                        "artist_name": "Flagged Artist",
                        "stage": "early",
                        "final_score": 0.75,
                        "prior_gate": 0.6,
                        "trust_score": 0.65,
                        "candidate_priority": 0.7,
                        "spike_only": False,
                        "suspicious": True,
                        "inflection_detected": False,
                    },
                ],
            )
            write_jsonl(
                root / "outputs" / "candidates.jsonl",
                [
                    {
                        "track_id": "sp_cand1",
                        "artist_id": "sp_cand_artist1",
                        "track_name": "Candidate Song",
                        "artist_name": "Candidate Artist",
                        "candidate_priority": 0.81,
                        "established_artist": False,
                        "manual_seeded": False,
                    },
                    {
                        "track_id": "sp_established1",
                        "artist_id": "sp_big_artist1",
                        "track_name": "Established Song",
                        "artist_name": "Established Artist",
                        "candidate_priority": 0.91,
                        "established_artist": True,
                        "manual_seeded": False,
                    },
                ],
            )

            ui_state_path = root / "outputs" / "state" / "ui_state.json"
            ensure_parent(ui_state_path)
            ui_state_path.write_text(
                json.dumps(
                    {
                        "tracked_track_ids": ["sp_ui1", "yt_vid1", "sp_score1"],
                        "ignored_track_ids": ["sp_score1"],
                    }
                ),
                encoding="utf-8",
            )

            stats = refresh_tracking_pool(config, root)
            self.assertTrue(stats["enabled"])
            self.assertGreaterEqual(int(stats["items_active"]), 1)
            self.assertGreaterEqual(int(stats["removed_ignored"]), 0)

            targets, target_stats = build_tracking_targets(config, root)
            self.assertTrue(target_stats["enabled"])
            self.assertIn("ui1", targets["spotify_track_ids"])
            self.assertIn("cand1", targets["spotify_track_ids"])
            self.assertIn("vid1", targets["youtube_video_ids"])
            self.assertNotIn("score1", targets["spotify_track_ids"])
            self.assertNotIn("flagged1", targets["spotify_track_ids"])

    def test_build_targets_prunes_expired_items(self) -> None:
        config = self._config_for_tmp()

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pool_path = root / "outputs" / "state" / "tracking_pool.json"
            ensure_parent(pool_path)
            pool_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "updated_at": "2026-02-22",
                        "items": [
                            {
                                "track_id": "sp_expired1",
                                "artist_id": "sp_artist_expired1",
                                "spotify_track_id": "expired1",
                                "priority": 0.9,
                                "sources": ["scored_auto"],
                                "first_added": "2026-01-01",
                                "last_seen": "2026-01-02",
                                "expires_on": "2026-01-10",
                            },
                            {
                                "track_id": "sp_active1",
                                "artist_id": "sp_artist_active1",
                                "spotify_track_id": "active1",
                                "priority": 0.7,
                                "sources": ["ui_tracked"],
                                "first_added": "2026-02-20",
                                "last_seen": "2026-02-22",
                                "expires_on": "2999-01-01",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            targets, stats = build_tracking_targets(config, root)
            self.assertEqual(1, int(stats["expired_removed"]))
            self.assertEqual(["active1"], targets["spotify_track_ids"])


if __name__ == "__main__":
    unittest.main()
