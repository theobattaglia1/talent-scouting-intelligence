from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from talent_scouting_intel.ui.state import load_ui_state, merge_source_registry, save_ui_state


class UIStateTests(unittest.TestCase):
    def test_merge_source_registry_dedupes(self) -> None:
        starter = {
            "youtube_channels": [{"channel_id": "abc", "name": "Starter", "genre_tags": ["pop"], "estimated_followers": 0}],
            "spotify_playlists": [],
            "reddit_subreddits": [],
            "lastfm_tags": [],
            "rss_feeds": [],
        }
        custom = {
            "youtube_channels": [
                {"channel_id": "abc", "name": "Starter Copy", "genre_tags": ["pop"], "estimated_followers": 12},
                {"channel_id": "xyz", "name": "Custom", "genre_tags": ["indie pop"], "estimated_followers": 5},
            ]
        }

        merged = merge_source_registry(starter, custom)
        self.assertEqual(2, len(merged["youtube_channels"]))
        ids = {row["channel_id"] for row in merged["youtube_channels"]}
        self.assertEqual({"abc", "xyz"}, ids)

    def test_load_save_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            state = load_ui_state(root)
            self.assertFalse(state["onboarding_complete"])

            state["onboarding_complete"] = True
            state["run_mode"] = "hybrid"
            state["custom_sources"]["reddit_subreddits"].append({"name": "indieheads", "genre_tags": ["indie pop"]})
            save_ui_state(root, state)

            loaded = load_ui_state(root)
            self.assertTrue(loaded["onboarding_complete"])
            self.assertEqual("hybrid", loaded["run_mode"])
            self.assertEqual("indieheads", loaded["custom_sources"]["reddit_subreddits"][0]["name"])


if __name__ == "__main__":
    unittest.main()
