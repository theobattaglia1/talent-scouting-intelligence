from __future__ import annotations

import datetime as dt
import unittest

from talent_scouting_intel.adapters.shortform_proxy_adapter import synthesize_shortform_proxy_rows
from talent_scouting_intel.utils.io import load_config


class ShortformProxyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config("configs/default.yaml")

    def test_proxy_rows_emitted_on_tiktok_mentions(self) -> None:
        rows = [
            {
                "date": "2026-02-20",
                "platform": "reddit",
                "track_id": "sp_abc123",
                "track_name": "Signal Track",
                "artist_id": "sp_art_1",
                "artist_name": "Nova",
                "artist_followers": 10000,
                "release_date": "2026-02-10",
                "genre_hint": "indie pop",
                "views": 12000,
                "likes": 900,
                "comments": 110,
                "shares": 180,
                "streams": 2000,
                "listeners": 1200,
                "playlist_adds": 0,
                "creator_reuse": 0,
                "region_metrics": {},
                "metadata_text": "This is blowing up on TikTok and fyp",
                "comments_text": ["viral sound"],
                "collaborators": [],
                "manual_seeded": False,
            }
        ]

        proxy, meta = synthesize_shortform_proxy_rows(rows, self.config, today=dt.date(2026, 2, 21))
        self.assertEqual(meta.get("enabled"), True)
        self.assertEqual(len(proxy), 1)
        self.assertEqual(proxy[0]["platform"], "tiktok_proxy")
        self.assertGreater(int(proxy[0]["creator_reuse"]), 0)

    def test_no_proxy_rows_when_no_keywords(self) -> None:
        rows = [
            {
                "date": "2026-02-20",
                "platform": "reddit",
                "track_id": "sp_abc123",
                "track_name": "Signal Track",
                "artist_id": "sp_art_1",
                "artist_name": "Nova",
                "artist_followers": 10000,
                "release_date": "2026-02-10",
                "genre_hint": "indie pop",
                "views": 12000,
                "likes": 900,
                "comments": 110,
                "shares": 180,
                "streams": 2000,
                "listeners": 1200,
                "playlist_adds": 0,
                "creator_reuse": 0,
                "region_metrics": {},
                "metadata_text": "great track with clean growth",
                "comments_text": ["nice"],
                "collaborators": [],
                "manual_seeded": False,
            }
        ]

        proxy, _ = synthesize_shortform_proxy_rows(rows, self.config, today=dt.date(2026, 2, 21))
        self.assertEqual(len(proxy), 0)


if __name__ == "__main__":
    unittest.main()
