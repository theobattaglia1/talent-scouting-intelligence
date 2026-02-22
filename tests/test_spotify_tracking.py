from __future__ import annotations

import datetime as dt
import os
import unittest
from unittest.mock import patch

from talent_scouting_intel.adapters.spotify_adapter import collect_spotify_snapshots
from talent_scouting_intel.utils.io import load_config


class SpotifyTrackingTests(unittest.TestCase):
    def test_collect_spotify_includes_tracking_follow_rows(self) -> None:
        config = load_config("configs/default.yaml")
        registry = {
            "spotify_playlists": [
                {
                    "playlist_id": "pl1",
                    "name": "Starter Playlist",
                    "genre_tags": ["indie pop"],
                    "region": "US",
                }
            ]
        }

        def fake_get(path: str, token: str, params: dict | None = None) -> dict:
            if path == "playlists/pl1/tracks":
                return {
                    "items": [
                        {
                            "track": {
                                "id": "pltrack1",
                                "name": "Playlist Track",
                                "popularity": 50,
                                "artists": [{"id": "artist_pl", "name": "Playlist Artist"}],
                                "album": {"release_date": "2026-01-15", "name": "Album A"},
                            }
                        }
                    ]
                }
            if path == "artists":
                ids = (params or {}).get("ids", "")
                artist_rows = []
                for artist_id in str(ids).split(","):
                    if not artist_id:
                        continue
                    artist_rows.append(
                        {
                            "id": artist_id,
                            "followers": {"total": 12345},
                            "genres": ["indie pop"],
                        }
                    )
                return {"artists": artist_rows}
            if path == "tracks":
                ids = str((params or {}).get("ids", ""))
                tracks = []
                for track_id in ids.split(","):
                    if not track_id:
                        continue
                    tracks.append(
                        {
                            "id": track_id,
                            "name": f"Tracked {track_id}",
                            "popularity": 61,
                            "artists": [{"id": "artist_tracked", "name": "Tracked Artist"}],
                            "album": {"release_date": "2026-02-01", "name": "Album B"},
                        }
                    )
                return {"tracks": tracks}
            return {}

        with patch.dict(
            os.environ,
            {
                "SPOTIFY_CLIENT_ID": "id",
                "SPOTIFY_CLIENT_SECRET": "secret",
            },
            clear=False,
        ):
            with patch("talent_scouting_intel.adapters.spotify_adapter._spotify_token", return_value="token"):
                with patch("talent_scouting_intel.adapters.spotify_adapter._spotify_get", side_effect=fake_get):
                    rows, meta = collect_spotify_snapshots(
                        config,
                        registry,
                        today=dt.date(2026, 2, 22),
                        tracking_targets={"spotify_track_ids": ["tracked1"]},
                    )

        sources = {str(row.get("source", "")) for row in rows}
        self.assertIn("spotify_api", sources)
        self.assertIn("spotify_track_follow", sources)
        self.assertEqual(2, len(rows))
        self.assertEqual(1, int((meta.get("tracking_follow") or {}).get("rows", 0)))


if __name__ == "__main__":
    unittest.main()
