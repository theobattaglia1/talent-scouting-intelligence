from __future__ import annotations

import csv
import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path

from talent_scouting_intel.pipeline.backtest import run_backtest
from talent_scouting_intel.utils.io import load_config


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


class BacktestCalibrationTests(unittest.TestCase):
    def test_backtest_writes_calibration_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = load_config("configs/default.yaml")
            config["priors"]["enabled"] = False
            config["ingest"]["append_history"] = False
            config["backtest"]["top_k"] = 2
            config["calibration"]["threshold_grid"]["final_score"] = {"min": 0.2, "max": 0.8, "step": 0.3}
            config["calibration"]["threshold_grid"]["trust_score"] = {"min": 0.2, "max": 0.8, "step": 0.3}
            config["backtest"]["min_history_days"] = 28
            config["backtest"]["prediction_horizon_days"] = 42
            config["backtest"]["replay_step_days"] = 14
            config["backtest"]["min_windows_for_calibration"] = 1

            snapshots_path = root / "outputs" / "snapshots.jsonl"
            config["paths"]["snapshots"] = str(snapshots_path.relative_to(root))
            config["paths"]["backtest_json"] = "outputs/backtest.json"
            config["paths"]["backtest_md"] = "outputs/backtest.md"
            config["paths"]["calibration_json"] = "outputs/calibration.json"
            config["paths"]["calibration_md"] = "outputs/calibration.md"
            config["paths"]["calibration_backtest_md"] = "outputs/backtest_calibrated.md"
            config["paths"]["breakout_templates"] = "data/user/historical_breakouts_v2.csv"

            breakout_rows = [
                {
                    "artist_name": "Rising Artist",
                    "track_name": "Rising Song",
                    "approx_window": "2025-03",
                    "iso_start": "",
                    "iso_end": "",
                    "primary_platform": "spotify",
                    "region": "US",
                    "lane": "indie pop",
                    "notes": "template breakout",
                }
            ]
            _write_csv(root / "data" / "user" / "historical_breakouts_v2.csv", breakout_rows)

            start = dt.date(2025, 1, 1)
            rows: list[dict[str, object]] = []
            for day_offset in range(120):
                day = start + dt.timedelta(days=day_offset)
                # Track A: compounding breakout profile.
                if day_offset < 60:
                    streams_a = int(3000 + (day_offset * 45))
                else:
                    streams_a = int(6000 * (1.08 ** (day_offset - 60)))
                rows.append(
                    {
                        "date": day.isoformat(),
                        "platform": "spotify",
                        "track_id": "sp_rising_song",
                        "track_name": "Rising Song",
                        "artist_id": "sp_rising_artist",
                        "artist_name": "Rising Artist",
                        "artist_followers": 22000,
                        "release_date": start.isoformat(),
                        "genre_hint": "indie pop",
                        "views": int(streams_a * 1.05),
                        "likes": int(streams_a * 0.08),
                        "comments": int(streams_a * 0.008),
                        "shares": int(streams_a * 0.012),
                        "streams": streams_a,
                        "listeners": int(streams_a * 0.62),
                        "playlist_adds": int(streams_a * 0.03),
                        "creator_reuse": 0,
                        "region_metrics": {"US": int(streams_a * 0.6), "UK": int(streams_a * 0.2)},
                        "tastemaker_id": "tm_rising",
                        "tastemaker_name": "Rising TM",
                        "event_type": "playlist_add",
                        "source": "test",
                        "comments_text": ["on repeat", "lyrics hit hard"],
                        "collaborators": ["Producer A"],
                        "manual_seeded": False,
                        "metadata_text": "indie pop rising song",
                    }
                )

                # Track B: flat/low-signal control.
                streams_b = int(11000 + (day_offset * 8))
                rows.append(
                    {
                        "date": day.isoformat(),
                        "platform": "spotify",
                        "track_id": "sp_flat_song",
                        "track_name": "Flat Song",
                        "artist_id": "sp_flat_artist",
                        "artist_name": "Flat Artist",
                        "artist_followers": 90000,
                        "release_date": start.isoformat(),
                        "genre_hint": "pop",
                        "views": int(streams_b * 1.02),
                        "likes": int(streams_b * 0.02),
                        "comments": int(streams_b * 0.0015),
                        "shares": int(streams_b * 0.002),
                        "streams": streams_b,
                        "listeners": int(streams_b * 0.5),
                        "playlist_adds": int(streams_b * 0.005),
                        "creator_reuse": 0,
                        "region_metrics": {"US": int(streams_b * 0.7)},
                        "tastemaker_id": "tm_flat",
                        "tastemaker_name": "Flat TM",
                        "event_type": "playlist_add",
                        "source": "test",
                        "comments_text": ["cool"],
                        "collaborators": [],
                        "manual_seeded": False,
                        "metadata_text": "pop flat song",
                    }
                )

            _write_jsonl(snapshots_path, rows)

            cfg_path = root / "config.json"
            cfg_path.write_text(json.dumps(config), encoding="utf-8")

            payload = run_backtest(str(cfg_path), project_root=root)
            self.assertIn("summary", payload)
            self.assertIn("calibration", payload)

            calibration_json = root / "outputs" / "calibration.json"
            self.assertTrue(calibration_json.exists())
            calibration = json.loads(calibration_json.read_text(encoding="utf-8"))
            recommended = calibration.get("recommended_thresholds", {})
            self.assertIn("final_score", recommended)
            self.assertIn("trust_score", recommended)
            self.assertGreaterEqual(float(recommended.get("final_score", 0.0)), 0.0)
            self.assertGreaterEqual(float(recommended.get("trust_score", 0.0)), 0.0)

            self.assertTrue((root / "outputs" / "backtest.md").exists())
            self.assertTrue((root / "outputs" / "calibration.md").exists())
            self.assertTrue((root / "outputs" / "backtest_calibrated.md").exists())


if __name__ == "__main__":
    unittest.main()
