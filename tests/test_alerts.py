from __future__ import annotations

import unittest

from talent_scouting_intel.pipeline.alerts import build_alerts
from talent_scouting_intel.utils.io import load_config


class AlertTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config("configs/default.yaml")

    def test_new_inflection_and_stage_upgrade_emit_alerts(self) -> None:
        prev_state = {
            "trk_1": {
                "score": 0.44,
                "stage": "early",
                "inflection_detected": False,
                "risky": False,
            }
        }
        scored = [
            {
                "track_id": "trk_1",
                "track_name": "Signal Bloom",
                "artist_name": "Nova Lane",
                "genre": "indie pop",
                "final_score": 0.56,
                "stage": "emerging",
                "inflection_detected": True,
                "suspicious": False,
                "spike_only": False,
                "playlist_dependent": False,
                "explanation": "acceleration + depth",
            }
        ]

        alerts, _ = build_alerts(scored, prev_state, self.config)
        types = {item["type"] for item in alerts}
        self.assertIn("inflection", types)
        self.assertIn("stage_upgrade", types)
        self.assertIn("momentum_surge", types)

    def test_risk_flag_when_signature_turns_on(self) -> None:
        prev_state = {
            "trk_2": {
                "score": 0.5,
                "stage": "emerging",
                "inflection_detected": False,
                "risky": False,
            }
        }
        scored = [
            {
                "track_id": "trk_2",
                "track_name": "Promo Burst",
                "artist_name": "Lumen",
                "genre": "pop",
                "final_score": 0.52,
                "stage": "emerging",
                "inflection_detected": False,
                "suspicious": True,
                "spike_only": False,
                "playlist_dependent": False,
                "explanation": "risk",
            }
        ]
        alerts, _ = build_alerts(scored, prev_state, self.config)
        types = {item["type"] for item in alerts}
        self.assertIn("risk_flag", types)


if __name__ == "__main__":
    unittest.main()
