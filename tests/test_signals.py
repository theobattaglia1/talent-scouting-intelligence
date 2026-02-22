from __future__ import annotations

import unittest

from talent_scouting_intel.pipeline.features import detect_spike_only
from talent_scouting_intel.pipeline.scoring import detect_inflection
from talent_scouting_intel.utils.io import load_config


class SignalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config("configs/default.yaml")

    def test_inflection_detected_when_accel_persists_and_buckets_corroborate(self) -> None:
        feature = {
            "size_norm_accel": 0.09,
            "consecutive_positive_accel": 3,
            "acceleration_score": 0.73,
            "depth_score": 0.64,
            "cross_platform_score": 0.61,
            "network_score": 0.56,
            "geo_score": 0.42,
            "consistency_score": 0.62,
        }
        detected, corroborated = detect_inflection(feature, self.config)
        self.assertTrue(detected)
        self.assertGreaterEqual(len(corroborated), 2)

    def test_inflection_not_detected_for_single_bucket(self) -> None:
        feature = {
            "size_norm_accel": 0.08,
            "consecutive_positive_accel": 2,
            "acceleration_score": 0.72,
            "depth_score": 0.4,
            "cross_platform_score": 0.35,
            "network_score": 0.3,
            "geo_score": 0.2,
            "consistency_score": 0.3,
        }
        detected, _ = detect_inflection(feature, self.config)
        self.assertFalse(detected)

    def test_spike_only_flag(self) -> None:
        streams = [200, 210, 215, 220, 950, 160, 150, 148]
        anti_cfg = self.config["features"]["anti_gaming"]
        self.assertTrue(detect_spike_only(streams, depth_score=0.2, echo_score=0.22, anti_cfg=anti_cfg))

    def test_spike_not_flagged_if_depth_and_echo_confirmed(self) -> None:
        streams = [200, 210, 215, 220, 950, 600, 700, 820]
        anti_cfg = self.config["features"]["anti_gaming"]
        self.assertFalse(detect_spike_only(streams, depth_score=0.7, echo_score=0.72, anti_cfg=anti_cfg))


if __name__ == "__main__":
    unittest.main()
