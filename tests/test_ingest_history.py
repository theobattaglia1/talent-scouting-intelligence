from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from talent_scouting_intel.pipeline.ingest import run_ingest
from talent_scouting_intel.utils.io import load_config


class IngestHistoryTests(unittest.TestCase):
    def test_mock_reingest_dedupes_when_history_enabled(self) -> None:
        base = load_config("configs/default.yaml")
        base["project"]["mock_days"] = 28
        base.setdefault("ingest", {})["append_history"] = True
        base["ingest"]["history_retention_days"] = 365

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cfg_path = root / "config.json"
            cfg_path.write_text(json.dumps(base), encoding="utf-8")

            first = run_ingest(str(cfg_path), mode="mock", project_root=root)
            second = run_ingest(str(cfg_path), mode="mock", project_root=root)

            self.assertEqual(first["rows"], second["rows"])
            history = second.get("source_stats", {}).get("history", {})
            self.assertTrue(history.get("enabled", False))
            self.assertGreaterEqual(int(history.get("existing_rows", 0)), 1)


if __name__ == "__main__":
    unittest.main()
