from __future__ import annotations

from pathlib import Path
from typing import Any

from talent_scouting_intel.pipeline.alerts import run_alerts
from talent_scouting_intel.pipeline.backtest import run_backtest
from talent_scouting_intel.pipeline.candidates import run_candidates
from talent_scouting_intel.pipeline.features import run_features
from talent_scouting_intel.pipeline.ingest import run_ingest
from talent_scouting_intel.pipeline.report import run_report
from talent_scouting_intel.pipeline.scoring import run_score
from talent_scouting_intel.pipeline.tastemakers import run_tastemakers
from talent_scouting_intel.utils.io import load_config


def run_autopilot(
    config_path: str,
    *,
    mode: str = "auto",
    with_backtest: bool = False,
    seed_urls: list[str] | None = None,
    as_of: str | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = project_root or Path.cwd()
    config = load_config(config_path)

    if mode == "auto":
        mode = str(config.get("ingest", {}).get("mode_default", "auto"))

    outputs: dict[str, Any] = {
        "ingest": run_ingest(
            config_path,
            mode=mode,
            seed_urls=seed_urls,
            project_root=root,
        ),
        "candidates": run_candidates(config_path, project_root=root, as_of=as_of),
        "features": run_features(config_path, project_root=root, as_of=as_of),
        "score": run_score(config_path, project_root=root),
        "tastemakers": run_tastemakers(config_path, project_root=root),
        "alerts": run_alerts(config_path, project_root=root),
        "report": run_report(config_path, project_root=root),
    }

    if with_backtest:
        outputs["backtest"] = run_backtest(config_path, project_root=root)

    return outputs
