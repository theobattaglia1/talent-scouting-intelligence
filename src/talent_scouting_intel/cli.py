from __future__ import annotations

import argparse
import json
from pathlib import Path

from talent_scouting_intel.pipeline.alerts import run_alerts
from talent_scouting_intel.pipeline.autopilot import run_autopilot
from talent_scouting_intel.pipeline.backtest import run_backtest
from talent_scouting_intel.pipeline.candidates import run_candidates
from talent_scouting_intel.pipeline.features import run_features
from talent_scouting_intel.pipeline.ingest import run_ingest
from talent_scouting_intel.pipeline.report import run_report
from talent_scouting_intel.pipeline.scoring import run_score
from talent_scouting_intel.pipeline.tastemakers import run_tastemakers


def _print(payload: dict) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=True))


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML/JSON config file",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root for relative output paths",
    )


def _add_mode_arg(parser: argparse.ArgumentParser, default: str = "auto") -> None:
    parser.add_argument("--mode", choices=["auto", "mock", "manual", "hybrid"], default=default)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tsi",
        description="Talent Scouting Intelligence CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest snapshots via adapters")
    _add_common_args(ingest)
    _add_mode_arg(ingest, default="auto")
    ingest.add_argument("--manual-import-path", default=None)
    ingest.add_argument("--seed-url", action="append", default=[])

    candidates = subparsers.add_parser("candidates", help="Generate early candidates")
    _add_common_args(candidates)
    candidates.add_argument("--as-of", default=None, help="ISO date cutoff")

    features = subparsers.add_parser("features", help="Compute feature buckets")
    _add_common_args(features)
    features.add_argument("--as-of", default=None, help="ISO date cutoff")

    score = subparsers.add_parser("score", help="Score and detect inflections")
    _add_common_args(score)

    report = subparsers.add_parser("report", help="Generate markdown/html report")
    _add_common_args(report)

    tastemakers = subparsers.add_parser("tastemakers", help="Quantify and rank tastemakers")
    _add_common_args(tastemakers)

    alerts = subparsers.add_parser("alerts", help="Generate proactive scout alerts")
    _add_common_args(alerts)

    backtest = subparsers.add_parser("backtest", help="Historical replay and metrics")
    _add_common_args(backtest)

    run_all = subparsers.add_parser("run-all", help="Run full pipeline")
    _add_common_args(run_all)
    _add_mode_arg(run_all, default="auto")
    run_all.add_argument("--manual-import-path", default=None)
    run_all.add_argument("--seed-url", action="append", default=[])
    run_all.add_argument("--as-of", default=None, help="ISO date cutoff for candidates/features")

    autopilot = subparsers.add_parser("autopilot", help="Run autonomous scouting loop with alerts")
    _add_common_args(autopilot)
    _add_mode_arg(autopilot, default="auto")
    autopilot.add_argument("--seed-url", action="append", default=[])
    autopilot.add_argument("--as-of", default=None, help="ISO date cutoff for candidates/features")
    autopilot.add_argument("--with-backtest", action="store_true")

    args = parser.parse_args()
    root = Path(args.project_root).resolve()

    if args.command == "ingest":
        payload = run_ingest(
            args.config,
            mode=args.mode,
            manual_import_path=args.manual_import_path,
            seed_urls=args.seed_url,
            project_root=root,
        )
        _print(payload)
        return

    if args.command == "candidates":
        payload = run_candidates(args.config, project_root=root, as_of=args.as_of)
        _print(payload)
        return

    if args.command == "features":
        payload = run_features(args.config, project_root=root, as_of=args.as_of)
        _print(payload)
        return

    if args.command == "score":
        payload = run_score(args.config, project_root=root)
        _print(payload)
        return

    if args.command == "report":
        payload = run_report(args.config, project_root=root)
        _print(payload)
        return

    if args.command == "tastemakers":
        payload = run_tastemakers(args.config, project_root=root)
        _print(payload)
        return

    if args.command == "alerts":
        payload = run_alerts(args.config, project_root=root)
        _print(payload)
        return

    if args.command == "backtest":
        payload = run_backtest(args.config, project_root=root)
        _print(payload)
        return

    if args.command == "run-all":
        outputs = {
            "ingest": run_ingest(
                args.config,
                mode=args.mode,
                manual_import_path=args.manual_import_path,
                seed_urls=args.seed_url,
                project_root=root,
            ),
            "candidates": run_candidates(args.config, project_root=root, as_of=args.as_of),
            "features": run_features(args.config, project_root=root, as_of=args.as_of),
            "score": run_score(args.config, project_root=root),
            "tastemakers": run_tastemakers(args.config, project_root=root),
            "alerts": run_alerts(args.config, project_root=root),
            "report": run_report(args.config, project_root=root),
            "backtest": run_backtest(args.config, project_root=root),
        }
        _print(outputs)
        return

    if args.command == "autopilot":
        outputs = run_autopilot(
            args.config,
            mode=args.mode,
            with_backtest=args.with_backtest,
            seed_urls=args.seed_url,
            as_of=args.as_of,
            project_root=root,
        )
        _print(outputs)
        return

    parser.error("Unsupported command")


if __name__ == "__main__":
    main()
