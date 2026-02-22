# Talent Scouting Intelligence - Runbook

## 1) First Run
1. Create env and install:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```
2. Optional API keys:
```bash
export SPOTIFY_CLIENT_ID=...
export SPOTIFY_CLIENT_SECRET=...
export YOUTUBE_API_KEY=...
export LASTFM_API_KEY=...
```
3. Run full local pipeline:
```bash
tsi run-all --config configs/default.yaml --mode auto
```

## 2) Dataset Ingestion (User Priors + Calibration)
Expected files:
- `data/user/affinity_artists.csv`
- `data/user/historical_breakouts_v2.csv`
- `data/user/signals_feature_schema.csv`
- `data/user/bootstrap_datasets.csv`
- `data/user/starter_sources_additions.csv`

How they are used:
- affinity artists -> `affinity_score`
- breakout templates -> `path_similarity_score` + calibration labels
- signal schema -> required signal table outputs
- bootstrap datasets -> historical baseline backfill adapters
- starter additions -> source registry initialization

## 3) Add New Sources
### Tastemaker starter pack
1. Append rows to `data/user/starter_sources_additions.csv`.
2. Run:
```bash
tsi ingest --config configs/default.yaml --mode auto
```
3. Confirm merge stats in ingest output (`source_registry_bootstrap`).

### New adapter lane
1. Add adapter module under `src/talent_scouting_intel/adapters/`.
2. Register it in `adapters/autonomous_collector.py`.
3. Add config block under `ingest.auto.<adapter_name>`.
4. Add tests with deterministic mock payload.

## 4) Add/Adjust Genre Lanes
1. Edit genre list and prototypes in `configs/default.yaml`:
- `genres.priority`
- `genres.prototypes`
2. Add lane-specific weight adjustments in:
- `weights.genre_adjustments`
3. Re-run backtest to verify precision/lead-time tradeoff.

## 5) Add Regions
1. Add region codes in source registry and relevant adapters.
2. Ensure snapshots populate `region_metrics`.
3. Validate geo signal behavior in:
- `features.csv` (`geo_score`)
- `features_daily_track.csv` / `features_weekly_track.csv`

## 6) Validate New Signals
Checklist:
1. Signal appears in `data/user/signals_feature_schema.csv`.
2. Feature engine computes field in:
- `signal_features.py` (daily + weekly)
3. Signal value lands in outputs:
- `features_daily_track.csv`
- `features_weekly_track.csv`
4. If used in ranking, wire into `features.py` and `scoring.py`.
5. Add/adjust tests in `tests/`.

## 7) Calibration and Backtesting
Run:
```bash
tsi backtest --config configs/default.yaml
```
Artifacts:
- `outputs/backtest.json`
- `outputs/backtest.md`
- `outputs/calibration.json`
- `outputs/calibration.md`
- `outputs/backtest_calibrated.md`

How to interpret:
- `recommended_thresholds` in `outputs/calibration.json` feed popping thresholds.
- review top failure cases before accepting threshold changes.

## 8) Trust/Quality Triage
If results look wrong:
1. Check candidate leakage:
- `established_artist`, `artist_track_footprint`, `established_penalty`
2. Check prior overreach:
- `prior_gate`, `prior_boost`, `prior_boost_capped`
3. Check low evidence:
- `acceleration_score`, `sustained_accel_windows`, `trust_score`
4. Check risk flags:
- `spike_only`, `suspicious`, `playlist_dependent`

## 9) Operational Commands
Daily run:
```bash
tsi autopilot --config configs/default.yaml --mode auto
```
Step-by-step:
```bash
tsi ingest --config configs/default.yaml --mode auto
tsi candidates --config configs/default.yaml
tsi features --config configs/default.yaml
tsi score --config configs/default.yaml
tsi tastemakers --config configs/default.yaml
tsi alerts --config configs/default.yaml
tsi report --config configs/default.yaml
```

## 10) UI Validation
Run:
```bash
make ui
```
Verify:
1. Home has non-empty watchlist and explainability.
2. Track detail shows priors + trust + flags.
3. Backtest page reads updated summary metrics.
4. Filters for genre/platform/region/stage operate on new fields.
