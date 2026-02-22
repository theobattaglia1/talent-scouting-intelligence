# Talent Scouting Intelligence

Python MVP for early music momentum detection with legal low-cost data paths and no paid vendor dependency.

## What It Implements
- Two-stage architecture:
  - Stage A: candidate generation (tastemakers, low-base anomalies, cross-platform echo, geo breakout, seed tracking)
  - Stage B: interpretable scoring, anti-gaming penalties, inflection trigger
- Autonomous scouting loop:
  - `autopilot` mode runs ingestion -> scoring -> tastemaker ranking -> alert inbox
  - no manual snapshot import required in default workflow
- Persistent tracking pool:
  - surfaced candidates are auto-added to a 30-day rolling follow pool
  - ingest re-polls tracked tracks/videos on each scan (when source APIs allow)
  - UI tracked/ignored actions feed this pool automatically
- Quant tastemaker intelligence:
  - Bayesian precision, lead-time advantage, novelty hit rate, reliability, centrality
  - outputs ranked tastemaker table and JSON
- Proactive alerting:
  - new inflections, stage upgrades, momentum surges, risk flags
- Multi-source autonomous adapters (all optional, legal/low-cost):
  - YouTube RSS (+ optional YouTube Data API stats)
  - Spotify Web API playlist pulls
  - Reddit public subreddit ingestion
  - Last.fm tag charts
  - Generic RSS tastemaker/blog feeds
  - TikTok workaround via `tiktok_proxy` synthesis from public short-form mentions across sources
  - Wikipedia pageview enrichment
  - MusicBrainz graph enrichment
- Genre priority and tagging:
  - pop, indie pop, singer-songwriter, country-pop, indie folk, alt rock
- Backtesting with lead-time and false-positive metrics
- CLI + Markdown/HTML report + chart
- Mock mode runs end-to-end with no API keys

## Repo Layout
- `configs/default.yaml` and `configs/default.json`: thresholds, weights, platform toggles, genre filters
- `src/talent_scouting_intel/`: pipeline and CLI
- `docs/system_design.md`: full design + formulas + thresholds
- `docs/technical_design_v2.md`: module/storage/job/rate-limit/failure-mode design
- `docs/runbook.md`: operational playbook for sources/lanes/regions/signal validation
- `tests/`: signal-level tests
- `data/mock/manual_seeds.json`: sample seed links
- `data/sources/source_registry.json`: autonomous source registry (YouTube channels)
- `data/sources/source_registry.json`: registry for YouTube, Spotify playlists, Reddit subs, Last.fm tags, RSS feeds
- `outputs/`: generated artifacts
- `docs/ui_ux_map.md`: UX map, onboarding copy, IA, and UI interaction spec

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run the autonomous loop (recommended):
```bash
tsi autopilot --config configs/default.yaml --mode auto
```

Run step-by-step:
```bash
tsi ingest --mode auto --config configs/default.yaml
tsi candidates --config configs/default.yaml
tsi features --config configs/default.yaml
tsi score --config configs/default.yaml
tsi tastemakers --config configs/default.yaml
tsi alerts --config configs/default.yaml
tsi report --config configs/default.yaml
tsi backtest --config configs/default.yaml
```

## Optional APIs For Richer Autonomous Ingestion
Set `YOUTUBE_API_KEY` to enrich YouTube RSS pulls with view/like/comment stats.
Set `SPOTIFY_CLIENT_ID` + `SPOTIFY_CLIENT_SECRET` to enable Spotify playlist pulls.
Set `LASTFM_API_KEY` to enable Last.fm tag chart pulls.

```bash
export YOUTUBE_API_KEY=your_key_here
export SPOTIFY_CLIENT_ID=your_spotify_client_id
export SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
export LASTFM_API_KEY=your_lastfm_api_key
tsi ingest --mode auto --config configs/default.yaml
```

Without these keys, adapters skip gracefully and the run continues.
Wikipedia/MusicBrainz/Reddit/RSS-based adapters do not require paid services.

## Manual Snapshot Import (Legacy Fallback)
This path remains available but is not required for the default autonomous workflow.
If you still want it:
```bash
tsi ingest --mode manual --manual-import-path /absolute/path/snapshots.csv --config configs/default.yaml
```

Required columns:
- `date`, `platform`, `track_id`, `track_name`, `artist_id`, `artist_name`

Optional columns map to internal schema (`views`, `likes`, `comments`, `shares`, `streams`, `listeners`, `playlist_adds`, `region_metrics`, etc.).
Template file:
- `data/mock/manual_snapshot_template.csv`

## Output Artifacts
- `outputs/snapshots.jsonl`
- `outputs/candidates.jsonl`
- `outputs/features.csv`
- `outputs/scored.csv`
- `outputs/inflections.jsonl`
- `outputs/tastemakers.csv`
- `outputs/tastemakers.json`
- `outputs/alerts.jsonl`
- `outputs/alerts.md`
- `outputs/state/alerts_state.json`
- `outputs/state/tracking_pool.json`
- `outputs/report.md`
- `outputs/report.html`
- `outputs/top_candidates.png`
- `outputs/backtest.json`
- `outputs/backtest.md`
- `outputs/calibration.json`
- `outputs/calibration.md`
- `outputs/backtest_calibrated.md`

## Tests
```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Optional Local Dashboard
```bash
make ui
# or
python -m app.ui
```

### UI Screenshot Walkthrough
1. Launch `python -m app.ui`.
2. Capture onboarding Step 1 (genre focus defaults).
3. Capture onboarding Step 2 (Starter Pack vs Custom).
4. Capture onboarding Step 3 after clicking `Run First Scan` (progress + logs).
5. Capture Home / Watchlist filters + bulk actions.
6. Open Explainability Drawer and capture score breakdown + flags.
7. Capture Track Detail timeline and feature table.
8. Capture Tastemakers page (quant table + add source form).
9. Capture Settings advanced controls.
