# Talent Scouting Intelligence - Technical Design v2

## Scope
This design operationalizes:
- dataset ingestion from `data/user/` for cold-start priors and calibration
- entity resolution across platform aliases
- daily + weekly feature computation per `signals_feature_schema.csv`
- interpretable scoring with fixed creative model weights
- threshold calibration and replay backtesting

## Pipeline Modules
1. `pipeline/ingest.py`
- boots source registry from `starter_sources_additions.csv`
- pulls autonomous snapshots from adapters
- optionally backfills baselines from `bootstrap_datasets.csv`
- appends/dedupes history
- writes canonicalized snapshots via entity resolution

2. `pipeline/entity_resolution.py`
- derives stable `canonical_artist_id` / `canonical_track_id`
- handles alias collapse (`sp_`, `mb_`, `yt_`, `lfm_`, fallback normalized names)
- writes alias registry (`outputs/entity_resolution.csv`, `outputs/state/entity_resolution.json`)

3. `pipeline/candidates.py` (Stage A)
- computes low-base acceleration, anomaly, tastemaker, echo, and geo scores
- applies established-artist leakage controls:
  - follower threshold
  - artist track-footprint threshold
  - strict strong-signal requirement for mature catalogs

4. `pipeline/features.py` + `pipeline/signal_features.py`
- computes feature buckets for Stage B
- computes signal tables exactly from `signals_feature_schema.csv`
- emits:
  - `outputs/features.csv`
  - `outputs/features_daily_track.csv`
  - `outputs/features_weekly_track.csv`
  - `outputs/features_daily_artist.csv`
  - `outputs/features_weekly_artist.csv`

5. `pipeline/priors.py`
- ingests `affinity_artists.csv` and `historical_breakouts_v2.csv`
- resolves IDs where possible (pipeline IDs + optional Spotify/MusicBrainz lookups)
- computes:
  - `affinity_score` (taste-fit prior)
  - `path_similarity_score` (trajectory prior)

6. `pipeline/scoring.py` (Stage B)
- base momentum model (genre-aware bucket weights)
- fixed creative model:
  - sonic 40%
  - persona/brand 30%
  - writing 15%
  - market position 15%
- blended base score + anti-gaming penalties
- gated prior boosts with hard caps (prevents prior-only lift)
- trust score and stage assignment
- calibrated popping decision + confidence

7. `pipeline/backtest.py`
- historical replay windows
- labels from:
  - future `broke_proxy` behavior
  - historical breakout windows from `historical_breakouts_v2.csv`
- grid-search calibration of score/trust thresholds
- writes:
  - `outputs/backtest.json`
  - `outputs/backtest.md`
  - `outputs/calibration.json`
  - `outputs/calibration.md`
  - `outputs/backtest_calibrated.md`

8. `pipeline/report.py`
- markdown/html top-candidate report
- includes explainability text + flags

## Storage Schema
Primary tables/files:
- `snapshots.jsonl`: atomic per-platform daily observations
- `candidates.jsonl`: Stage A candidate rows
- `features.csv`: Stage B aggregated candidate features
- `scored.csv`: final scored candidates with priors/trust/popping fields
- `inflections.jsonl`: inflection events
- `tastemakers.csv`: tastemaker quant scores
- calibration/backtest artifacts listed above

Key IDs:
- raw IDs: `track_id`, `artist_id`
- canonical IDs: `canonical_track_id`, `canonical_artist_id`
- alias keys: `track_alias_key`, `artist_alias_key`

## Jobs and Schedules
Recommended local cadence:
1. Daily ingest + scoring:
- `tsi autopilot --mode auto`
2. Weekly calibration refresh:
- `tsi backtest`
3. Weekly report export:
- `tsi report`

Job ordering:
1. ingest
2. candidates
3. features
4. score
5. tastemakers
6. alerts
7. report
8. backtest (weekly or after data/model changes)

## Rate Limits and API Hygiene
General:
- bounded per-run caps (`max_*` config keys)
- short request timeouts
- adapter-level graceful skip on missing keys/errors

Platform constraints:
- YouTube Data API: quota-sensitive; bounded by per-run channel/video/comment caps
- Spotify Web API: client-credentials flow with playlist/track cap limits
- Last.fm API: bounded `limit_per_tag`
- Wikipedia/MusicBrainz: public API best-effort, bounded artist counts
- Reddit/RSS: bounded post/feed entry pulls

## Failure Modes and Mitigations
1. Established artists falsely flagged as early
- mitigation: established leakage gating (followers + catalog footprint + strong-signal requirement)

2. Prior-only uplift on weak evidence
- mitigation: gated prior boost + low-signal multipliers + hard prior boost cap

3. Sparse history overconfidence
- mitigation: insufficient-history penalties and trust-tier demotion

4. Single-platform paid push spikes
- mitigation: spike/suspicious/playlist-dependent penalties and prior gate suppression

5. Alias fragmentation across sources
- mitigation: canonical ID layer + alias output registry

6. Missing API keys or outages
- mitigation: adapter skip/fallback behavior, pipeline stays runnable in mock mode

## Configuration Surface
Key tuning locations:
- candidate controls: `candidate_generation.*`
- signal thresholds: `thresholds.*`
- score weights: `weights.*`
- fixed creative blend: `model_blend.*`
- priors/gating: `priors.*`
- calibration grid/objective: `calibration.*`

## Explainability Contract
Each scored row exposes:
- bucket scores + weights
- penalties + trust components
- prior contributions (affinity/path)
- stage + inflection + popping decision/confidence
- flags and natural-language reason string
