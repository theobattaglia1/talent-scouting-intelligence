# Talent Scouting Intelligence (TSI) - System Design

## 1) Objective
Detect early, compounding momentum for artists/tracks before mainstream awareness, prioritizing:
- pop
- indie pop
- singer-songwriter
- country-pop
- indie folk
- alt rock

Core design goals:
1. Early inflection lead time.
2. Low false positives.
3. Interpretable recommendations.
4. Practical workflow (watchlist, filters, alerts/reporting).

## 2) Data & Compliance
No paid data dependencies are required.

Allowed inputs:
- Official free APIs (optional adapters with keys).
- Public pages where Terms and robots permit.
- Open metadata datasets (MusicBrainz/Wikidata adapters can be added).
- YouTube RSS + optional YouTube Data API stats enrichment.
- Spotify Web API (client credentials) playlist monitoring.
- Reddit public JSON endpoints for subreddit community signals.
- Last.fm public API tag charts.
- Tastemaker/news RSS feeds.
- Wikimedia pageviews API for off-platform attention.
- MusicBrainz public API for collaborator/relationship graph enrichment.
- TikTok workaround via proxy synthesis from cross-platform short-form mentions (`tiktok_proxy`), without direct TikTok scraping.
- Human seed URLs (artist/track/video) for forced tracking.

ToS-safe adapter strategy:
- Each platform adapter is optional and toggleable.
- Default autonomous path pulls from registry-driven sources and free APIs.
- Manual snapshots remain optional fallback, not primary workflow.
- Pipeline behavior is platform-agnostic after ingestion.

## 3) Two-Stage Architecture

### Stage A - Candidate Generation (Before-Everyone Engine)
A1) Tastemaker monitoring:
- Build daily snapshots of tastemaker adds/posts.
- Tastemaker score uses Bayesian success with recency weighting.
- Blend online Bayesian score with prior-day quantified tastemaker quality.
- Graph:
  - Nodes: tastemakers + tracks.
  - Edge: tastemaker featured track.
  - Track network signal: weighted tastemaker edges.

A1b) Parallel tastemaker discovery (quantifiable):
- Compute tastemaker `quant_score` from:
  - Bayesian precision on successful picks.
  - Lead-time advantage before break/inflection.
  - Early-capture rate (`lead_days >= min_lead_days`).
  - Novelty ratio (low-follower picks).
  - Genre alignment to target genres.
  - Reliability (inverse risky-pick rate).
  - Track-graph centrality proxy.
- Promote tastemakers by status:
  - `qualified` (enough trials),
  - `emergent` (strong score with low sample),
  - `incubating` (monitor only).

A2) New release + low-follower anomalies:
- Focus on low-obviousness tracks with high size-normalized acceleration.
- Detect engagement/conversion anomalies vs cohort baseline.

A3) Cross-platform triangulation:
- Compute lag-aware correlation (TikTok/YouTube/Instagram leading Spotify).
- Echo score boosts tracks where multiple weak signals align.
- Include `tiktok_proxy -> spotify` lag correlation as a legal workaround signal.

A4) Micro-geo breakout:
- Detect local concentration first, then geographic diffusion.

A5) Human-in-the-loop seeding:
- Scout-pasted links force tracking and receive candidate uplift.

Candidate output:
- `candidate_priority` + decomposed components (`tastemaker_score`, `anomaly_score`, `echo_score`, `geo_score`, `low_base_accel`).

### Stage B - Ranking + Explanation (A&R Layer)
- Compute transparent feature buckets.
- Score with genre-aware weight profiles.
- Apply anti-gaming penalties.
- Trigger inflection events.
- Emit explanation text for each recommendation.
- Generate proactive alert inbox:
  - new inflections
  - stage upgrades
  - momentum surges
  - risk flags

## 4) Signal Definitions (Exact Formulas)
Given daily metric series `y_t` (typically Spotify streams for core acceleration):

### 4.1 Acceleration
- Growth (log):
  - `g_t = log(y_t + 1) - log(y_(t-1) + 1)`
- Acceleration (2nd derivative style):
  - `a_t = g_t - g_(t-1)`
- Size-normalized acceleration:
  - `SNA = 100 * mean(a_t over last 7) / max(1, log(1 + mean(y_t over last 7)))`
  - Factor `100` is a readability scale so thresholding is human-legible.
- Positive acceleration rate:
  - `%a+ = count(a_t > 0) / count(a_t)`
- Tail sustain:
  - `CPA = consecutive positive a_t at tail`

Default threshold contributions:
- `size_norm_accel_min = 0.065`
- `min_consecutive_windows = 2`

### 4.2 Sustained Compounding
- Positive growth rate:
  - `%g+ = count(g_t > 0) / count(g_t)`
- Sign stability:
  - `stability = 1 - sign_changes(g_t) / (len(g_t) - 1)`
- Consistency score:
  - `consistency = 0.6 * %g+ + 0.4 * stability`

### 4.3 Depth / Conversion
From recent window (last 56 platform snapshots):
- `engagement_rate = (likes + comments + shares) / (views + 1)`
- `comments_per_view = comments / (views + 1)`
- `shares_per_view = shares / (views + 1)`
- `save_proxy = playlist_adds / (listeners + 1)`
- `follower_conversion = listeners / (artist_followers + 1)`
- `comment_specificity = specific_comments / total_comments`
  - specific comment if text contains one of: `lyrics, chorus, bridge, verse, on repeat, ...`

### 4.4 Cross-Platform Resonance
For source platform `p` and destination Spotify:
- Lag-aware max correlation:
  - `corr*(p->spotify) = max_{lag in [Lmin,Lmax]} corr(growth_p(t), growth_spotify(t+lag))`
- Echo score:
  - `echo = 0.55 * accel_strength + 0.45 * mean(corr*)`
- Final cross-platform score blends direct echo with short-form proxy:
  - `cross_platform = 0.82 * echo + 0.18 * shortform_proxy_score`

Default lag windows:
- TikTok -> Spotify: 3-10 days
- YouTube -> Spotify: 2-9 days
- Instagram -> Spotify: 1-7 days

### 4.5 Network Effects
- Tastemaker Bayesian score:
  - `score_tm = (alpha + weighted_successes_tm) / (beta + weighted_trials_tm)`
  - `weighted_*` uses recency half-life weighting.
- Track network score:
  - mean tastemaker scores of track edges + collaborator count normalization.
- Knowledge graph boost:
  - derive from Wikipedia growth and MusicBrainz relation/tag density.
- Tastemaker quant score:
  - `quant = w1*bayes_precision + w2*lead_score + w3*early_capture + w4*novelty + w5*genre_alignment + w6*reliability + w7*centrality`
  - `lead_score = minmax(avg_lead_days, 0, max_lead_score_days)`

### 4.6 Geo
- Concentration (`HHI`) from first-week regional shares.
- Diffusion gain:
  - `regions_last_week_above_5pct - regions_first_week_above_5pct`
- Geo score:
  - blend of `(1 - HHI)` and diffusion gain.

### 4.7 Content Velocity & Creator Reuse
- Content velocity proxy:
  - unique `(date, platform)` observations over last 28 days / 28.
- Creator reuse growth:
  - mean recent growth of TikTok creator reuse count.
- If direct TikTok is unavailable, use `tiktok_proxy` creator reuse estimate from source mentions.

### 4.8 Anti-Gaming / Fraud
Spike-only flag:
- Let daily pct growth be `r_t = (y_t - y_(t-1)) / (y_(t-1)+1)`.
- Flag if:
  - `max(r_t) >= 2.5`
  - next-day reversion `<= -0.55`
  - depth score `< 0.45`
  - echo score `< 0.5`

Suspicious flag:
- `views/comments >= 1200`
- `likes/comments >= 80`
- `echo <= 0.35`

Playlist-dependent flag:
- Spotify growth share `>= 0.8`
- playlist_add ratio `>= 0.008`
- echo `< 0.5`

## 5) Operational Inflection Trigger
Inflection event fires when all conditions pass:
1. `size_norm_accel >= 0.065`
2. `consecutive_positive_accel >= 2`
3. At least 2 corroborating buckets above thresholds:
   - acceleration >= 0.6
   - depth >= 0.55
   - cross_platform >= 0.55
   - network >= 0.5
   - geo >= 0.45
   - consistency >= 0.55

## 6) Genre Tagging & Filtering
MVP implemented:
- Inputs: `genre_hint + metadata_text + title`.
- Method: tokenized text -> cosine similarity vs genre prototype keywords.
- Output: `genre`, `genre_confidence`.

V2 extension path:
- Add legally accessible audio embeddings and nearest-neighbor clustering for “sounds like”.

## 7) Interpretable Scoring Model
Base sub-scores:
- momentum
- acceleration
- depth
- cross_platform
- network
- consistency
- geo

Base weights:
- momentum: 0.18
- acceleration: 0.22
- depth: 0.17
- cross_platform: 0.16
- network: 0.13
- consistency: 0.09
- geo: 0.05

Genre adjustments (examples):
- pop: +cross_platform, -depth
- singer-songwriter: +depth, +consistency, -cross_platform
- country-pop: +geo, +consistency
- indie folk: +depth, +network, -cross_platform

Penalties:
- spike_only: -0.20
- suspicious: -0.25
- playlist_dependent: -0.15

Final:
- `final_score = clamp01(weighted_sum - penalties)`

Stage labels:
- `breaking`: inflection true and score >= 0.72
- `emerging`: score >= 0.52
- `early`: else

## 8) Backtesting Protocol
Proxy label for "broke" (future horizon default 42 days):
- max Spotify streams >= 50k
- weekly growth >= 0.18
- sustained for >= 3 weeks

Replay:
- Rolling weekly cutoffs.
- At each cutoff, score using only past data.
- Evaluate top-k against future broke labels.

Metrics:
- precision@k
- recall@k
- lead time (days)
- false positive rate
- week-to-week stability (Jaccard overlap of top-k)

## 9) How Scouts Do This Without Paying
Practical low-cost workflow:
1. Curate 20-60 high-signal tastemakers across your target genres.
2. Snapshot daily new adds/uploads.
3. Compute deltas and rank low-base, high-acceleration adds.
4. Normalize by artist baseline size (followers / prior stream level).
5. Triangulate weak cross-platform echoes with lag.
6. Track geo micro-communities and whether breakout diffuses.
7. Update curator hit rates as outcomes arrive.

Equivalent in this repo:
- `ingest`: autonomous source collection from YouTube, Spotify, Reddit, Last.fm, and RSS (all optional/toggleable) with history appending.
- `candidates`: tastemaker + anomaly + echo + geo + seed uplift.
- `features`: interpretable bucket math + anti-gaming flags.
- `score`: weighted genre-tuned ranking + inflection detection.
- `tastemakers`: quantified tastemaker ranking and status.
- `alerts`: proactive scout inbox generation.
- `report`: watchlist-style output with “why now”.
- `backtest`: historical replay and lead-time metrics.

## 10) Extensibility
- Add real adapters as optional plugins (Spotify, YouTube, TikTok where legal).
- Keep a canonical snapshot schema.
- Preserve mock mode for reproducible offline testing.
- Add scheduler/daemon around `autopilot` to push daily inbox alerts automatically.
