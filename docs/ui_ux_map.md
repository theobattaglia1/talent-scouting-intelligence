# UI/UX Blueprint: Talent Scouting Intelligence

Treat this like a consumer-grade product (Notion-level polish), not an internal dashboard.

## 1. UX Map + Onboarding Flow (Exact Screen Copy)

### First-run Wizard (3 steps max)

#### Step 1: Genre Focus
- Header: `Step 1 of 3: Choose genre focus`
- Body: `Pick the lanes you care about. We preloaded your core scouting genres.`
- Preselected genres:
  - pop
  - indie pop
  - singer-songwriter
  - country-pop
  - indie folk
  - alt rock
- Primary CTA: `Continue`
- Secondary CTA: `Skip and use defaults`

#### Step 2: Discovery Sources
- Header: `Step 2 of 3: Pick discovery sources`
- Body: `Choose how to source candidates. Starter Pack is curated and works instantly.`
- Options:
  - `Starter Pack (recommended)`
  - `Custom`
- Primary CTA: `Continue`
- Secondary CTA: `Skip and use Starter Pack`
- Tertiary CTA: `Back`

#### Step 3: First Scan
- Header: `Step 3 of 3: Run first scan`
- Body: `We’re about to scan, score, and explain your first scout queue.`
- Scan mode control: `auto | mock | hybrid`
- Progress copy during run:
  - `Collecting fresh source snapshots and daily deltas.`
  - `Generating low-obviousness candidates with tastemaker + anomaly logic.`
  - `Computing acceleration, depth, resonance, network, and anti-gaming features.`
  - `Applying interpretable weighted scoring and inflection detection.`
  - `Re-estimating tastemaker quality with Bayesian hit-rate and lead-time.`
  - `Creating scout inbox alerts for what changed now.`
  - `Updating markdown/html scouting report artifacts.`
- Primary CTA: `Run First Scan`
- Secondary CTA: `Skip and open app`
- Tertiary CTA: `Back`

### If User Skips Everything
- System behavior:
  - Sets defaults (genre list + starter pack + auto mode).
  - Opens app directly.
  - Uses existing outputs/demo artifacts if present.
  - User can trigger scan from Home with one click.

## 2. Information Architecture

### Core Navigation
- Home / Watchlist
- Track Detail
- Artist Detail
- Tastemakers
- Alerts
- Backtest / History
- Settings

### Page Purpose
- Home / Watchlist:
  - Primary action: `Run New Scan`
  - Filter and action hub: track / ignore / export / explain
- Track Detail:
  - Primary action: `Track This`
  - Full explainability and trajectory
- Artist Detail:
  - Primary action: drill to highest-scoring track decision
- Tastemakers:
  - Primary action: `Add Tastemaker`
  - Quantified tastemaker quality table + source management
- Settings:
  - Primary action: `Save settings`
  - Advanced mode/source/genre/filter defaults
- Alerts (optional but implemented):
  - Primary action: `Generate alerts now` when empty
- Backtest / History (optional but implemented):
  - Primary action: `Run backtest` when empty

## 3. Visual Design System

### Typography
- Display: Fraunces (headings)
- UI/body: Manrope
- Scale:
  - H1: 2.0rem
  - H2: 1.5rem
  - H3: 1.2rem
  - Body: 0.95rem
  - Meta/help: 0.82-0.9rem

### Spacing Tokens
- `--space-1`: 0.25rem
- `--space-2`: 0.5rem
- `--space-3`: 0.75rem
- `--space-4`: 1rem
- `--space-5`: 1.5rem
- `--space-6`: 2rem

### Color Tokens (Light + Dark)
- Light:
  - `--bg: #f6f4ef`
  - `--surface: #ffffff`
  - `--text: #151a17`
  - `--muted: #5a615d`
  - `--accent: #145a42`
  - `--danger: #b3463e`
- Dark (prefers-color-scheme):
  - `--bg: #0f1413`
  - `--surface: #161d1b`
  - `--text: #edf3ef`
  - `--muted: #b5c3bd`
  - `--accent: #5ad0a0`
  - `--danger: #ef8d86`

### Components
- Metric cards
- Score/stage chips
- Editable watchlist table
- Sparklines (unicode micro-trends)
- Explainability drawer
- Empty states with 1-click CTAs
- Tooltips/help copy and flag definitions

### UI States
- Loading: progress bar + stage-by-stage logs
- Error: inline error block with failed stage
- Empty: educational empty-state card + one-click action
- Success: stateful confirmation toast/messages

## 4. Interaction Design

### Required Behaviors Implemented
- First scan guided tour (dismissible)
- Smart filter defaults (genre focus + risk flag defaults)
- Single search bar for artist/track/tastemaker
- Bulk watchlist actions:
  - Track selected
  - Ignore selected
  - Export selected CSV
- Explainability drawer includes:
  - Score breakdown (score x weight x contribution)
  - Top features
  - Inflection date
  - Cross-platform echo summary
  - Flags and definitions

### Tight Feedback Loop
- Per-candidate controls:
  - Thumbs Up / Thumbs Down
  - Track This
  - Ignore
- Feedback persisted in local UI state and reflected in watchlist state.

## 5. Implementation Plan and Stack Choice

### Selected Stack: Streamlit
- Why:
  - Fastest local shipping with full Python pipeline reuse.
  - Supports zero-config first launch and progressive disclosure.
  - Allows polished presentation with custom CSS + guided onboarding.
  - Does not block required UX for MVP scope.

### Command to Run
- `make ui`
- or `python -m app.ui`

## 6. Code Deliverable + Screenshot Instructions

### Added UI Code
- `src/talent_scouting_intel/ui/streamlit_app.py`
- `src/talent_scouting_intel/ui/data.py`
- `src/talent_scouting_intel/ui/state.py`
- `src/talent_scouting_intel/ui/styles.py`
- `app/ui.py`
- `Makefile`

### Screenshot Steps
1. Start app: `python -m app.ui`
2. On wizard Step 1, capture genre preselection state.
3. Step 2, capture Starter Pack vs Custom selector.
4. Step 3, click `Run First Scan` and capture progress log.
5. Home page:
- capture filters + scout queue table + bulk actions.
6. Click `Open Explainability` for any candidate:
- capture right-side Explainability Drawer.
7. Open `Track Detail` and capture momentum timeline + feature table.
8. Open `Tastemakers` and capture quant table + add source form.
9. Open `Settings` and capture advanced controls.

