#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f ".env.local" ]]; then
  echo "Missing .env.local in $ROOT_DIR"
  echo "Create .env.local with SPOTIFY_CLIENT_ID/SPOTIFY_CLIENT_SECRET/YOUTUBE_API_KEY/LASTFM_API_KEY."
  exit 1
fi

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Missing virtualenv python at .venv/bin/python"
  echo "Run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && pip install -e ."
  exit 1
fi

set -a
source ".env.local"
set +a

# Start from clean outputs to avoid mixed mock/real rows.
rm -f outputs/snapshots.jsonl outputs/candidates.jsonl outputs/features.csv outputs/scored.csv outputs/inflections.jsonl

PYTHONPATH=src ./.venv/bin/python -m talent_scouting_intel run-all --config configs/default.yaml --mode auto --project-root .
