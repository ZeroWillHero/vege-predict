#!/usr/bin/env bash
# Runs the full research pipeline: trains all 9 model families x 6 vegetables,
# then generates genuine future forecasts. Writes into trained_models/ and
# results/ at the repo root (both gitignored) — the exact locations
# app/backend/seed.py and the API expect.
#
# Ensures uv + the root .venv exist and are up to date (installing uv itself
# if needed) before training — data/raw/ still has to be uploaded manually
# once (see deploy/setup-ec2.md), since there's no source to fetch it from.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source deploy/ensure_venv.sh
ensure_venv .venv requirements.txt

if [ ! -d data/raw/vegetable_prices ] && [ ! -d data/raw/weather ]; then
  echo "ERROR: data/raw/ looks empty. Upload the raw source data before training" >&2
  echo "       (see deploy/setup-ec2.md)." >&2
  exit 1
fi

echo "==> Training all model families (scripts/train_all.py)"
.venv/bin/python3 scripts/train_all.py

echo "==> Generating genuine future forecasts (scripts/predict_future.py)"
.venv/bin/python3 scripts/predict_future.py

echo "==> Training complete. trained_models/ and results/metrics/ are up to date."
