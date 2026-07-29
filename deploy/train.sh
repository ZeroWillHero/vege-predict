#!/usr/bin/env bash
# Runs the full research pipeline: trains all 9 model families x 6 vegetables,
# then generates genuine future forecasts. Writes into trained_models/ and
# results/ at the repo root (both gitignored) — the exact locations
# app/backend/seed.py and the API expect.
#
# Assumes data/raw/ and the root .venv already exist on this machine (see
# deploy/setup-ec2.md for one-time setup) — this script does not fetch
# or rebuild either.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ ! -d .venv ]; then
  echo "ERROR: .venv not found at repo root. Run the one-time EC2 setup first" >&2
  echo "       (see deploy/setup-ec2.md)." >&2
  exit 1
fi

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
