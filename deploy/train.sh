#!/usr/bin/env bash
# Full research pipeline, mirroring run_training_pipeline.sh's data-through-
# forecast steps (everything except the final seed, which deploy/seed.sh does
# separately, after the docker stack is up): collect fresh source data,
# quality-audit it, rebuild the processed datasets, train all 9 model
# families x 6 vegetables, then generate genuine future forecasts. Writes
# into data/raw/, data/processed/, trained_models/, and results/ at the repo
# root (all gitignored) — the exact locations app/backend/seed.py and the API
# expect.
#
# Ensures uv + the root .venv exist and are up to date (installing uv itself
# if needed) before running anything.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source deploy/ensure_venv.sh
ensure_venv .venv requirements.txt

PYTHON=".venv/bin/python3"

if [ ! -d data/raw/vegetable_prices ] && [ ! -d data/raw/weather ]; then
  echo "WARNING: data/raw/ looks empty. The scrapers below only fetch new" >&2
  echo "         bulletins ('update' mode) — they won't backfill full history." >&2
  echo "         Upload the historical data once (see deploy/setup-ec2.md) if" >&2
  echo "         this is a fresh instance." >&2
fi

echo "==> Collecting fresh source data"
# HARTI weekly bulletin: vegetable wholesale/retail prices (all 6 vegetables).
# "update" is the strategic/incremental path — only fetches bulletins not
# already recorded in data/raw/vegetable_prices/.harti_ingestion_state.json.
"$PYTHON" src/pipeline/scrapers/harti_prices.py update

# CEYPETCO diesel price revisions, forward-filled to weekly cadence.
"$PYTHON" src/pipeline/scrapers/cpc_fuel.py update

# Open-Meteo weather, district-wise (2014-present).
"$PYTHON" scripts/fetch_weather_openmeteo.py

echo "==> Data quality audit (read-only, does not block the pipeline)"
"$PYTHON" src/pipeline/quality_checks.py --audit

echo "==> Building processed datasets (src/data_processing/build_dataset.py)"
"$PYTHON" src/data_processing/build_dataset.py

echo "==> Training all model families (scripts/train_all.py)"
"$PYTHON" scripts/train_all.py

echo "==> Generating genuine future forecasts (scripts/predict_future.py)"
"$PYTHON" scripts/predict_future.py

echo "==> Training complete. trained_models/ and results/metrics/ are up to date."
