#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
STAMP=$(date +%Y%m%dT%H%M%S)
mkdir -p logs work
LOG="logs/${STAMP}.log"

exec > >(tee -a "$LOG") 2>&1

# Install uv if it isn't already on PATH.
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found — installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create the virtual environment if it doesn't exist yet.
if [ ! -d ".venv" ]; then
    uv venv .venv --python 3.13
fi

PYTHON=".venv/bin/python3"

# Install/sync dependencies if requirements.txt changed since last install.
STAMP_FILE=".venv/.requirements.stamp"
if [ ! -f "$STAMP_FILE" ] || [ requirements.txt -nt "$STAMP_FILE" ]; then
    uv pip install -r requirements.txt --python "$PYTHON"
    touch "$STAMP_FILE"
fi

echo "=== [$STAMP] Data collecting ==="
# HARTI weekly bulletin: vegetable wholesale/retail prices (all 6 vegetables).
# "update" is the strategic/incremental path — only fetches bulletins not
# already recorded in data/raw/vegetable_prices/.harti_ingestion_state.json.
"$PYTHON" src/pipeline/scrapers/harti_prices.py update

# CEYPETCO diesel price revisions, forward-filled to weekly cadence.
"$PYTHON" src/pipeline/scrapers/cpc_fuel.py update

# Open-Meteo weather, district-wise (2014-present).
"$PYTHON" scripts/fetch_weather_openmeteo.py

echo "=== [$STAMP] Data quality audit ==="
# Read-only report on data/raw/* — surfaces missing weeks / outliers introduced
# by the scrapers above; does not block the pipeline (see
# src/pipeline/quality_checks.py docstring).
"$PYTHON" src/pipeline/quality_checks.py --audit

echo "=== [$STAMP] Build processed datasets ==="
"$PYTHON" src/data_processing/build_dataset.py

echo "=== [$STAMP] Train all model families ==="
"$PYTHON" scripts/train_all.py

echo "=== [$STAMP] Genuine future forecast ==="
"$PYTHON" scripts/predict_future.py

echo "=== [$STAMP] Seed backend (Postgres + Redis) ==="
app/backend/.venv/bin/python3 app/backend/seed.py

echo "=== [$STAMP] Done ==="
