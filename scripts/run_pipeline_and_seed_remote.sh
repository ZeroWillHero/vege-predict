#!/usr/bin/env bash
# End-to-end pipeline: rebuild datasets, train every model family, forecast the next
# 2 weeks, then migrate + seed a REMOTE Postgres/Redis instance (not the local Docker
# containers described in CLAUDE.md).
#
# Usage:
#   ./scripts/run_pipeline_and_seed_remote.sh
#
# Remote connection info is read from REMOTE_DATABASE_URL / REMOTE_REDIS_URL if set,
# otherwise falls back to the defaults below.

set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE_HOST="3.111.37.80"
: "${REMOTE_DATABASE_URL:=postgresql+asyncpg://postgres:pg123@${REMOTE_HOST}:5432/vegepredict}"
: "${REMOTE_REDIS_URL:=redis://${REMOTE_HOST}:6379/0}"
: "${FUTURE_HORIZON_WEEKS:=8}"

export VEGEPREDICT_DATABASE_URL="$REMOTE_DATABASE_URL"
export VEGEPREDICT_REDIS_URL="$REMOTE_REDIS_URL"

echo "=== 1/5: rebuilding weather data ==="
.venv/bin/python3 scripts/fetch_weather_openmeteo.py

echo "=== 2/5: rebuilding per-vegetable processed datasets ==="
for veg in carrot brinjal pumpkin cabbage snake_gourd leeks; do
    .venv/bin/python3 src/data_processing/build_dataset.py --vegetable "$veg"
done

echo "=== 3/5: training all model families for all vegetables ==="
.venv/bin/python3 scripts/train_all.py

echo "=== 4/5: forecasting the next ${FUTURE_HORIZON_WEEKS} weeks ==="
.venv/bin/python3 scripts/predict_future.py --weeks "$FUTURE_HORIZON_WEEKS"

echo "=== 5/5: migrating + seeding remote database (${REMOTE_HOST}) ==="
(cd app/backend && .venv/bin/python3 -m alembic upgrade head)
app/backend/.venv/bin/python3 app/backend/seed.py

echo "Done. Remote Postgres (${REMOTE_HOST}) and Redis seeded/invalidated."

