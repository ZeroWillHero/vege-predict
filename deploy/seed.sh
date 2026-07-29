#!/usr/bin/env bash
# Loads the freshly trained results (results/metrics/*.csv, data/processed/*.csv)
# into Postgres and invalidates the Redis cache, via app/backend/seed.py.
#
# Runs from the backend's own venv (app/backend/.venv), NOT the root research
# venv — seed.py only needs pandas/sqlalchemy/asyncpg, already in
# app/backend/requirements.txt. Assumes the docker-compose stack (postgres,
# redis) is already up and its ports are reachable at localhost, since
# config.py's defaults point there.
#
# Ensures uv + this venv exist and are up to date (installing uv itself if
# needed) before seeding.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source deploy/ensure_venv.sh
ensure_venv app/backend/.venv app/backend/requirements.txt

echo "==> Seeding Postgres from results/metrics/*.csv and invalidating Redis cache"
app/backend/.venv/bin/python3 app/backend/seed.py
