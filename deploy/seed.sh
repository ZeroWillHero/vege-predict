#!/usr/bin/env bash
# Loads the freshly trained results (results/metrics/*.csv, data/processed/*.csv)
# into Postgres and invalidates the Redis cache, via app/backend/seed.py.
#
# Runs from the backend's own venv (app/backend/.venv), NOT the root research
# venv — seed.py only needs pandas/sqlalchemy/asyncpg, already in
# app/backend/requirements.txt. Assumes the docker-compose stack (postgres,
# redis) is already up and its ports are reachable at localhost, since
# config.py's defaults point there.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ ! -d app/backend/.venv ]; then
  echo "ERROR: app/backend/.venv not found. Run the one-time EC2 setup first" >&2
  echo "       (see deploy/setup-ec2.md)." >&2
  exit 1
fi

echo "==> Seeding Postgres from results/metrics/*.csv and invalidating Redis cache"
app/backend/.venv/bin/python3 app/backend/seed.py
