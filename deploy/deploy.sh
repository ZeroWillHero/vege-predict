#!/usr/bin/env bash
# Run this ON the server, from the repo root, to (re)deploy the vegepredict stack:
# nginx + api + postgres + redis, all defined in docker-compose.yml.
#
# Run manually any time; the GitHub Actions workflow (.github/workflows/deploy.yml)
# also calls this after training and building the image, so `git pull` is skipped
# there (the workflow's checkout step already has the latest code) but harmless
# to run again standalone.
#
# Usage: deploy/deploy.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ -z "${SKIP_GIT_PULL:-}" ]; then
  echo "==> Pulling latest code"
  git pull --ff-only
fi

echo "==> Building images"
docker compose build

echo "==> Starting/updating stack"
docker compose up -d

echo "==> Waiting for postgres/redis health checks"
until [ "$(docker inspect -f '{{.State.Health.Status}}' vegepredict-postgres)" = "healthy" ]; do sleep 2; done
until [ "$(docker inspect -f '{{.State.Health.Status}}' vegepredict-redis)" = "healthy" ]; do sleep 2; done

echo "==> Running database migrations"
# -c is required: the container's WORKDIR is /srv but alembic.ini lives at
# /srv/app/backend/alembic.ini, so plain `alembic upgrade head` looks for
# ./alembic.ini in /srv, doesn't find it, and fails with "No 'script_location'
# key found in configuration" instead of a normal file-not-found error.
docker compose exec -T api alembic -c app/backend/alembic.ini upgrade head

echo "==> Reloading nginx (picks up any config change without downtime)"
docker compose exec -T nginx nginx -s reload

echo "==> Pruning dangling images from previous builds"
docker image prune -f

echo "==> Done. Current stack:"
docker compose ps
