#!/usr/bin/env bash
# Run this ON the server, from the repo root, to (re)deploy the vegepredict
# stack using a PRE-BUILT api image from GHCR instead of building locally.
#
# Unlike deploy/deploy.sh (which does `docker compose build` on this box),
# this script only pulls images and starts/updates the stack — the api image
# was already built and pushed to ghcr.io by the GitHub Actions workflow's
# build-and-push job, on a GitHub-hosted runner. This is what
# .github/workflows/deploy.yml now calls; keeps EC2 from ever needing to
# build the image itself (sidesteps needing a full source checkout on the
# box, and any local build-toolchain/Docker-build issues).
#
# Requires API_IMAGE to be set to the exact image reference to deploy (the
# workflow passes ghcr.io/<owner>/<repo>-api:<commit-sha>). If the GHCR
# package is private, GHCR_TOKEN (a token with read:packages, e.g. the
# workflow's own GITHUB_TOKEN) must also be set so this can `docker login`.
#
# Usage: API_IMAGE=ghcr.io/owner/repo-api:sha GHCR_TOKEN=... GHCR_USER=... deploy/pull-and-deploy.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${API_IMAGE:?API_IMAGE must be set to the image reference to deploy}"

if [ -n "${GHCR_TOKEN:-}" ]; then
  echo "==> Logging in to ghcr.io"
  echo "$GHCR_TOKEN" | docker login ghcr.io -u "${GHCR_USER:-github-actions}" --password-stdin
fi

echo "==> Pulling images (api=$API_IMAGE)"
docker compose pull api
docker compose pull postgres redis nginx

echo "==> Starting/updating stack"
docker compose up -d --no-build

echo "==> Waiting for postgres/redis health checks"
until [ "$(docker inspect -f '{{.State.Health.Status}}' vegepredict-postgres)" = "healthy" ]; do sleep 2; done
until [ "$(docker inspect -f '{{.State.Health.Status}}' vegepredict-redis)" = "healthy" ]; do sleep 2; done

echo "==> Running database migrations"
docker compose exec -T api alembic upgrade head

echo "==> Reloading nginx (picks up any config change without downtime)"
docker compose exec -T nginx nginx -s reload

echo "==> Pruning dangling images from previous pulls"
docker image prune -f

echo "==> Done. Current stack:"
docker compose ps
