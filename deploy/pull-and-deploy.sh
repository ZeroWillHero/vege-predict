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

# The deploy SSH user is *supposed* to be in the `docker` group (see
# deploy/setup-ec2.md), but `usermod -aG docker` only takes effect on a
# session started after the change — a stale login (or a freshly created
# user whose group grant hasn't been re-applied yet) still gets
# "permission denied ... docker.sock" even though the group is correctly
# configured. Rather than fail the whole deploy on that, probe once and
# fall back to `sudo docker` (passwordless sudo is the default for the
# `ubuntu` user on AWS AMIs) so this script self-heals instead of requiring
# a manual re-login every time a box/user is set up.
if docker info >/dev/null 2>&1; then
  DOCKER="docker"
elif sudo -n docker info >/dev/null 2>&1; then
  echo "==> Note: current session can't reach the Docker socket directly (docker group not yet active for this login) - falling back to 'sudo docker'. Re-login once to pick up group membership and drop this fallback." >&2
  DOCKER="sudo docker"
else
  echo "==> ERROR: docker is unreachable both directly and via passwordless sudo." >&2
  echo "    Fix: sudo usermod -aG docker \$USER, then fully log out/in - or grant" >&2
  echo "    passwordless sudo for this user. See deploy/setup-ec2.md." >&2
  exit 1
fi

if [ -n "${GHCR_TOKEN:-}" ]; then
  echo "==> Logging in to ghcr.io"
  echo "$GHCR_TOKEN" | $DOCKER login ghcr.io -u "${GHCR_USER:-github-actions}" --password-stdin
fi

echo "==> Pulling images (api=$API_IMAGE)"
$DOCKER compose pull api
$DOCKER compose pull postgres redis nginx

echo "==> Starting/updating stack"
$DOCKER compose up -d --no-build

echo "==> Waiting for postgres/redis health checks"
until [ "$($DOCKER inspect -f '{{.State.Health.Status}}' vegepredict-postgres)" = "healthy" ]; do sleep 2; done
until [ "$($DOCKER inspect -f '{{.State.Health.Status}}' vegepredict-redis)" = "healthy" ]; do sleep 2; done

echo "==> Running database migrations"
$DOCKER compose exec -T api alembic upgrade head

echo "==> Reloading nginx (picks up any config change without downtime)"
$DOCKER compose exec -T nginx nginx -s reload

echo "==> Pruning dangling images from previous pulls"
$DOCKER image prune -f

echo "==> Done. Current stack:"
$DOCKER compose ps
