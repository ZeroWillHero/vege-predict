#!/usr/bin/env bash
# One-time: obtain a Let's Encrypt cert once your domain's DNS points at this
# server and the stack (deploy/deploy.sh) is already running on port 80.
#
# Usage: deploy/enable-tls.sh your-domain.example you@example.com
set -euo pipefail

DOMAIN="${1:?usage: enable-tls.sh <domain> <email>}"
EMAIL="${2:?usage: enable-tls.sh <domain> <email>}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "==> Requesting certificate for ${DOMAIN}"
docker run --rm \
  -v vegepredict_certbot_www:/var/www/certbot \
  -v vegepredict_certbot_certs:/etc/letsencrypt \
  certbot/certbot certonly --webroot \
  -w /var/www/certbot \
  -d "${DOMAIN}" \
  --email "${EMAIL}" --agree-tos --non-interactive

echo "==> Cert obtained. Now:"
echo "    1. Edit deploy/nginx/nginx.conf: uncomment the 443 server block,"
echo "       replace 'your-domain.example' with ${DOMAIN}."
echo "    2. Re-run deploy/deploy.sh to reload nginx with the new config."
echo "    Renewal: re-run this script periodically (certs expire every 90 days),"
echo "    or add a cron entry calling 'certbot renew' against the same volumes."
