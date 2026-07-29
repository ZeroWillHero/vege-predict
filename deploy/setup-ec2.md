# One-time EC2 setup

Do this once per EC2 instance, before the first `Train and Deploy` workflow run.
After this, every push to `main` (or a manual `workflow_dispatch`) trains,
builds, and deploys automatically — GitHub's own hosted runner SSHes in and
drives it (see `.github/workflows/deploy.yml`); nothing needs to be registered
as a GitHub Actions runner on the box itself.

## 1. Provision the instance

- Ubuntu 22.04/24.04 LTS, recommend at least 4 vCPU / 8GB RAM (LSTM/CatBoost
  training is the heaviest step; see CLAUDE.md's ~10-15 min full-retrain note).
- Security group: allow inbound 22 (SSH, ideally restricted to GitHub Actions'
  published IP ranges or your own IP), 80, 443. Do **not** open 5432/6379 to
  the world — Postgres/Redis only need to be reachable from this same host.

## 2. Create a deploy SSH key pair for GitHub Actions

Use a dedicated key (not your personal one) so it can be revoked independently:

```bash
# on your dev machine
ssh-keygen -t ed25519 -f deploy_key -C "github-actions-deploy" -N ""
```

Append `deploy_key.pub` to `~/.ssh/authorized_keys` for the user you'll SSH in
as on the instance (e.g. `ubuntu`).

## 3. Add GitHub repo secrets

Repo → Settings → Secrets and variables → Actions → New repository secret:

| Secret | Value |
|---|---|
| `EC2_HOST` | the instance's public IP or DNS name |
| `EC2_USER` | the SSH user (e.g. `ubuntu`) |
| `EC2_SSH_KEY` | the full contents of the **private** key file (`deploy_key`, the one that stays secret — never `deploy_key.pub`) |

## 4. Install Docker + Compose plugin

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
newgrp docker
docker compose version   # confirms the compose plugin is present
```

## 5. Create the repo directory

The workflow syncs tracked files into this path via `git archive | ssh ... tar
-x` — it does not `git clone` on your behalf, so create the directory once:

```bash
mkdir -p ~/Research-Project
```

## 6. Upload the raw source data

`data/raw/` is gitignored — the workflow never touches it, so seed it once:

```bash
# from your dev machine
rsync -avz data/raw/ <EC2_USER>@<EC2_HOST>:~/Research-Project/data/raw/
```

## 7. One manual sync + create both virtualenvs

The venvs need the tracked `requirements.txt` files present first, so do one
manual sync now (the workflow handles this automatically from here on):

```bash
# from your dev machine
git archive --format=tar HEAD | ssh <EC2_USER>@<EC2_HOST> 'tar -x -C ~/Research-Project'
```

```bash
# on the instance — two separate venvs, matching the project's existing local
# convention (root research pipeline vs. backend — see CLAUDE.md, don't mix them)
cd ~/Research-Project
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

python3 -m venv app/backend/.venv
app/backend/.venv/bin/pip install -r app/backend/requirements.txt
```

## 8. First run

```bash
chmod +x deploy/*.sh
deploy/train.sh    # ~10-15 min, writes trained_models/ + results/
deploy/deploy.sh   # brings up nginx/postgres/redis/api, runs migrations
deploy/seed.sh     # loads results into Postgres, warms/invalidates Redis
```

Confirm at `http://<ec2-host>/health`, then set up TLS once DNS points at the
instance (`deploy/enable-tls.sh`, see its usage comment).

From here on, pushes to `main` re-run all three steps automatically over SSH
via `.github/workflows/deploy.yml` — no further manual steps needed.
