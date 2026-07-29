# Sourced by deploy/train.sh and deploy/seed.sh — not meant to be run directly.
# Ensures uv is installed, then ensures the given venv exists with its
# requirements installed. Cheap to call every run: `uv pip install` no-ops
# fast when nothing changed.
#
# Usage: ensure_venv <venv_dir> <requirements_file>
ensure_venv() {
  local venv_dir="$1"
  local requirements_file="$2"

  if ! command -v uv >/dev/null 2>&1; then
    echo "==> uv not found, installing"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi

  if [ ! -d "$venv_dir" ]; then
    echo "==> Creating venv at $venv_dir"
    uv venv "$venv_dir" --python 3.13
  fi

  echo "==> Installing/syncing $requirements_file into $venv_dir"
  uv pip install -r "$requirements_file" --python "$venv_dir/bin/python"
}
