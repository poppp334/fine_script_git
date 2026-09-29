#!/usr/bin/env bash
set -euo pipefail

SKIP_SMOKE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    *) echo "unknown option: $1"; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "creating venv (python 3.12)..."
  uv venv --python 3.12 .venv
fi

echo "installing dependencies from requirements.lock..."
uv pip install --python .venv/bin/python -r requirements.lock

echo "checking GPU..."
.venv/bin/python - <<'PYEOF'
import torch

print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PYEOF

if [[ $SKIP_SMOKE -eq 0 ]]; then
  echo "running smoke test..."
  .venv/bin/python scripts/smoke_test.py
fi

echo "setup done"
