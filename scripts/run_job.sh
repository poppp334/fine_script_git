#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: run_job.sh --config <config.yaml> [--prompts FILE] [--regression FILE] [--gguf] [--package] [--dry-run]"
}

CONFIG=""
PROMPTS=""
REGRESSION=""
DO_GGUF=0
DO_PACKAGE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --prompts) PROMPTS="$2"; shift 2 ;;
    --regression) REGRESSION="$2"; shift 2 ;;
    --gguf) DO_GGUF=1; shift ;;
    --package) DO_PACKAGE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) usage; exit 2 ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  usage
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "config not found: $CONFIG" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
if [[ -x "$REPO_DIR/.venv/bin/python" ]]; then
  PY="$REPO_DIR/.venv/bin/python"
else
  PY="python3"
fi

read -r OUTPUT_DIR CLIENT DATASET_PATH < <("$PY" - "$CONFIG" <<'PYEOF'
import sys

import yaml

cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print(cfg.get("output_dir", "output"), cfg.get("client", "client"), cfg["dataset_path"])
PYEOF
)

run() {
  echo "+ $*"
  if [[ $DRY_RUN -eq 0 ]]; then
    "$@"
  fi
}

cd "$REPO_DIR"

run "$PY" "$SCRIPT_DIR/validate_data.py" --data "$DATASET_PATH"
run "$PY" "$SCRIPT_DIR/train_lora.py" --config "$CONFIG"
run "$PY" "$SCRIPT_DIR/merge_lora.py" --adapter "$OUTPUT_DIR/lora_adapter" --output "$OUTPUT_DIR/merged"

if [[ $DO_GGUF -eq 1 ]]; then
  run "$PY" "$SCRIPT_DIR/export_gguf.py" --adapter "$OUTPUT_DIR/lora_adapter" --output "$OUTPUT_DIR/gguf"
fi

if [[ -n "$PROMPTS" ]]; then
  eval_args=(--config "$CONFIG" --prompts-file "$PROMPTS" --out-dir "$OUTPUT_DIR/eval")
  if [[ -n "$REGRESSION" ]]; then
    eval_args+=(--regression-file "$REGRESSION")
  fi
  run "$PY" "$SCRIPT_DIR/evaluate.py" "${eval_args[@]}"
fi

if [[ $DO_PACKAGE -eq 1 ]]; then
  run "$PY" "$SCRIPT_DIR/package_deliverables.py" --run-dir "$OUTPUT_DIR" --client "$CLIENT" --out "deliverable_${CLIENT}.zip"
fi

echo "job finished: $CLIENT"
