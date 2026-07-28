#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_stage1_partitions.sh <dataset|all>

Datasets:
  uci_har
  mhealth
  pamap2
  all

Examples:
  scripts/run_stage1_partitions.sh uci_har
  scripts/run_stage1_partitions.sh all

Override interpreter:
  PYTHON_BIN=/path/to/python scripts/run_stage1_partitions.sh all
EOF
}

run_one() {
  local dataset="$1"
  local config="${ROOT}/configs/${dataset}.yaml"
  if [[ ! -f "${config}" ]]; then
    echo "Missing config: ${config}" >&2
    exit 1
  fi
  echo "== Stage 1 partition: ${dataset} =="
  "${PYTHON_BIN}" "${ROOT}/scripts/stage1_partition.py" --config "${config}"
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

case "$1" in
  uci_har|mhealth|pamap2)
    run_one "$1"
    ;;
  all)
    run_one uci_har
    run_one mhealth
    run_one pamap2
    ;;
  -h|--help)
    usage
    ;;
  *)
    echo "Unknown dataset: $1" >&2
    usage
    exit 2
    ;;
esac
