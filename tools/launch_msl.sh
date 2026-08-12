#!/usr/bin/env bash
#
# Unified MSL launcher.
#
# Examples:
#   PYTHON=/home/shuang/miniconda3/envs/mpsl/bin/python bash tools/launch_msl.sh --dataset uci_har
#   PYTHON=/home/shuang/miniconda3/envs/mpsl/bin/python bash tools/launch_msl.sh --dataset all
#   PYTHON=/home/shuang/miniconda3/envs/mpsl/bin/python bash tools/launch_msl.sh --dataset pamap2 --clients 20

set -euo pipefail
cd "$(dirname "$0")/.."
source "tools/lib/msl_common.sh"

dataset="all"
clients="10"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset)
      dataset="$2"
      shift 2
      ;;
    --clients)
      clients="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: bash tools/launch_msl.sh [--dataset all|uci_har|iemocap|mhealth|pamap2] [--clients N]"
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [ "$dataset" = "all" ]; then
  for item in uci_har iemocap mhealth pamap2; do
    echo "[$(date '+%F %T')] ===== start: ${item} ====="
    run_msl_dataset "$item" "$clients"
    echo "[$(date '+%F %T')] ===== done: ${item} ====="
  done
else
  run_msl_dataset "$dataset" "$clients"
fi

echo "[$(date '+%F %T')] MSL launcher finished."
