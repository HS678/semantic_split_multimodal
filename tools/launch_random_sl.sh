#!/usr/bin/env bash
#
# Unified randomSL baseline launcher. Reuses mainline Stage1/Stage2 artifacts.
#
# Examples:
#   PYTHON=/home/shuang/miniconda3/envs/mpsl/bin/python bash tools/launch_random_sl.sh --dataset all
#   PYTHON=/home/shuang/miniconda3/envs/mpsl/bin/python bash tools/launch_random_sl.sh --dataset mhealth

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
RESULTS_ROOT="local/results_baseline/randomSL"

dataset="all"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset)
      dataset="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: bash tools/launch_random_sl.sh [--dataset all|uci_har|iemocap|mhealth|pamap2]"
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

run_stage3() {
  local name="$1" config="$2" seed="$3" fold="${4:-}"
  local fold_args=()
  if [ -n "$fold" ]; then
    fold_args=(--fold "$fold")
  fi
  echo "[$(date '+%F %T')] randomSL stage3 dataset=${name} config=${config} seed=${seed}${fold:+ fold=${fold}}"
  "$PYTHON" scripts/baseline_random_sl.py --config "$config" --seed "$seed" "${fold_args[@]}"
}

run_dataset() {
  local name="$1"
  case "$name" in
    uci_har)
      for seed in 101 202 303 404 505; do
        run_stage3 uci_har configs/baseline/randomSL/uci_har.config "$seed"
      done
      ;;
    mhealth)
      for fold in 1 2 3 4 5; do
        run_stage3 mhealth configs/baseline/randomSL/mhealth.config 42 "$fold"
      done
      ;;
    iemocap)
      for fold in 1 2 3 4 5; do
        run_stage3 iemocap configs/baseline/randomSL/iemocap.config 42 "$fold"
      done
      ;;
    pamap2)
      for fold in 1 2 3 4 5 6 7 8 9; do
        run_stage3 pamap2 configs/baseline/randomSL/pamap2.config 42 "$fold"
      done
      ;;
    *)
      echo "unknown dataset: $name" >&2
      return 1
      ;;
  esac
}

if [ "$dataset" = "all" ]; then
  for item in uci_har mhealth iemocap pamap2; do
    run_dataset "$item"
  done
else
  run_dataset "$dataset"
fi

"$PYTHON" scripts/summarize_results.py --results-root "$RESULTS_ROOT"
echo "[$(date '+%F %T')] randomSL launcher finished."
