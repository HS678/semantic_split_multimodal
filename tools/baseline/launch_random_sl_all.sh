#!/usr/bin/env bash
#
# randomSL baseline 一键运行（四数据集串行）。
# 只运行 Stage 3（复用主线 Stage1/Stage2 产物），最后用现有汇总脚本汇总。
#
# 启动命令（在项目根目录执行）：
#   nohup bash tools/baseline/launch_random_sl_all.sh > "tools/baseline/random_sl_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
RESULTS_ROOT="local/results_baseline/randomSL"

run_stage3() {
  local config="$1" seed="$2"
  echo "[$(date '+%F %T')] randomSL stage3 config=${config} seed=${seed}"
  "$PYTHON" scripts/baseline_random_sl.py --config "$config" --seed "$seed"
}

echo "[$(date '+%F %T')] UCI-HAR (5 seeds)"
for seed in 101 202 303 404 505; do
  run_stage3 configs/baseline/randomSL/uci_har.config "$seed"
done

echo "[$(date '+%F %T')] MHEALTH (5 folds)"
for fold in 1 2 3 4 5; do
  run_stage3 "configs/baseline/randomSL/mhealth/fold${fold}.config" 42
done

echo "[$(date '+%F %T')] IEMOCAP (5 folds)"
for fold in 1 2 3 4 5; do
  run_stage3 "configs/baseline/randomSL/iemocap/fold${fold}.config" 42
done

echo "[$(date '+%F %T')] PAMAP2 (9 folds)"
for fold in 1 2 3 4 5 6 7 8 9; do
  run_stage3 "configs/baseline/randomSL/pamap2/fold${fold}.config" 42
done

echo "[$(date '+%F %T')] summarize"
"$PYTHON" scripts/summarize_results.py --results-root "$RESULTS_ROOT"
echo "[$(date '+%F %T')] randomSL all done."
