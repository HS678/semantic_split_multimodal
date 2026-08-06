#!/usr/bin/env bash
#
# MSL 串行调度：顺序执行四个数据集的单数据集脚本（不并行）。
# 顺序：uci_har -> iemocap -> mhealth -> pamap2
#
# 启动命令（在项目根目录执行）：
#   nohup bash tools/serial/launch_msl_all.sh > "tools/serial/msl_all_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &
#
# 查看进度：tail -f "$(ls -t tools/serial/msl_all_*.log | head -1)"

set -u
cd "$(dirname "$0")/../.."
source "$(dirname "$0")/../lib/msl_common.sh"

for dataset in uci_har iemocap mhealth pamap2; do
  echo "[$(date '+%F %T')] ===== start: ${dataset} ====="
  run_msl_dataset "$dataset"
  echo "[$(date '+%F %T')] ===== done: ${dataset} ====="
done
echo "[$(date '+%F %T')] all datasets finished."
