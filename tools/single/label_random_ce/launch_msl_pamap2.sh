#!/usr/bin/env bash
#
# PAMAP2 label_random_ce 对比实验：9 折，每折 Stage1 → Stage2 → Stage3 → 汇总。
# 配置：configs/label_random_ce/pamap2/foldN.config
#
# 启动命令（在项目根目录执行）：
#   nohup bash tools/single/label_random_ce/launch_msl_pamap2.sh > "tools/single/label_random_ce/pamap2_lrce_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../../.."
source "$(dirname "$0")/../../lib/msl_common.sh"

CONFIG_DIR=configs/label_random_ce/pamap2
for fold in $(seq 1 9); do
  config="${CONFIG_DIR}/fold${fold}.config"
  stage1 pamap2 "$config"
  stage2 pamap2 "$config"
  stage3 pamap2 "$config" 42
done
summarize pamap2
echo "[$(date '+%F %T')] pamap2 label_random_ce all done."
