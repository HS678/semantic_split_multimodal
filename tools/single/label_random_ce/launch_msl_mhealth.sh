#!/usr/bin/env bash
#
# MHEALTH label_random_ce 对比实验：5 折，每折 Stage1 → Stage2 → Stage3 → 汇总。
# 配置：configs/label_random_ce/mhealth/foldN.config
#
# 启动命令（在项目根目录执行）：
#   nohup bash tools/single/label_random_ce/launch_msl_mhealth.sh > "tools/single/label_random_ce/mhealth_lrce_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../../.."
source "$(dirname "$0")/../../lib/msl_common.sh"

CONFIG_DIR=configs/label_random_ce/mhealth
for fold in $(seq 1 5); do
  config="${CONFIG_DIR}/fold${fold}.config"
  stage1 mhealth "$config"
  stage2 mhealth "$config"
  stage3 mhealth "$config" 42
done
summarize mhealth
echo "[$(date '+%F %T')] mhealth label_random_ce all done."
