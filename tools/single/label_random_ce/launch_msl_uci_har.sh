#!/usr/bin/env bash
#
# UCI-HAR label_random_ce 对比实验：Stage1 → Stage2 → Stage3（5 个 seed）→ 汇总。
# 配置：configs/label_random_ce/uci_har.config
#
# 启动命令（在项目根目录执行）：
#   nohup bash tools/single/label_random_ce/launch_msl_uci_har.sh > "tools/single/label_random_ce/uci_har_lrce_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../../.."
source "$(dirname "$0")/../../lib/msl_common.sh"

CONFIG=configs/label_random_ce/uci_har.config
stage1 uci_har "$CONFIG"
stage2 uci_har "$CONFIG"
for seed in 101 202 303 404 505; do
  stage3 uci_har "$CONFIG" "$seed"
done
summarize uci_har
echo "[$(date '+%F %T')] uci_har label_random_ce all done."
