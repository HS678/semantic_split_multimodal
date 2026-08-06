#!/usr/bin/env bash
#
# UCI-HAR 单数据集运行：Stage1 → Stage2 → Stage3（5 个 seed）→ 汇总。
#
# 启动命令（在项目根目录执行）：
#   nohup bash tools/single/launch_msl_uci_har.sh > "tools/single/uci_har_msl_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &
#
# 查看进度：tail -f "$(ls -t tools/single/uci_har_msl_*.log | head -1)"

set -euo pipefail
cd "$(dirname "$0")/../.."
source "$(dirname "$0")/../lib/msl_common.sh"
run_msl_dataset uci_har
echo "[$(date '+%F %T')] uci_har all done."
