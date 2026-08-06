#!/usr/bin/env bash
#
# MHEALTH 单数据集运行：5 折，每折 Stage1 → Stage2 → Stage3 → 汇总。
#
# 启动命令（在项目根目录执行）：
#   nohup bash tools/single/launch_msl_mhealth.sh > "tools/single/mhealth_msl_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &
#
# 查看进度：tail -f "$(ls -t tools/single/mhealth_msl_*.log | head -1)"

set -euo pipefail
cd "$(dirname "$0")/../.."
source "$(dirname "$0")/../lib/msl_common.sh"
run_msl_dataset mhealth
echo "[$(date '+%F %T')] mhealth all done."
