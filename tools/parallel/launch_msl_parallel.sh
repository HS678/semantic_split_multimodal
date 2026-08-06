#!/usr/bin/env bash
#
# MSL 并行调度：四个数据集同时运行（各数据集独立 Stage1 → Stage2 → Stage3 → 汇总）。
#
# 启动命令（在项目根目录执行）：
#   nohup bash tools/parallel/launch_msl_parallel.sh > "tools/parallel/main_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &
#
# 查看进度：tail -f "$(ls -t tools/parallel/main_*.log | head -1)"
#
# 注意：不要与其他 MSL 启动脚本同时运行；四路并行会同时占 GPU，OOM 时可在下方只留 2 个数据集。

set -u
cd "$(dirname "$0")/../.."
source "$(dirname "$0")/../lib/msl_common.sh"

# 冲突检测：已有 MSL 实验进程在运行（stage 脚本 或 single/serial 入口）则拒绝。
# 注意：检测目标是"实际干活"的进程，不含本并行脚本自身，避免 nohup 包装导致误报。
EXISTING=$(pgrep -f "scripts/stage[123]_(partition|discovery|train)\.py|tools/(single/launch_msl_(uci_har|iemocap|mhealth|pamap2)|serial/launch_msl_all)\.sh" || true)
if [ -n "$EXISTING" ]; then
  echo "ERROR: 检测到已有 MSL 启动脚本在运行，请先停止旧进程再启动并行脚本。"
  pgrep -af "scripts/stage[123]_|tools/(single|serial)/launch_msl" || true
  exit 1
fi

PIDS=()
for dataset in uci_har iemocap mhealth pamap2; do
  mkdir -p "local/results_msl/logs/${dataset}"
  local_log="local/results_msl/logs/${dataset}/parallel_${dataset}_$(date '+%Y%m%d_%H%M%S').log"
  echo "[$(date '+%F %T')] start: ${dataset} -> ${local_log}"
  run_msl_dataset "$dataset" > "$local_log" 2>&1 &
  PIDS+=("$!")
done

FAILED=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || FAILED=1
done

if [ "$FAILED" -ne 0 ]; then
  echo "some datasets failed, check local/results_msl/logs/<dataset>/parallel_*.log"
  exit 1
fi
echo "[$(date '+%F %T')] all datasets finished."
