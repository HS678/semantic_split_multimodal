#!/usr/bin/env bash
#
# MSL 批量实验脚本：按 数据集 × 客户端数 循环执行 Stage1 → Stage2 → Stage3 → 汇总。
# 所有产物输出到 local/results_msl/（由 config 的 base_dir 决定，路径自动生成）。
#
# 启动命令（在项目根目录执行）：
#   nohup bash tools/batch/launch_msl_batch.sh > "tools/batch/batch_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &
#
# 注意：不同客户端数会生成新的 partition 签名目录；Stage2/3 依赖 config 中的 stage1_dir，
# 客户端数 != 10 时需要对应的 config 变体（stage1_dir 指向该客户端数的 partition 目录）。

set -u
cd "$(dirname "$0")/../.."
source "$(dirname "$0")/../lib/msl_common.sh"

# ===== 实验参数（可修改） =====
DATASETS=(uci_har mhealth pamap2 iemocap)   # 数据集
CLIENTS=(10)                                # 客户端数变体，如 (10 20)

# ===== 批量执行：客户端数 × 数据集 =====
for clients in "${CLIENTS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    echo "[$(date '+%F %T')] ===== ${dataset} (clients_per_modality=${clients}) ====="
    run_msl_dataset "$dataset" "$clients"
  done
done
echo "[$(date '+%F %T')] all experiments finished."
