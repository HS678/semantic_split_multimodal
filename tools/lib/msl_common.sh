#!/usr/bin/env bash
#
# MSL 实验公共运行库：Stage1/2 作为可复用公共产物，launcher 默认只跑 Stage3。
# 用法：先 cd 到项目根，再 source 本文件。

set -euo pipefail

PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"

# 执行命令；若因输出目录已存在而失败（防覆盖）则视为跳过，实现断点续跑。
run_or_skip() {
  local label="$1" log="$2"
  shift 2
  echo "[$(date '+%F %T')] ${label} start"
  if "$@" > "$log" 2>&1; then
    return 0
  fi
  if rg -q "Refusing to overwrite existing" "$log"; then
    echo "[$(date '+%F %T')] ${label} output already exists, skipping."
    return 0
  fi
  echo "${label} FAILED"
  tail -20 "$log"
  return 1
}

stage1() {
  local name="$1" config="$2" clients="${3:-10}" fold="${4:-}"
  local log_dir="results/MSL/logs/${name}"
  mkdir -p "$log_dir"
  local fold_tag=""
  local fold_args=()
  if [ -n "$fold" ]; then
    fold_tag="_fold${fold}"
    fold_args=(--fold "$fold")
  fi
  local log_file="${log_dir}/stage1_${name}${fold_tag}_$(date '+%Y%m%d_%H%M%S').log"
  run_or_skip "$name Stage1" "$log_file" "$PYTHON" \
    scripts/MSL/stage1_partition.py --config "$config" --clients "$clients" "${fold_args[@]}"
}
stage2() {
  local name="$1" config="$2" fold="${3:-}"
  local log_dir="results/MSL/logs/${name}"
  mkdir -p "$log_dir"
  local fold_tag=""
  local fold_args=()
  if [ -n "$fold" ]; then
    fold_tag="_fold${fold}"
    fold_args=(--fold "$fold")
  fi
  local log_file="${log_dir}/stage2_${name}${fold_tag}_$(date '+%Y%m%d_%H%M%S').log"
  run_or_skip "$name Stage2" "$log_file" "$PYTHON" scripts/MSL/stage2_discovery.py --config "$config" "${fold_args[@]}"
}
stage3() {
  local name="$1" config="$2" seed="$3" fold="${4:-}"
  local log_dir="results/MSL/logs/${name}"
  mkdir -p "$log_dir"
  local fold_tag=""
  local fold_args=()
  if [ -n "$fold" ]; then
    fold_tag="_fold${fold}"
    fold_args=(--fold "$fold")
  fi
  local log_file="${log_dir}/stage3_${name}${fold_tag}_seed${seed}_$(date '+%Y%m%d_%H%M%S').log"
  run_or_skip "$name Stage3" "$log_file" "$PYTHON" scripts/MSL/stage3_train.py --config "$config" --seed "$seed" "${fold_args[@]}"
}

summarize() {
  "$PYTHON" scripts/MSL/summarize_results.py --results-root results/MSL --dataset "$1"
}

# 固定划分数据集：只跑 Stage3，复用已存在的 Stage1/2。
run_fixed_dataset_stage3() {
  local name="$1" config="$2" clients="$3"
  shift 3
  local seed
  for seed in "$@"; do
    stage3 "$name" "$config" "$seed"
  done
  summarize "$name"
}

# 多折数据集：只跑 Stage3，复用每折已存在的 Stage1/2。
run_folds_dataset_stage3() {
  local name="$1" folds="$2" seed="$3" clients="$4"
  local config="configs/MSL/${name}.config"
  local fold
  for fold in $(seq 1 "$folds"); do
    stage3 "$name" "$config" "$seed" "$fold"
  done
  summarize "$name"
}

# Stage1/2 产物准备入口：按需手动调用，生成后可长期复用。
prepare_fixed_dataset_artifacts() {
  local name="$1" config="$2" clients="$3"
  stage1 "$name" "$config" "$clients"
  stage2 "$name" "$config"
}

prepare_folds_dataset_artifacts() {
  local name="$1" folds="$2" clients="$3"
  local config="configs/MSL/${name}.config"
  local fold
  for fold in $(seq 1 "$folds"); do
    stage1 "$name" "$config" "$clients" "$fold"
    stage2 "$name" "$config" "$fold"
  done
}

prepare_msl_artifacts() {
  local dataset="$1" clients="${2:-10}"
  case "$dataset" in
    uci_har) prepare_fixed_dataset_artifacts uci_har configs/MSL/uci_har.config "$clients" ;;
    iemocap) prepare_folds_dataset_artifacts iemocap 5 "$clients" ;;
    mhealth) prepare_folds_dataset_artifacts mhealth 5 "$clients" ;;
    pamap2)  prepare_folds_dataset_artifacts pamap2 9 "$clients" ;;
    *) echo "unknown dataset: $dataset" >&2; return 1 ;;
  esac
}

# Stage3 入口：运行该数据集的训练实验，要求 Stage1/2 产物已存在。
run_msl_stage3_dataset() {
  local dataset="$1" clients="${2:-10}"
  case "$dataset" in
    uci_har) run_fixed_dataset_stage3 uci_har configs/MSL/uci_har.config "$clients" 101 202 303 404 505 ;;
    iemocap) run_folds_dataset_stage3 iemocap 5 42 "$clients" ;;
    mhealth) run_folds_dataset_stage3 mhealth 5 42 "$clients" ;;
    pamap2)  run_folds_dataset_stage3 pamap2 9 42 "$clients" ;;
    *) echo "unknown dataset: $dataset" >&2; return 1 ;;
  esac
}
