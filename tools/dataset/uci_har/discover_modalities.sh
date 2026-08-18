#!/usr/bin/env bash
# 串行运行 UCI-HAR 各正式 seed 的模态发现。
set -euo pipefail

for seed in 42 123 2025 3407 7777; do
  python pipeline/discover_modalities.py \
    --dataset uci_har \
    --seed $seed \
    --device cuda
done
