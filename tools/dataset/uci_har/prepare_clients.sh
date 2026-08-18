#!/usr/bin/env bash
# 串行生成 UCI-HAR 各正式 seed 的 client partition。
set -euo pipefail

for seed in 42 123 2025 3407 7777; do
  python pipeline/prepare_clients.py \
    --dataset uci_har \
    --seed $seed
done
