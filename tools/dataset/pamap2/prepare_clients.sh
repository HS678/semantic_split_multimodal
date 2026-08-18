#!/usr/bin/env bash
# 串行生成 PAMAP2 各 fold 的 client partition。
set -euo pipefail

for fold in 1 2 3 4 5 6 7 8; do
  python pipeline/prepare_clients.py --dataset pamap2 --fold $fold
done
