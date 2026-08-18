#!/usr/bin/env bash
# 串行运行 PAMAP2 各 fold 的模态发现。
set -euo pipefail

for fold in 1 2 3 4 5 6 7 8; do
  python pipeline/discover_modalities.py --dataset pamap2 --fold $fold --device cuda
done
