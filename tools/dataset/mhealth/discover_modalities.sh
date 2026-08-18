#!/usr/bin/env bash
# 串行运行 MHEALTH 各 fold 的模态发现。
set -euo pipefail

for fold in 1 2 3 4 5; do
  python pipeline/discover_modalities.py --dataset mhealth --fold $fold --device cuda
done
