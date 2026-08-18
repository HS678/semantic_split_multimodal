#!/usr/bin/env bash
# 串行生成 MHEALTH 各 fold 的 client partition。
set -euo pipefail

for fold in 1 2 3 4 5; do
  python pipeline/prepare_clients.py --dataset mhealth --fold $fold
done
