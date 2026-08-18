#!/usr/bin/env bash
set -euo pipefail

for fold in 1 2 3 4 5 6 7 8; do
  python pipeline/discover_modalities.py --dataset pamap2 --fold $fold --device cuda
done
