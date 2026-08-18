#!/usr/bin/env bash
set -euo pipefail

for fold in 1 2 3 4 5 6 7 8; do
  python pipeline/prepare_clients.py --dataset pamap2 --fold $fold
done
