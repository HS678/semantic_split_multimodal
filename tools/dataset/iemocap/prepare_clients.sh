#!/usr/bin/env bash
set -euo pipefail

for fold in 1 2 3 4 5; do
  python pipeline/prepare_clients.py --dataset iemocap --fold $fold
done
