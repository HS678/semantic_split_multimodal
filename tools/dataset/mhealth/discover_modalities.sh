#!/usr/bin/env bash
set -euo pipefail

for fold in 1 2 3 4 5; do
  python pipeline/discover_modalities.py --dataset mhealth --fold $fold --device cuda
done
