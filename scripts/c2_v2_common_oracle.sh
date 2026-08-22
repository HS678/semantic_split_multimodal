#!/usr/bin/env bash
set -euo pipefail

cd /home/shuang/myWorkspace/semantic_split_multimodal
export PATH=/home/shuang/miniconda3/envs/mpsl/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD
export MPLCONFIGDIR=/tmp/MSL_matplotlib

V2_ROOT="results/c2_v2_common_oracle"
NEW_CURVE_ROOT="local/c2_v2_common_oracle_cache/curve"
NEW_FORMAL_ROOT="local/c2_v2_common_oracle_cache/formal"
NEW_METHODS=(kmeans2 kmeans3 kmeans4 kmeans5 auto_kmeans gmm_bic)

echo "[BASELINE] $(date)"
git branch --show-current
git rev-parse HEAD
git status --short
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("cuda_device_count", torch.cuda.device_count())'

if [ -d "$V2_ROOT" ] && [ "$(find "$V2_ROOT" -mindepth 1 -print -quit)" ]; then
  echo "Refusing to append into existing non-empty v2 root: $V2_ROOT" >&2
  exit 2
fi

echo "[PHASE 1] build explicit v2 manifest"
python experiments/c2_v2_artifacts.py \
  --output-root "$V2_ROOT" \
  --legacy-curve-root results/c2_curve \
  --legacy-formal-root results/c2_formal \
  --new-curve-root "$NEW_CURVE_ROOT" \
  --new-formal-root "$NEW_FORMAL_ROOT" \
  --action manifest

echo "[PHASE 2] run missing C2 v2 curve methods"
python experiments/msl/run_all.py \
  --results-root "$NEW_CURVE_ROOT" \
  --methods "${NEW_METHODS[@]}" \
  --device cuda \
  --evaluation-mode curve \
  --require-cuda

echo "[PHASE 3] run missing C2 v2 formal methods"
python experiments/msl/run_all.py \
  --results-root "$NEW_FORMAL_ROOT" \
  --methods "${NEW_METHODS[@]}" \
  --device cuda \
  --evaluation-mode formal \
  --require-cuda

echo "[PHASE 4] export compact v2 artifacts"
python experiments/c2_v2_artifacts.py \
  --output-root "$V2_ROOT" \
  --legacy-curve-root results/c2_curve \
  --legacy-formal-root results/c2_formal \
  --new-curve-root "$NEW_CURVE_ROOT" \
  --new-formal-root "$NEW_FORMAL_ROOT" \
  --action export

python experiments/c2_v2_artifacts.py \
  --output-root "$V2_ROOT" \
  --legacy-curve-root results/c2_curve \
  --legacy-formal-root results/c2_formal \
  --new-curve-root "$NEW_CURVE_ROOT" \
  --new-formal-root "$NEW_FORMAL_ROOT" \
  --action curve-manifest

echo "[PHASE 5] common Oracle target aggregation and tables"
python experiments/convergence.py \
  --output-root "$V2_ROOT" \
  --manifest "$V2_ROOT/common_targets/run_manifest.json"

echo "[PHASE 6] artifact audit"
python experiments/c2_v2_artifacts.py \
  --output-root "$V2_ROOT" \
  --legacy-curve-root results/c2_curve \
  --legacy-formal-root results/c2_formal \
  --new-curve-root "$NEW_CURVE_ROOT" \
  --new-formal-root "$NEW_FORMAL_ROOT" \
  --action audit

echo "[DONE] $(date)"
