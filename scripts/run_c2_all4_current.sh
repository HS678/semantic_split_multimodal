#!/usr/bin/env bash
set -euo pipefail

cd /home/shuang/myWorkspace/semantic_split_multimodal

export PATH=/home/shuang/miniconda3/envs/mpsl/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD
export MPLCONFIGDIR=/tmp/MSL_matplotlib

mkdir -p logs results results/c2_v2_common_oracle/common_targets

[ -e results/pipeline ] || ln -s ../results-old/pipeline results/pipeline
[ -e results/c1_formal_discovery ] || ln -s ../results-old/c1_formal_discovery results/c1_formal_discovery

test -d results/pipeline/clients
test -d results/pipeline/discovery
test -d results/c1_formal_discovery/discovery

echo "[BASELINE] $(date)"
git branch --show-current
git rev-parse HEAD
git status --short
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("cuda_device_count", torch.cuda.device_count())'

METHODS=(randomsl kmeans2 kmeans3 kmeans4 kmeans5 auto_kmeans gmm_bic ours)

echo "[PHASE 1] curve all 4 datasets parallel"
pids=()
for D in uci_har mhealth pamap2 iemocap; do
  python experiments/msl/run_all.py \
    --results-root results/c2_curve \
    --datasets "$D" \
    --methods "${METHODS[@]}" \
    --device cuda \
    --evaluation-mode curve \
    --require-cuda \
    --retry-failed \
    > "logs/c2_curve_${D}_current.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

echo "[PHASE 2] formal all 4 datasets parallel"
pids=()
for D in uci_har mhealth pamap2 iemocap; do
  python experiments/msl/run_all.py \
    --results-root results/c2_formal \
    --datasets "$D" \
    --methods "${METHODS[@]}" \
    --device cuda \
    --evaluation-mode formal \
    --require-cuda \
    --retry-failed \
    > "logs/c2_formal_${D}_current.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

echo "[PHASE 3] build explicit common-target manifest"
python - <<'PY'
import json
from pathlib import Path

from experiments.common import DATASET_PROTOCOLS, formal_run_grid
from experiments.convergence import C2_V2_METHODS, ORACLE_TARGET_METHOD
from experiments.training import training_run_dir

root = Path(".").resolve()
out = root / "results/c2_v2_common_oracle/common_targets"
out.mkdir(parents=True, exist_ok=True)

records = []
methods = list(C2_V2_METHODS) + [ORACLE_TARGET_METHOD]

for dataset in DATASET_PROTOCOLS:
    for fold, seed in formal_run_grid(dataset):
        for method in methods:
            curve_root = root / "results-old/c2_curve" if method == ORACLE_TARGET_METHOD else root / "results/c2_curve"
            run_dir = training_run_dir(curve_root, dataset, fold, int(seed), method, None)
            records.append({
                "dataset": dataset,
                "fold": fold,
                "seed": int(seed),
                "method": method,
                "evaluation_mode": "curve",
                "source_root": str(curve_root),
                "run_dir": str(run_dir),
                "curve_file": str(run_dir / "test_curve.csv"),
                "source_kind": "oracle_target_source" if method == ORACLE_TARGET_METHOD else "current_run",
                "v2_output_root": str(root / "results/c2_v2_common_oracle"),
            })

path = out / "run_manifest.json"
with path.open("w", encoding="utf-8") as handle:
    json.dump({"runs": records}, handle, indent=2, sort_keys=True)
print(path)
PY

echo "[PHASE 4] common Oracle target aggregation"
python experiments/convergence.py \
  --output-root results/c2_v2_common_oracle \
  --manifest results/c2_v2_common_oracle/common_targets/run_manifest.json

echo "[DONE] $(date)"
