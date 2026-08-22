#!/usr/bin/env bash
set -euo pipefail

cd /home/shuang/myWorkspace/semantic_split_multimodal
export PATH=/home/shuang/miniconda3/envs/mpsl/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD
export MPLCONFIGDIR=/tmp/MSL_matplotlib

echo "[BASELINE] $(date)"
git branch --show-current
git rev-parse HEAD
git status --short
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("cuda_device_count", torch.cuda.device_count())'

echo "[PHASE 0A] prepare clients"
bash tools/data/launch_prepare_clients_all.sh

echo "[PHASE 0B] adaptive discovery artifacts"
bash tools/data/launch_discover_modalities_all.sh

echo "[PHASE 1] C1 formal discovery grid"
python experiments/run_all_discovery.py --results-root results/c1_formal_discovery

echo "[PHASE 2] C2 curve mode"
python experiments/msl/run_all.py --results-root results/c2_curve --methods randomsl ours oracle --device cuda --evaluation-mode curve --require-cuda

echo "[PHASE 3] aggregate convergence metrics"
python experiments/convergence.py --results-root results/c2_curve --methods randomsl ours oracle

echo "[PHASE 4] C2 formal final"
python experiments/msl/run_all.py --results-root results/c2_formal --methods randomsl ours oracle --device cuda --evaluation-mode formal --require-cuda

echo "[PHASE 5A] Fig.2 fingerprint PCA for MHEALTH/IEMOCAP"
python - <<'PY'
import torch
from experiments.common import find_clients_dir, find_discovery_dir, project_root, resolved_cfg
from tools.plot_fingerprint_embedding import rebuild_fingerprint_figure

root = project_root()
for dataset, fold, seed in [("mhealth", 1, 42), ("iemocap", 1, 42)]:
    cfg = resolved_cfg(dataset, fold, seed)
    clients_dir = find_clients_dir(root, cfg)
    discovery_dir = find_discovery_dir(root, clients_dir, "adaptive_isodata")
    outputs = rebuild_fingerprint_figure(cfg, clients_dir, discovery_dir, torch.device("cuda"))
    print(dataset, {key: str(value) for key, value in outputs.items()})
PY

echo "[PHASE 5B] Fig.3 convergence macro-F1 curves"
python - <<'PY'
import csv
from collections import defaultdict
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.common import DATASET_PROTOCOLS, formal_run_grid
from experiments.training import training_run_dir

root = Path("results/c2_curve")
out_dir = root / "convergence"
out_dir.mkdir(parents=True, exist_ok=True)
methods = ["randomsl", "ours", "oracle"]

for dataset in DATASET_PROTOCOLS:
    grouped = {method: defaultdict(list) for method in methods}
    for fold, seed in formal_run_grid(dataset):
        for method in methods:
            path = training_run_dir(root, dataset, fold, seed, method, None) / "test_curve.csv"
            with path.open("r", newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    grouped[method][int(float(row["round"]))].append(float(row["test_macro_f1"]))
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for method in methods:
        rounds = sorted(grouped[method])
        means = [statistics.mean(grouped[method][round_idx]) for round_idx in rounds]
        ax.plot(rounds, means, marker="o", linewidth=1.4, markersize=3, label=method)
    ax.set_title(f"{dataset} convergence")
    ax.set_xlabel("Round")
    ax.set_ylabel("Test Macro-F1")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"fig3_{dataset}_macro_f1.png", dpi=220)
    fig.savefig(out_dir / f"fig3_{dataset}_macro_f1.pdf")
    plt.close(fig)
PY

echo "[DONE] $(date)"
