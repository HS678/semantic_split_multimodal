# MSL

Semantic-aligned Distributed Split Multimodal Learning in Unknown Modality Environments.

中文说明见 [README.zh-CN.md](README.zh-CN.md)。完整设计决策见 [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md)。

## Overview

Three-stage reproducible experiment framework:

1. **Stage 1 (partition)** — `scripts/stage1_partition.py` splits naturally paired multimodal data into single-modality clients. Only the train split is partitioned; the test split stays naturally paired (`test_multimodal.pt`).
2. **Stage 2 (discovery)** — `scripts/stage2_discovery.py` pretrains one client encoder per client, extracts fingerprints, clusters them with adaptive ISODATA, and writes `pred_cluster.csv` (plus `true_cluster.csv` for audit).
3. **Stage 3 (training)** — `scripts/stage3_train.py` runs cluster-aware scheduling, label-guided semantic pseudo binding, `ClusterAdapter` + concat fusion, and split learning with fixed rounds. There is no validation set: after `global_rounds`, `last_model.pt` is evaluated on `test_multimodal.pt` exactly once.

This is not Federated Learning and does not use FedAvg. Clients upload detached activations; the server computes the loss, backpropagates through the fusion model, and routes activation gradients back to the client encoders.

## Installation

Python 3.10+ with PyTorch. Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

For CUDA, install the matching `torch` / `torchvision` / `torchaudio` build first, then install the remaining requirements.

## Protocol

- Training, scheduling, binding, and fusion-slot construction use only `pred_cluster` and labels. `hidden_modality_id` / true modality names are used only for Stage 2 discovery audit and evaluation-only oracle mapping.
- No validation set: fixed `global_rounds=200`, no early stopping, no best-checkpoint selection.
- Official metrics: `acc`, `macro_f1`, `weighted_f1`.
- Config switch `training.cluster_assignment_source=true_cluster` selects an oracle upper-bound run; the no-leakage main line is `pred_cluster`.
- D2D is not implemented yet; `d2d.enabled=false` is kept as an extension hook.

## Datasets

Four datasets are supported; splits are hardcoded in `src/MSL/data/datasets.py` / `src/MSL/data/iemocap.py`:

| Dataset | Split protocol | Notes |
| --- | --- | --- |
| UCI-HAR | `subject_disjoint_70_30` | Official 70/30 fixed split; 5 seeds |
| MHEALTH | `subject_5fold_foldN` | 5-fold subject CV; 1 seed |
| PAMAP2 | `subject_9fold_loso_foldN` | 9-fold LOSO, 12 activities, no heart rate; 1 seed |
| IEMOCAP | `session_5fold_loso_foldN` | 5-fold session-LOSO, audio/video/text; 1 seed |

Per-dataset fixed parameters (num_classes, roots, encoders, pretrain/train lr, mmbind weights, clustering defaults) live in `src/MSL/data/dataset_defaults.py` and are not duplicated in config files.

Download the public datasets and place them under `local/datasets/`:

- **UCI-HAR**: the official `UCI HAR Dataset` (put the `train/` and `test/` folders under `local/datasets/uci_har/`).
- **MHEALTH**: `MHEALTHDATASET.zip` from UCI, extracted as `local/datasets/mhealth/` with `MHEALTHDATASET/` inside.
- **PAMAP2**: `PAMAP2_Dataset.zip` from UCI, extracted as `local/datasets/pamap2/` with `PAMAP2_Dataset/Protocol/subject10*.dat`.
- **IEMOCAP**: the full release under `local/datasets/IEMOCAP/IEMOCAP_full/IEMOCAP_full_release/` (requires the CMU IEMOCAP license).

IEMOCAP frozen features must be prepared once before Stage 1:

```bash
PYTHONPATH=src python -m MSL.data.prepare_iemocap --device cuda
```

## Config

`configs/` contains one self-contained config per dataset/fold (about 14 lines). Every config uses the same sections:

```ini
[config]      # experiment_name, base_dir (seed/device built-in; --seed overrides)
[partition]   # type, split_protocol, clients_per_modality
[train]       # cluster_assignment_source, scheduler, fusion_training_objective, global_rounds
# cluster / d2d / other sections keep built-in defaults; write them only when overriding.
```

All output paths are generated automatically from `base_dir` + dataset + split protocol. `configs/config.config` documents every field.

## Running

Run one stage:

```bash
python scripts/stage1_partition.py --config configs/uci_har.config
python scripts/stage2_discovery.py --config configs/uci_har.config
python scripts/stage3_train.py --config configs/uci_har.config --seed 101
```

One-command launchers (each dataset runs its full Stage1 → Stage2 → Stage3 → summarize flow):

```bash
nohup bash tools/single/launch_msl_uci_har.sh > "tools/single/uci_har_msl_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &
nohup bash tools/serial/launch_msl_all.sh > "tools/serial/msl_all_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &
nohup bash tools/parallel/launch_msl_parallel.sh > "tools/parallel/main_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &
```

Launchers skip steps whose output directories already exist (resume-friendly) and write logs under `local/results_msl/logs/`.

## Result layout

```text
local/results_msl/
├── partition/<dataset>/<partition_signature>/   # Stage 1 (train_clients/, test_multimodal.pt, ...)
├── cluster/<dataset>/<partition_signature>/adaptive_isodata/   # Stage 2 (+ visualization/)
├── experiments/<scope>/<dataset>/<loss>/attempt-<nn>/   # Stage 3 runs
│   ├── seed-<ss>/          # fixed-split datasets (e.g. UCI-HAR, one dir per seed)
│   ├── fold-<n>/           # multi-fold datasets (one dir per fold)
│   └── summary.json        # aggregates every seed/fold under this attempt
└── summary/<loss>/<dataset>.json  # per-dataset aggregate, grouped by loss
```

Stage outputs never overwrite an existing non-empty directory. Re-running the same fold/seed creates the next `attempt-<nn>` automatically.

## Summary format

```bash
python scripts/summarize_results.py --results-root local/results_msl
```

```json
{
  "fold1": {"acc": 0.85, "macro_f1": 0.81, "weighted_f1": 0.84},
  "fold2": {"acc": 0.87, "macro_f1": 0.82, "weighted_f1": 0.85},
  "average": {"acc": 0.86, "macro_f1": 0.815, "weighted_f1": 0.845}
}
```

## Tests

```bash
PYTHONPATH=src python -m pytest tests -q
```
