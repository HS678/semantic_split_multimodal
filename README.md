# MSL

Semantic-aligned Distributed Split Multimodal Learning in Unknown Modality Environments.

中文说明见 [README.zh-CN.md](README.zh-CN.md)。

## Overview

Three-stage reproducible experiment framework:

1. **Stage 1 (partition)** — `scripts/MSL/stage1_partition.py` splits naturally paired multimodal data into single-modality clients. Only the train split is partitioned; the test split stays naturally paired (`test_multimodal.pt`).
2. **Stage 2 (discovery)** — `scripts/MSL/stage2_discovery.py` pretrains one client encoder per client, extracts fingerprints, clusters them with adaptive ISODATA, and writes `pred_cluster.csv` (plus `true_cluster.csv` for audit).
3. **Stage 3 (training)** — `scripts/MSL/stage3_train.py` runs cluster-aware scheduling, label-guided semantic pseudo binding, `ClusterAdapter` + concat fusion, and split learning with fixed rounds. There is no validation set: after `global_rounds`, `last_model.pt` is evaluated on `test_multimodal.pt` exactly once.

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

- **UCI-HAR**: the official `UCI HAR Dataset` (put the `train/` and `test/` folders under `local/datasets/UCI-HAR/`).
- **MHEALTH**: `MHEALTHDATASET.zip` from UCI, extracted as `local/datasets/MHEALTH/` with `MHEALTHDATASET/` inside.
- **PAMAP2**: `PAMAP2_Dataset.zip` from UCI, extracted as `local/datasets/PAMAP2/` with `PAMAP2_Dataset/Protocol/subject10*.dat`.
- **IEMOCAP**: the full release under `local/datasets/IEMOCAP/IEMOCAP_full/IEMOCAP_full_release/` (requires the CMU IEMOCAP license).

IEMOCAP frozen features must be prepared once before Stage 1:

```bash
PYTHONPATH=src python -m MSL.data.prepare_iemocap --device cuda
```

## Config

Each script can load dataset defaults directly with `--dataset`. Use `--print-config` to inspect the complete resolved parameters for a dataset:

```bash
python scripts/MSL/stage3_train.py --dataset mhealth --fold 1 --print-config
python scripts/baseline/randomSL/stage3_train.py --dataset pamap2 --fold 2 --print-config
```

Command-line options override dataset defaults:

```bash
python scripts/MSL/stage3_train.py --dataset uci_har --seed 101 --global-rounds 50 --client-lr 0.0001
```

Dataset defaults are centralized in `src/MSL/data/dataset_defaults.py`; the official CLI parser is `src/MSL/utils/experiment_args.py`. Multi-fold datasets use `--fold N` to derive the split protocol from dataset defaults, and command-line options override those defaults.

Use `--print-config` to inspect the full resolved parameters before running:

```bash
python scripts/MSL/stage3_train.py --dataset mhealth --fold 1 --print-config
```

All output paths are generated automatically from `base_dir` + dataset + split protocol. Mainline outputs default to `results/MSL/`; baseline outputs default to `results/baseline/randomSL/`. Each run writes `resolved_config.json` next to its results.

## Running

Prepare reusable Stage1 and Stage2 artifacts first. Stage2 requires the Stage1 directory to exist.

```bash
bash tools/dataset/uci_har/stage1.sh
bash tools/dataset/uci_har/stage2.sh
```

For multi-fold datasets, the dataset scripts contain explicit fold loops:

```bash
bash tools/dataset/mhealth/stage1.sh
bash tools/dataset/mhealth/stage2.sh
bash tools/dataset/pamap2/stage1.sh
bash tools/dataset/pamap2/stage2.sh
bash tools/dataset/iemocap/stage1.sh
bash tools/dataset/iemocap/stage2.sh
```

After Stage1 and Stage2 exist, Stage3 can be run repeatedly with different seeds, losses, or attempts:

```bash
python scripts/MSL/stage3_train.py --dataset uci_har --seed 101
```

Stage3 launchers reuse existing Stage1/Stage2 artifacts and write only Stage3 results:

```bash
bash tools/launch_msl.sh
bash tools/launch_random_sl.sh
```

Stage3 launchers run jobs in parallel with `MAX_JOBS=2` by default. Use `MAX_JOBS=1 bash tools/launch_msl.sh` for serial execution or increase `MAX_JOBS` if resources allow. They fail fast if the required Stage1/Stage2 artifact directories are missing.

## Result layout

```text
results/MSL/
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
python scripts/MSL/summarize_results.py --results-root results/MSL
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
