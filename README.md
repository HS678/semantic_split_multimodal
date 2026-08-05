# MSL

Semantic-aligned distributed Split Multimodal Learning in unknown modality environments.

中文说明见 [README.zh-CN.md](README.zh-CN.md).

## Overview

This repository provides a reproducible three-stage experiment framework:

1. Stage 1 partitions naturally paired multimodal data into single-modality clients.
2. Stage 2 discovers unknown client modality clusters from client encoder fingerprints.
3. Stage 3 trains an MMBind-style fusion Split Learning model from `pred_cluster`.

The project is not Federated Learning and does not use FedAvg. Clients upload detached activations. The server computes cross-entropy loss, backpropagates through the fusion model, and routes activation gradients back to client encoders.

## Protocol

Training forward/backward does not use true modality names, true modality IDs, true Q, or an oracle modality scheduler. `hidden_modality_id` is saved by Stage 1 only for post-hoc discovery audit and no-gradient naturally paired validation/test evaluation-only oracle mapping.

Stage 3 uses balanced per-cluster random round-robin scheduling, label-guided semantic pseudo binding, `ClusterAdapter`, concat fusion, the existing classifier, and Split Learning gradient return. It validates on `validation_multimodal.pt`, selects `best_model.pt` by validation weighted-F1, then evaluates `test_multimodal.pt` once after training. Validation/test labels are used only to compute loss, accuracy, macro-F1, and weighted-F1.

Supported clustering methods are:

- `kmeans`
- `adaptive_isodata`

## Installation

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Datasets are not included. Put raw datasets under:

```text
local/datasets/
```

Expected roots:

```text
local/datasets/uci_har/
local/datasets/mhealth/
local/datasets/pamap2/
local/datasets/IEMOCAP/IEMOCAP_full/IEMOCAP_full_release/
```

IEMOCAP uses the Full release and the four-class `angry / happy-or-excited / sad / neutral` protocol. Prepare its frozen MFCC, MobileViT-XS, and DistilBERT sequence features before Stage 1:

```bash
PYTHONPATH=src python -m MSL.data.prepare_iemocap --device cuda
```

The fixed split is Session 1-3 train, Session 4 validation, and Session 5 test. Audio uses three 1D convolution blocks followed by a GRU; video and text use GRUs over the frozen MobileViT-XS frame embeddings and DistilBERT token embeddings.

Local references may stay under `local/references/`. The whole `local/` tree is ignored by Git.

## Result Layout

Stage 1 partitions are reusable assets:

```text
local/results_msl/partition/<dataset>/<partition_signature>/
```

Stage 2 cluster outputs are separated by dataset, partition signature, and clustering method:

```text
local/results_msl/cluster/<dataset>/<partition_signature>/<cluster_method>/
```

Stage 3 training and evaluation outputs are full experiment runs:

```text
local/results_msl/experiments/<oracle_true_cluster|predicted_cluster>/<dataset>/<config_signature>/seed-<seed>/attempt-<nn>/
```

Default partition signatures with `clients_per_modality: 10`:

```text
UCI-HAR: local/results_msl/partition/uci_har/acc_10clients_gyro_10clients__subject_disjoint_tvt_v1
MHEALTH: local/results_msl/partition/mhealth/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients_ecg_10clients__subject_disjoint_tvt_v1
PAMAP2:  local/results_msl/partition/pamap2/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients__subject_disjoint_tvt_v1
IEMOCAP: local/results_msl/partition/iemocap/audio_10clients_video_10clients_text_10clients__session_disjoint_123_4_5_v1
```

No stage overwrites an existing non-empty output directory. Increment `stage3.attempt` in the `.config` when repeating the same config and seed.

## Stage 1

Run one dataset:

```bash
python scripts/stage1_partition.py --config configs/uci_har.config
```

Run all four datasets one by one:

```bash
python scripts/stage1_partition.py --config configs/uci_har.config
python scripts/stage1_partition.py --config configs/mhealth/fold1.config
python scripts/stage1_partition.py --config configs/pamap2/fold1.config
python scripts/stage1_partition.py --config configs/iemocap/fold1.config
```

Stage 1 writes directly under the partition directory:

```text
train_clients/client_*.pt
client_meta.csv
validation_multimodal.pt
test_multimodal.pt
partition_config.json
```

UCI-HAR, MHEALTH, and PAMAP2 use fixed, disjoint subject-level train/validation/test splits. Only train enters client partitioning and Stage 2. Validation/test remain naturally paired.

IEMOCAP uses 5-fold session-level leave-one-session-out splits with 5,531 four-class utterances. Its padded sequence lengths are propagated through Stage 1, encoder pretraining, fingerprint extraction, Split Learning, and naturally paired evaluation. `configs/iemocap/fold1.config` intentionally selects `true_cluster` for the requested oracle/debug comparison; it is not a no-leakage main result.

## Stage 2

UCI-HAR:

```bash
python scripts/stage2_discovery.py --config configs/uci_har.config
```

MHEALTH:

```bash
python scripts/stage2_discovery.py --config configs/mhealth/fold1.config
```

PAMAP2:

```bash
python scripts/stage2_discovery.py --config configs/pamap2/fold1.config
```

IEMOCAP:

```bash
python scripts/stage2_discovery.py --config configs/iemocap/fold1.config
```

Stage 2 keeps only:

```text
true_cluster.csv
pred_cluster.csv
pretrained_encoders/
fingerprints.npz
fingerprint_pca.pdf
fingerprint_pca.png
fingerprint_pca_metadata.json
stage2_metadata.json
```

`fingerprint_pca.pdf` is a publication-ready vector figure and `fingerprint_pca.png` is a 600-DPI preview. PCA coordinates are computed only from pre-clustering client fingerprints; true modalities and predicted clusters are used only to color the two post-hoc audit panels and never enter PCA fitting or clustering.

Stage 2 always produces `pred_cluster.csv`, PCA figures, and discovery audits. The current development configs deliberately select `true_cluster.csv` in Stage 3; the paper protocol will later switch to `pred_cluster` after tuning the existing clustering parameters.

## Stage 3

Stage 3 reads its frozen Stage 1/Stage 2 inputs, seed, output root, and attempt from `.config`. CLI path arguments remain optional debugging overrides.

Select the Stage 3 cluster assignment source with:

```ini
[training]
cluster_assignment_source=true_cluster
```

For Stage 3 debugging, set it to `true_cluster` to bypass predicted clustering. Training, scheduling, binding, fusion slots, and the evaluation mapping will then consistently read `true_cluster.csv`. This is an oracle/debug mode that uses true modality clusters; do not report it as the no-leakage main result, and use a clearly distinguishable `attempt`.

UCI-HAR:

```bash
python scripts/stage3_train.py --config configs/uci_har.config
```

MHEALTH:

```bash
python scripts/stage3_train.py --config configs/mhealth/fold1.config
```

PAMAP2:

```bash
python scripts/stage3_train.py --config configs/pamap2/fold1.config
```

IEMOCAP true-cluster Oracle/debug comparison:

```bash
python scripts/stage3_train.py --config configs/iemocap/fold1.config
```

Stage 3 writes under the config-signature/seed/attempt directory shown above:

```text
source_config.config
resolved_config.config
train_log.csv
validation_log.csv
final_metrics.json
best_metrics.json
best_model.pt
last_model.pt
training_curves.png
stage3_metadata.json
```

Formal configs train for at most 200 rounds, run naturally paired validation every 10 rounds, require at least 50 rounds, and early-stop after three validation checks without a weighted-F1 improvement greater than `0.001`. `best_model.pt` is the official validation-selected checkpoint; `last_model.pt` is diagnostic only. After training, Stage 3 reloads `best_model.pt`, evaluates test exactly once, and writes `final_metrics.json`.

Stage 3 generates `training_curves.png` automatically. To redraw it from existing CSV files:

```bash
PYTHONPATH=src /home/shuang/miniconda3/envs/mpsl/bin/python \
  -m MSL.evaluation.plot_training_curves \
  --run-dir local/results_msl/experiments/<cluster_scope>/<dataset>/<config_signature>/seed-<seed>/attempt-<nn>
```

The five formal Stage 3 seeds are `101`, `202`, `303`, `404`, and `505`. Change `seed` in `.config`; increment `stage3.attempt` only when repeating a seed.

```bash
python scripts/stage3_train.py --config configs/uci_har.config
```

All four datasets have independent launchers that run Stage 1, Stage 2, and all five Stage 3 seeds. With the current `true_cluster` configs, Stage 3 outputs are Oracle/debug results:

```bash
nohup bash local/tools/launch_uci_har_formal.sh \
  > "local/tools/uci_har_formal_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &

nohup bash local/tools/launch_mhealth_formal.sh \
  > "local/tools/mhealth_formal_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &

nohup bash local/tools/launch_pamap2_formal.sh \
  > "local/tools/pamap2_formal_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &

nohup bash local/tools/launch_iemocap_formal.sh \
  > "local/tools/iemocap_formal_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &
```

`launch_stage3_formal.sh` sequentially starts experiments for all four datasets:

```bash
nohup bash local/tools/launch_stage3_formal.sh \
  > "local/tools/formal_all_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &
```

It creates independent `seed-<seed>/attempt-01` directories and therefore does not overwrite old runs. The entire `local/` tree, including this launcher and its logs, is ignored by Git.

## Testing

```bash
PYTHONPATH=src python -m pytest tests -q
```
