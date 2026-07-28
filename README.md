# semantic_split_multimodal

Semantic-aligned distributed Split Multimodal Learning in unknown modality environments.

中文说明见 [README.zh-CN.md](README.zh-CN.md).

## Overview

This repository provides a reproducible three-stage experiment framework:

1. Stage 1 partitions naturally paired multimodal data into single-modality clients.
2. Stage 2 discovers unknown client modality clusters from client encoder fingerprints.
3. Stage 3 trains an MMBind-style fusion Split Learning model from `pred_cluster`.

The project is not Federated Learning and does not use FedAvg. Clients upload detached activations. The server computes cross-entropy loss, backpropagates through the fusion model, and routes activation gradients back to client encoders.

## Protocol

Training does not use true modality names, true modality IDs, true Q, or an oracle modality scheduler. `hidden_modality_id` is saved by Stage 1 only for post-hoc discovery audit and evaluation-only oracle mapping.

Stage 3 uses balanced per-cluster random round-robin scheduling, label-guided semantic pseudo binding, `ClusterAdapter`, concat fusion, the existing classifier, and Split Learning gradient return. Final evaluation reads the frozen Stage 1 `test_multimodal.pt`; test labels are used only to compute metrics.

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
```

Local references may stay under `local/references/`. The whole `local/` tree is ignored by Git.

## Result Layout

Stage 1 partitions are reusable assets:

```text
local/results/partition/<dataset>/<modality_1>_<n>clients_<modality_2>_<n>clients_.../
```

Stage 2 cluster outputs are separated by dataset, partition signature, and clustering method:

```text
local/results/cluster/<dataset>/<partition_signature>/<cluster_method>/
```

Stage 3 training and evaluation outputs are full experiment runs:

```text
local/results/experiments/<dataset>/<run_id>/
```

Default partition signatures with `clients_per_modality: 10`:

```text
UCI-HAR: local/results/partition/uci_har/acc_10clients_gyro_10clients
MHEALTH: local/results/partition/mhealth/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients_ecg_10clients
PAMAP2:  local/results/partition/pamap2/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients
```

No stage overwrites an existing non-empty output directory. Use a new Stage 3 `run_id`, such as `run_1`, `run_2`, or `adaptive_seed7`.

## Stage 1

Run one dataset:

```bash
python scripts/stage1_partition.py --config configs/uci_har.yaml
```

Run all three datasets one by one:

```bash
python scripts/stage1_partition.py --config configs/uci_har.yaml
python scripts/stage1_partition.py --config configs/mhealth.yaml
python scripts/stage1_partition.py --config configs/pamap2.yaml
```

Stage 1 writes directly under the partition directory:

```text
train_clients/client_*.pt
client_meta.csv
test_multimodal.pt
partition_config.json
```

## Stage 2

UCI-HAR:

```bash
python scripts/stage2_discovery.py \
  --config configs/uci_har.yaml \
  --stage1-dir local/results/partition/uci_har/acc_10clients_gyro_10clients \
  --output-root local/results/cluster \
  --run-type user_formal
```

MHEALTH:

```bash
python scripts/stage2_discovery.py \
  --config configs/mhealth.yaml \
  --stage1-dir local/results/partition/mhealth/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients_ecg_10clients \
  --output-root local/results/cluster \
  --run-type user_formal
```

PAMAP2:

```bash
python scripts/stage2_discovery.py \
  --config configs/pamap2.yaml \
  --stage1-dir local/results/partition/pamap2/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients \
  --output-root local/results/cluster \
  --run-type user_formal
```

Stage 2 keeps only:

```text
true_cluster.csv
pred_cluster.csv
pretrained_encoders/
stage2_metadata.json
```

Check `stage2_metadata.json` before Stage 3. A successful adaptive discovery should report `metrics.discovery_status: discovery_success`.

## Stage 3

Run Stage 3 from a frozen Stage 1 partition and a frozen Stage 2 cluster directory. Start with UCI-HAR, then run the larger datasets.

UCI-HAR:

```bash
python scripts/stage3_train.py \
  --config configs/uci_har.yaml \
  --stage1-dir local/results/partition/uci_har/acc_10clients_gyro_10clients \
  --stage2-dir local/results/cluster/uci_har/acc_10clients_gyro_10clients/adaptive_isodata \
  --output-root local/results/experiments \
  --run-id run_1 \
  --run-type user_formal
```

MHEALTH:

```bash
python scripts/stage3_train.py \
  --config configs/mhealth.yaml \
  --stage1-dir local/results/partition/mhealth/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients_ecg_10clients \
  --stage2-dir local/results/cluster/mhealth/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients_ecg_10clients/adaptive_isodata \
  --output-root local/results/experiments \
  --run-id run_1 \
  --run-type user_formal
```

PAMAP2:

```bash
python scripts/stage3_train.py \
  --config configs/pamap2.yaml \
  --stage1-dir local/results/partition/pamap2/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients \
  --stage2-dir local/results/cluster/pamap2/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients/adaptive_isodata \
  --output-root local/results/experiments \
  --run-id run_1 \
  --run-type user_formal
```

Stage 3 writes directly under `local/results/experiments/<dataset>/<run_id>/`:

```text
train_log.csv
eval_log.csv
final_metrics.json
best_metrics.json
best_model.pt
final_model.pt
stage3_metadata.json
```

Use `best_metrics.json` as the paper primary result. Keep `final_metrics.json` for final-round diagnostics.

## Testing

```bash
PYTHONPATH=src python -m pytest tests -q
```
