# semantic_split_multimodal

This repository is a lightweight, single-process simulation framework for split multimodal learning on the UCI-HAR inertial-signal dataset.

The current project target is a strict three-stage pipeline:

1. Stage 1: `dataPartition`
2. Stage 2: `pretrain + clustering`
3. Stage 3: `split multimodal training`

D2D is not implemented in this phase. The config only reserves:

```yaml
d2d:
  enabled: false
```

## Dataset

Place UCI-HAR under `data/uci-har` by default. The expected structure is:

```text
data/uci-har/
  train/
    Inertial Signals/
      body_acc_x_train.txt
      body_acc_y_train.txt
      body_acc_z_train.txt
      total_acc_x_train.txt
      total_acc_y_train.txt
      total_acc_z_train.txt
      body_gyro_x_train.txt
      body_gyro_y_train.txt
      body_gyro_z_train.txt
    y_train.txt
  test/
    Inertial Signals/
      body_acc_x_test.txt
      body_acc_y_test.txt
      body_acc_z_test.txt
      total_acc_x_test.txt
      total_acc_y_test.txt
      total_acc_z_test.txt
      body_gyro_x_test.txt
      body_gyro_y_test.txt
      body_gyro_z_test.txt
    y_test.txt
```

If the path or required files are missing, Stage 1 raises a clear `FileNotFoundError` listing the missing paths.

## Stage 1: Data Partition

Run:

```bash
python experiments/stage1_partition.py --config configs/uci_har.yaml
```

Input:

- Raw UCI-HAR inertial signals from `dataset.root`.

Behavior:

- Builds two modalities:
  - `acc`: `body_acc_x/y/z + total_acc_x/y/z`
  - `gyro`: `body_gyro_x/y/z`
- Creates single-modality training clients. Each client owns exactly one modality.
- Creates paired multimodal test data containing `acc`, `gyro`, and `label`.

Output under `data_partition/`:

```text
data_partition/
  train_clients/
    client_000.pt
    client_001.pt
    ...
  test_multimodal.pt
  client_meta.csv
  partition_config.json
```

`client_meta.csv` contains `client_id, modality_id, modality_name, num_samples`. The true `modality_name` is for clustering evaluation only and is not used for Stage 3 training scheduling.

## Stage 2: Pretrain + Clustering

Run:

```bash
python experiments/stage2_pretrain_cluster.py --config configs/uci_har.yaml
```

Input:

- `data_partition/train_clients/client_*.pt` from Stage 1.

Behavior:

- Initializes one encoder per client.
- Runs local autoencoder-style representation learning.
- Extracts each fingerprint as the mean encoder output over several batches.
- Clusters fingerprints with `kmeans` or simplified `isodata`, controlled by `cluster.method`.
- Computes clustering accuracy, NMI, and ARI. True modality names are used only here for evaluation.

Output under `cluster/`:

```text
cluster/
  fingerprints.npy
  cluster_assignments.csv
  cluster_metrics.json
  pretrained_encoders/
  cluster_config.json
```

Synchronized output under `result/`:

```text
result/
  cluster_result.txt
  cluster_metrics.json
```

`cluster_assignments.csv` contains `client_id, true_modality, pred_cluster`.

## Stage 3: Split Multimodal Training

Run:

```bash
python experiments/stage3_train_sl.py --config configs/uci_har.yaml
```

Input:

- `data_partition/train_clients/client_*.pt`
- `data_partition/test_multimodal.pt`
- `cluster/cluster_assignments.csv`
- `cluster/pretrained_encoders/*.pt`

Behavior:

- Builds `cluster_to_clients` from `pred_cluster` only.
- Uses balanced cluster scheduling: each round samples `r = training.clients_per_cluster_per_round` clients from each predicted cluster.
- If the predicted cluster count is `Q_star`, each round uses `K_t = Q_star * r` clients.
- The scheduler returns a two-dimensional structure: `selected[cluster_id][group_id] = client`.
- Builds `r` modality-complete groups from selected predicted clusters.
- Concatenates group features in sorted `cluster_id` order before the server fusion classifier.
- Preserves `feature_map[(cluster_id, group_id)] -> client` and returns each server gradient to the matching client encoder.
- Evaluates on paired multimodal test data from `data_partition/test_multimodal.pt`.

Output under `result/`:

```text
result/
  train_log.csv
  eval_log.csv
  final_metrics.json
  best_metrics.json
  config_used.yaml
```

Output under `result_model/`:

```text
result_model/
  best_server_model.pt
  best_client_encoders/
  best_model_info.json
```

## Configuration

Main config: `configs/uci_har.yaml`.

Important fields:

- `partition.clients_per_modality`: number of training clients per true modality.
- `cluster.method`: `kmeans` or `isodata`.
- `cluster.known_k`: expected modality cluster count, default `2`.
- `training.clients_per_cluster_per_round`: scheduling parameter `r`.
- `training.global_rounds`: Stage 3 training rounds.
- `result.output_dir`, `result_model.output_dir`: output locations.

## Current Implementation

Implemented:

- UCI-HAR inertial signal loading with required-file validation.
- Stage 1 data partition artifacts.
- Stage 2 local encoder pretraining, fingerprint extraction, KMeans and simplified ISODATA clustering.
- Cluster metrics: clustering accuracy, NMI, ARI.
- Stage 3 balanced predicted-cluster scheduling.
- Ordered feature concat with explicit `feature_map` gradient return to clients.
- Paired multimodal test evaluation.
- Required result and model output files.

TODO:

- Improve representation learning beyond the simple local autoencoder objective.
- Add richer server fusion modules if needed.
- Add a separate future D2D latency/offloading module; no real D2D training logic exists now.
- Add focused smoke tests once a small fixture or mocked UCI-HAR subset is available.
