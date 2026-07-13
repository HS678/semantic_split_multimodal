# Extension Guide

This guide explains how to extend the non-D2D foundation to new multimodal datasets and how to keep future D2D integration clean.

## Current Scope

The implemented framework is:

1. Stage 1: dataset loading and single-modality client partitioning.
2. Stage 2: local encoder pretraining, fingerprint extraction, and modality-aware clustering.
3. Stage 3: balanced split multimodal learning based on predicted clusters.

D2D training/offloading is intentionally not implemented in this phase. Keep D2D as a future module that consumes trained client/server metadata and latency inputs, rather than mixing D2D logic into Stage 1/2/3.

## Result Layout

Every Stage 1 run creates a new timestamped run folder:

```text
results/<dataset_name>/<yy_mm_dd_HH_MM_SS_mmm>/
  run_meta.yaml
  01_dataset_partition/
  02_cluster_results/
  03_training_evaluation/
  04_model_artifacts/
```

Stage 2 and Stage 3 reuse `results/<dataset_name>/latest_run.txt` by default. This keeps all artifacts for one experiment run together while preserving the three independent stage entry scripts.

## Built-In Real Datasets

Current real dataset adapters:

- `uci_har`: two-modality inertial HAR sanity check.
- `mhealth`: four-modality wearable health/activity recognition dataset.
- `pamap2`: four-modality physical activity monitoring dataset.

MHEALTH active modality grouping uses sensor types:

- `accelerometer`: acceleration from chest, left ankle, and right lower arm.
- `gyroscope`: gyroscope from left ankle and right lower arm.
- `magnetometer`: magnetometer from left ankle and right lower arm.
- `ecg`: two ECG leads from chest.

The adapter still supports the older body-position grouping through `dataset.modality_scheme: position`, but the active config uses `sensor_type`.

Run MHEALTH with:

```bash
python experiments/stage1_partition.py --config configs/mhealth.yaml
python experiments/stage2_pretrain_cluster.py --config configs/mhealth.yaml
python experiments/stage3_train_sl.py --config configs/mhealth.yaml
```

PAMAP2 active modality grouping:

- `accelerometer`: 16g acceleration from hand, chest, and ankle IMUs.
- `gyroscope`: gyroscope from hand, chest, and ankle IMUs.
- `magnetometer`: magnetometer from hand, chest, and ankle IMUs.

The active PAMAP2 config disables heart rate with `dataset.include_heart_rate: false`.

Run PAMAP2 with:

```bash
python experiments/stage1_partition.py --config configs/pamap2.yaml
python experiments/stage2_pretrain_cluster.py --config configs/pamap2.yaml
python experiments/stage3_train_sl.py --config configs/pamap2.yaml
```

## Adding A 3+ Modality Dataset

Add one dataset adapter under `data/`, then register it in `data/dataset_registry.py`.

The loader must return this contract:

```python
{
    "train": {
        "modalities": [x_mod0, x_mod1, x_mod2],
        "labels": y_train,
    },
    "test": {
        "modalities": [x_mod0_test, x_mod1_test, x_mod2_test],
        "labels": y_test,
    },
    "root": str(resolved_root),
    "modality_names": ["mod0", "mod1", "mod2"],
    "modality_input_dims": [dim0, dim1, dim2],
    "modality_input_shapes": [shape0, shape1, shape2],
}
```

Rules:

- Every modality tensor must have the same first dimension as the label tensor.
- `test.modalities` must be paired multimodal data: sample `i` across all modalities must share the same label.
- `modality_names` are saved for clustering evaluation only. Stage 3 scheduling must continue to use `pred_cluster`.
- For 4, 5, or 6 modality datasets, extend the lists above; do not change the stage entry scripts.

## Config For 3 To 6 Modalities

Create a dataset-specific YAML file, for example `configs/my_dataset.yaml`.

Important fields:

```yaml
dataset:
  type: my_dataset
  root: ./data/my-dataset

num_modalities: 4
num_classes: 10

partition:
  output_dir: ./results/data_partition  # overridden by the run-local result manager
  clients_per_modality: 10

cluster:
  output_dir: ./results/cluster  # overridden by the run-local result manager
  method: isodata
  known_k: null
  use_input_dim_hint: false
  isodata:
    initial_k: 3
    min_clusters: 3
    max_clusters: 6
    max_iter: 30
    min_cluster_size: 2
    split_std_threshold: 1.0
    merge_distance_threshold: 0.25

training:
  clients_per_cluster_per_round: 4
```

`Q_star` is determined by the number of predicted clusters. Stage 3 then uses `K_t = Q_star * r`, where `r = training.clients_per_cluster_per_round`.

## ISODATA Settings

Use these experiment variants:

- Main unknown-modality setting: `cluster.method: isodata`, `known_k: null`, and `use_input_dim_hint: false`.
- Structural-hint or upper-bound setting: `use_input_dim_hint: true` when modalities have clearly different input shapes.
- Fixed-k ablation: set `known_k` to the expected modality count.

For UCI-HAR, the current config keeps `min_clusters=max_clusters=2` because UCI-HAR is only a two-modality sanity check. For paper datasets with 3+ modalities, set a wider range such as `[3, 6]`.

## Adding Encoders

Client encoders are created only through `models/encoders.py`.

To add a new encoder:

1. Implement a new `nn.Module`.
2. Add a branch in `create_client_encoder()`.
3. Select it in YAML:

```yaml
model:
  encoder:
    type: my_encoder
```

If different modalities need different encoders, use:

```yaml
model:
  encoder:
    type: mlp
    by_modality:
      image: cnn2d
      inertial: cnn_gru
      text: transformer_stub_or_mlp
```

Do not add dataset-specific branches in Stage 2 or Stage 3. The encoder factory should absorb those differences.

## Stage 3 Contract

The split learning trainer must preserve these invariants:

- Build `cluster_to_clients` from `pred_cluster`.
- Never use true `modality_name` for training scheduling.
- Return `selected[cluster_id][group_id] = client`.
- Concatenate features in sorted `cluster_id` order.
- Preserve `feature_map[(cluster_id, group_id)] -> client`.
- After server backward, return `z_server.grad` to the exact corresponding client encoder.
- Evaluate only on paired multimodal test data from the run-local `01_dataset_partition/test_multimodal.pt`.

These invariants are what make 3, 4, 5, and 6 modality datasets work without changing the scheduling logic.

## Future D2D Integration

Do not place D2D logic inside the current Stage 3 trainer.

Recommended future integration:

- Add a new module such as `d2d/` only when real D2D experiments begin.
- Let D2D consume client metadata, encoder partition points, feature sizes, and latency/bandwidth profiles.
- Keep the current config gate:

```yaml
d2d:
  enabled: false
```

When D2D is implemented, add a separate experiment entry or trainer wrapper so the non-D2D baseline remains reproducible.
