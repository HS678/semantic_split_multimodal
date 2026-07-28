# semantic_split_multimodal

Semantic-aligned distributed Split Multimodal Learning in unknown modality environments.

中文说明见 [README.zh-CN.md](README.zh-CN.md).

## Overview

This repository provides a reproducible three-stage experiment framework for distributed multimodal Split Learning. The active method line is `mmbind_fusion_split_learning`:

1. Stage 1: partition naturally paired multimodal training data into single-modality clients.
2. Stage 2: discover unknown client modality clusters with adaptive ISODATA.
3. Stage 3: train an MMBind-style fusion Split Learning model from `pred_cluster`.

The project is not Federated Learning and does not use FedAvg. Clients upload detached activations. The server computes cross-entropy loss, backpropagates through the fusion model, and routes activation gradients back to the originating client encoders.

## Protocol

Training does not use true modality names, true modality IDs, true Q, or an oracle modality scheduler. `hidden_modality_id` is saved by Stage 1 only for post-hoc discovery audit and evaluation-only oracle mapping.

`hidden_modality_id` must not feed PCA, split/merge decisions, Q selection, seed selection, scheduling, binding, fusion slots, model inputs, or training loss.

Final evaluation reads `test_multimodal.pt` from the frozen Stage 1 partition directory. Test labels are used only to compute metrics.

## Installation

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

`hdbscan` is optional and only needed when using the HDBSCAN clustering mode. The default configs use adaptive ISODATA.

## Data Layout

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

Local references may stay under:

```text
local/references/
```

The entire `local/` tree is ignored by Git. Do not commit datasets, references, checkpoints, logs, fingerprints, generated configs, or experiment results.

## Public Configs

```text
configs/uci_har.yaml
configs/mhealth.yaml
configs/pamap2.yaml
```

Default partition names with `clients_per_modality: 10`:

```text
UCI-HAR: local/results/partition/uci_har/acc-gyro_10clients
MHEALTH: local/results/partition/mhealth/accelerometer-gyroscope-magnetometer-ecg_10clients
PAMAP2:  local/results/partition/pamap2/accelerometer-gyroscope-magnetometer_10clients
```

## Output Layout

Stage 1 creates reusable partition assets:

```text
local/results/partition/<dataset>/<modality_names>_<clients_per_modality>clients/
```

Stage 2, Stage 3, and future D2D runs share one experiment run directory:

```text
local/results/experiment/<dataset>/<run_id>/
  02_cluster_results/
  02_discovery_logs/
  03_training_evaluation/
  04_model_artifacts/
```

Use `run_1`, `run_2`, `run_3`, or another explicit tag for `run_id`.

Overwrite policy:

- Stage 1 refuses to overwrite an existing non-empty partition directory.
- Stage 2 refuses to overwrite existing Stage 2 outputs.
- Stage 3 may reuse the experiment directory created by Stage 2, but refuses to overwrite existing Stage 3 outputs.

## Stage 1: Build Reusable Partitions

Run one dataset:

```bash
python scripts/stage1_partition.py --config configs/uci_har.yaml
```

Or run the convenience wrapper:

```bash
scripts/run_stage1_partitions.sh uci_har
scripts/run_stage1_partitions.sh mhealth
scripts/run_stage1_partitions.sh pamap2
scripts/run_stage1_partitions.sh all
```

Stage 1 outputs:

```text
train_clients/client_*.pt
client_meta.csv
test_multimodal.pt
partition_config.json
```

## Stage 2: Unknown-Q Modality Discovery

UCI-HAR:

```bash
python scripts/stage2_discovery_only.py \
  --config configs/uci_har.yaml \
  --stage1-dir local/results/partition/uci_har/acc-gyro_10clients \
  --output-root local/results/experiment \
  --tag run_1 \
  --run-type user_formal
```

MHEALTH:

```bash
python scripts/stage2_discovery_only.py \
  --config configs/mhealth.yaml \
  --stage1-dir local/results/partition/mhealth/accelerometer-gyroscope-magnetometer-ecg_10clients \
  --output-root local/results/experiment \
  --tag run_1 \
  --run-type user_formal
```

PAMAP2:

```bash
python scripts/stage2_discovery_only.py \
  --config configs/pamap2.yaml \
  --stage1-dir local/results/partition/pamap2/accelerometer-gyroscope-magnetometer_10clients \
  --output-root local/results/experiment \
  --tag run_1 \
  --run-type user_formal
```

Stage 2 outputs:

```text
02_cluster_results/pretrained_encoders/*_encoder.pt
02_cluster_results/fingerprints.npy
02_cluster_results/cluster_assignments.csv
02_cluster_results/cluster_metrics.json
02_cluster_results/adaptive_diagnostics.json
02_discovery_logs/stage2_only_metadata.json
02_discovery_logs/stage2_only_config_used.yaml
```

Check `02_cluster_results/cluster_metrics.json` before running Stage 3. A successful discovery should report `discovery_status: discovery_success`.

## Stage 3: Fusion Split Learning

Run Stage 3 with the same `run_id` used by Stage 2.

UCI-HAR:

```bash
python scripts/stage3_train_only.py \
  --config configs/uci_har.yaml \
  --stage1-dir local/results/partition/uci_har/acc-gyro_10clients \
  --stage2-dir local/results/experiment/uci_har/run_1/02_cluster_results \
  --output-root local/results/experiment \
  --tag run_1 \
  --run-type user_formal
```

MHEALTH:

```bash
python scripts/stage3_train_only.py \
  --config configs/mhealth.yaml \
  --stage1-dir local/results/partition/mhealth/accelerometer-gyroscope-magnetometer-ecg_10clients \
  --stage2-dir local/results/experiment/mhealth/run_1/02_cluster_results \
  --output-root local/results/experiment \
  --tag run_1 \
  --run-type user_formal
```

PAMAP2:

```bash
python scripts/stage3_train_only.py \
  --config configs/pamap2.yaml \
  --stage1-dir local/results/partition/pamap2/accelerometer-gyroscope-magnetometer_10clients \
  --stage2-dir local/results/experiment/pamap2/run_1/02_cluster_results \
  --output-root local/results/experiment \
  --tag run_1 \
  --run-type user_formal
```

Stage 3 outputs:

```text
03_training_evaluation/train_log.csv
03_training_evaluation/eval_log.csv
03_training_evaluation/final_metrics.json
04_model_artifacts/best_mmbind_fusion_checkpoint.pt
04_model_artifacts/last_mmbind_fusion_checkpoint.pt
04_model_artifacts/cluster_to_slot.json
```

Use `03_training_evaluation/final_metrics.json` for naturally paired final evaluation metrics.

## Re-running Experiments

If the partition config is unchanged, reuse the existing Stage 1 partition and start a new experiment run:

```bash
python scripts/stage2_discovery_only.py \
  --config configs/uci_har.yaml \
  --stage1-dir local/results/partition/uci_har/acc-gyro_10clients \
  --output-root local/results/experiment \
  --tag run_2 \
  --run-type user_formal
```

Then run Stage 3 with `--tag run_2`.

If you change dataset preprocessing, modality scheme, client count, or another Stage 1 partition setting, rebuild Stage 1. If the old partition directory already exists, move or delete it manually after deciding it is no longer needed.

## Tests

Run syntax checks:

```bash
python -m compileall src scripts tests
```

Run the full test suite:

```bash
PYTHONPATH=src python -m pytest tests -q
```

## Project Structure

```text
configs/    # UCI-HAR, MHEALTH, and PAMAP2 configs
docs/       # protocol, architecture, output, and handoff notes
scripts/    # Stage 1, Stage 2-only, and Stage 3-only CLIs
src/        # semantic_split_multimodal package
tests/      # unit and regression tests for the active method line
local/      # ignored local datasets, references, outputs, and checkpoints
```

This repository does not publish datasets, checkpoints, formal results, or unreleased ablation claims.
