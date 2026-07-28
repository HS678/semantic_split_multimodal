# semantic_split_multimodal

Semantic-aligned distributed Split Multimodal Learning in unknown modality environments.

This repository contains a clean three-stage experiment framework for distributed multimodal Split Learning. The active method line is `mmbind_fusion_split_learning`: naturally paired multimodal training data is first partitioned into single-modality clients, client modalities are discovered without using modality identities, and Stage 3 trains an MMBind-style fusion Split Learning model from the predicted clusters.

The project is not Federated Learning and does not use FedAvg. Clients send detached activations to the server; the server computes cross-entropy loss, backpropagates through the fusion model, and routes activation gradients back to the originating client encoders.

## Research Question

The framework studies whether multimodal Split Learning can recover and use semantic modality structure when clients arrive as single-modality holders and their true modality identities are unknown during training.

## Method Line

The current public branch keeps one active method:

1. Stage 1: single-modality client partition
2. Stage 2: adaptive ISODATA modality discovery
3. Stage 3: MMBind-style fusion Split Learning

The Stage 2 output is `pred_cluster`. Stage 3 scheduling, binding, ClusterAdapter slots, concat fusion, classifier training, and Split Learning backward all use `pred_cluster`, not true modality identity.

## Unknown-Modality Constraint

Training does not use true modality names, true modality IDs, true Q, or an oracle modality scheduler. The `hidden_modality_id` field is saved by Stage 1 only for post-hoc discovery audit and evaluation-only oracle mapping. It must not feed PCA, split/merge decisions, Q selection, seed selection, scheduling, binding, fusion slots, model input construction, or training loss.

Naturally paired final evaluation reads `01_dataset_partition/test_multimodal.pt`. The evaluation-only oracle mapping is used only to map test modalities to discovered cluster slots after discovery and training have already fixed their decisions. Test labels are used only to compute metrics.

## Installation

Create and activate a Python environment, then install dependencies and the package:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

`hdbscan` is optional and only needed when using the HDBSCAN clustering mode. The default unknown-Q configs use adaptive ISODATA.

## Data Layout

Datasets are not included in this repository. Place raw datasets under:

```text
local/datasets/
```

Reference materials are local-only and should stay under:

```text
local/references/
```

The entire `local/` tree is ignored by Git. Do not commit datasets, references, checkpoints, logs, fingerprints, generated configs, or formal experiment results.

Expected dataset roots:

```text
local/datasets/uci_har/
local/datasets/mhealth/
local/datasets/pamap2/
```

## Configs

The public configs are:

```text
configs/uci_har.yaml
configs/mhealth.yaml
configs/pamap2.yaml
```

Each config defines dataset loading, Stage 1 partitioning, Stage 2 adaptive discovery, Stage 3 training, binding, fusion, and evaluation settings. The true `num_modalities` value is kept as dataset metadata and for audit sanity checks; it is not used as the unknown-Q discovery answer during Stage 3 training.

## Stage 1

Run one dataset:

```bash
python scripts/stage1_partition.py --config configs/uci_har.yaml
```

Or use the convenience runner:

```bash
scripts/run_stage1_partitions.sh uci_har
scripts/run_stage1_partitions.sh all
```

Stage 1 creates a new timestamped run directory under:

```text
local/results/<dataset>/<run_id>/01_dataset_partition/
```

Important outputs:

```text
train_clients/client_*.pt
client_meta.csv
test_multimodal.pt
partition_config.json
```

## Stage 2

For formal unknown-Q discovery, use the Stage2-only entry so Stage 1 input and Stage 2 output stay separated:

```bash
python scripts/stage2_discovery_only.py \
  --config configs/uci_har.yaml \
  --stage1-dir local/results/uci_har/<stage1_run_id>/01_dataset_partition \
  --output-root local/results/stage2_frozen \
  --tag uci_har_stage2_adaptive_<sha> \
  --run-type user_formal
```

Stage 2 writes:

```text
local/results/stage2_frozen/<dataset>/<tag>/02_cluster_results/
local/results/stage2_frozen/<dataset>/<tag>/02_discovery_logs/
```

Important outputs:

```text
02_cluster_results/pretrained_encoders/*_encoder.pt
02_cluster_results/fingerprints.npy
02_cluster_results/cluster_assignments.csv
02_cluster_results/cluster_metrics.json
02_cluster_results/adaptive_diagnostics.json
```

The legacy `scripts/stage2_discovery.py` entry remains as a compatibility wrapper for configs that use a single shared run directory.

## Stage 3

For formal training from frozen Stage 1 and Stage 2 inputs, use the Stage3-only entry:

```bash
python scripts/stage3_train_only.py \
  --config configs/uci_har.yaml \
  --stage1-dir local/results/uci_har/<stage1_run_id>/01_dataset_partition \
  --stage2-dir local/results/stage2_frozen/uci_har/<stage2_run_id>/02_cluster_results \
  --output-root local/results/stage3_formal \
  --tag uci_har_stage3_adaptive_<sha> \
  --run-type user_formal
```

Stage 3 writes:

```text
local/results/stage3_formal/<dataset>/<tag>/03_training_evaluation/
local/results/stage3_formal/<dataset>/<tag>/04_model_artifacts/
```

Important outputs:

```text
03_training_evaluation/train_log.csv
03_training_evaluation/eval_log.csv
03_training_evaluation/final_metrics.json
04_model_artifacts/best_mmbind_fusion_checkpoint.pt
04_model_artifacts/last_mmbind_fusion_checkpoint.pt
04_model_artifacts/cluster_to_slot.json
```

The legacy `scripts/stage3_train.py` entry remains as a compatibility wrapper for configs that use a single shared run directory.

## Output Policy

Generated outputs are local artifacts and are not part of the public repository:

```text
local/results/
local/checkpoints/
local/logs/
```

The Stage2-only and Stage3-only entries refuse to silently overwrite existing run directories. Use a new `--tag` for each formal run.

## Tests

Run syntax checks:

```bash
python -m compileall src scripts tests
```

Run the full test suite:

```bash
PYTHONPATH=src python -m pytest tests -q
```

The tests cover dataset adapters, Stage 1 partitioning, adaptive discovery, discovery audit metrics, Stage2-only output isolation, scheduler behavior, label-guided pseudo binding, ClusterAdapter and concat fusion, Split Learning training, naturally paired evaluation, Stage3-only output isolation, and hidden-modality leakage guards.

## Project Structure

```text
configs/    # UCI-HAR, MHEALTH, and PAMAP2 configs
docs/       # protocol, architecture, output, and handoff notes
scripts/    # Stage 1, Stage 2-only, Stage 3-only, and compatibility CLIs
src/        # semantic_split_multimodal package
tests/      # unit and regression tests for the active method line
local/      # ignored local datasets, references, outputs, and checkpoints
```

This repository does not publish datasets, checkpoints, formal results, or unreleased ablation claims.
