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
local/datasets/cmu_mosei/
local/datasets/IEMOCAP/IEMOCAP_full/IEMOCAP_full_release/
```

CMU-MOSEI requires:

```text
features/BERT_MOSEI.pkl
features/COAVAREP_aligned_MOSEI.pkl
features/FACET_aligned_MOSEI.pkl
splits/df_MOSEI.tsv
splits/df_valid_MOSEI.tsv
splits/df_test_MOSEI.tsv
```

The split TSV files come from the feature source repository [Ighina/MultiModalSA](https://github.com/Ighina/MultiModalSA/tree/master/data). Their original row order must be preserved for exact BERT feature/label verification.

IEMOCAP uses the Full release and the four-class `angry / happy-or-excited / sad / neutral` protocol. Prepare its frozen MFCC, MobileViT-XS, and DistilBERT sequence features before Stage 1:

```bash
python scripts/prepare_iemocap.py --device cuda
```

The fixed split is Session 1-3 train, Session 4 validation, and Session 5 test. Audio uses three 1D convolution blocks followed by a GRU; video and text use GRUs over the frozen MobileViT-XS frame embeddings and DistilBERT token embeddings.

Local references may stay under `local/references/`. The whole `local/` tree is ignored by Git.

## Result Layout

Stage 1 partitions are reusable assets:

```text
local/results/partition/<dataset>/<partition_signature>/
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
UCI-HAR: local/results/partition/uci_har/acc_10clients_gyro_10clients__subject_disjoint_tvt_v1
MHEALTH: local/results/partition/mhealth/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients_ecg_10clients__subject_disjoint_tvt_v1
PAMAP2:  local/results/partition/pamap2/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients__subject_disjoint_tvt_v1
CMU-MOSEI: local/results/partition/cmu_mosei/text_10clients_audio_10clients_visual_10clients__official_video_disjoint_tvt_v1
IEMOCAP: local/results/partition/iemocap/audio_10clients_video_10clients_text_10clients__session_disjoint_123_4_5_v1
```

No stage overwrites an existing non-empty output directory. Use a three-split Stage 3 `run_id`, such as `adaptive_tvt_seed101`.

## Stage 1

Run one dataset:

```bash
python scripts/stage1_partition.py --config configs/uci_har.yaml
```

Run all five datasets one by one:

```bash
python scripts/stage1_partition.py --config configs/uci_har.yaml
python scripts/stage1_partition.py --config configs/mhealth.yaml
python scripts/stage1_partition.py --config configs/pamap2.yaml
python scripts/stage1_partition.py --config configs/cmu_mosei.yaml
python scripts/stage1_partition.py --config configs/iemocap.yaml
```

Stage 1 writes directly under the partition directory:

```text
train_clients/client_*.pt
client_meta.csv
validation_multimodal.pt
test_multimodal.pt
partition_config.json
```

UCI-HAR, MHEALTH, and PAMAP2 use fixed, disjoint subject-level train/validation/test splits. CMU-MOSEI uses the source repository's official video-disjoint train/validation/test splits with 16,327/1,871/4,662 samples. Its task is binary `polarity < 0` versus `polarity >= 0`; audio/visual sequences are mean-pooled, and all three modalities are standardized from train statistics only. Only train enters client partitioning and Stage 2. Validation/test remain naturally paired.

IEMOCAP uses the fixed disjoint Session 1-3/4/5 split with 5,531 four-class utterances. Its padded sequence lengths are propagated through Stage 1, encoder pretraining, fingerprint extraction, Split Learning, and naturally paired evaluation. `configs/iemocap.yaml` intentionally selects `true_cluster` for the requested oracle/debug comparison; it is not a no-leakage main result.

## Stage 2

UCI-HAR:

```bash
python scripts/stage2_discovery.py \
  --config configs/uci_har.yaml \
  --stage1-dir local/results/partition/uci_har/acc_10clients_gyro_10clients__subject_disjoint_tvt_v1 \
  --output-root local/results/cluster
```

MHEALTH:

```bash
python scripts/stage2_discovery.py \
  --config configs/mhealth.yaml \
  --stage1-dir local/results/partition/mhealth/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients_ecg_10clients__subject_disjoint_tvt_v1 \
  --output-root local/results/cluster
```

PAMAP2:

```bash
python scripts/stage2_discovery.py \
  --config configs/pamap2.yaml \
  --stage1-dir local/results/partition/pamap2/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients__subject_disjoint_tvt_v1 \
  --output-root local/results/cluster
```

CMU-MOSEI:

```bash
python scripts/stage2_discovery.py \
  --config configs/cmu_mosei.yaml \
  --stage1-dir local/results/partition/cmu_mosei/text_10clients_audio_10clients_visual_10clients__official_video_disjoint_tvt_v1 \
  --output-root local/results/cluster
```

IEMOCAP:

```bash
python scripts/stage2_discovery.py \
  --config configs/iemocap.yaml \
  --stage1-dir local/results/partition/iemocap/audio_10clients_video_10clients_text_10clients__session_disjoint_123_4_5_v1 \
  --output-root local/results/cluster
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

By default, the technical Stage 3 inputs are a complete `pred_cluster.csv` and one pretrained encoder per client. In the default mode, `true_cluster.csv` and `stage2_metadata.json` are optional audit inputs; missing audit files, inconsistent true clusters, or a non-success `discovery_status` do not gate Stage 3 startup.

## Stage 3

Run Stage 3 from a frozen Stage 1 partition and a frozen Stage 2 cluster directory. Formal YAML files keep the base `seed` at `42` for Stage 1/Stage 2 and default Stage 3 behavior. `--seed` overrides only the in-memory Stage 3 experiment seed; it does not modify YAML or affect the frozen Stage 1/Stage 2 artifacts. Start with UCI-HAR, then run the larger datasets.

Select the Stage 3 cluster assignment source with:

```yaml
training:
  cluster_assignment_source: pred_cluster  # formal default
```

For Stage 3 debugging, set it to `true_cluster` to bypass predicted clustering. Training, scheduling, binding, fusion slots, and the evaluation mapping will then consistently read `true_cluster.csv`. This is an oracle/debug mode that uses true modality clusters; do not report it as the no-leakage main result, and use a clearly distinguishable `run-id`.

UCI-HAR:

```bash
python scripts/stage3_train.py \
  --config configs/uci_har.yaml \
  --stage1-dir local/results/partition/uci_har/acc_10clients_gyro_10clients__subject_disjoint_tvt_v1 \
  --stage2-dir local/results/cluster/uci_har/acc_10clients_gyro_10clients__subject_disjoint_tvt_v1/adaptive_isodata \
  --output-root local/results/experiments \
  --run-id adaptive_tvt_seed101 \
  --seed 101
```

MHEALTH:

```bash
python scripts/stage3_train.py \
  --config configs/mhealth.yaml \
  --stage1-dir local/results/partition/mhealth/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients_ecg_10clients__subject_disjoint_tvt_v1 \
  --stage2-dir local/results/cluster/mhealth/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients_ecg_10clients__subject_disjoint_tvt_v1/adaptive_isodata \
  --output-root local/results/experiments \
  --run-id adaptive_tvt_seed101 \
  --seed 101
```

PAMAP2:

```bash
python scripts/stage3_train.py \
  --config configs/pamap2.yaml \
  --stage1-dir local/results/partition/pamap2/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients__subject_disjoint_tvt_v1 \
  --stage2-dir local/results/cluster/pamap2/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients__subject_disjoint_tvt_v1/adaptive_isodata \
  --output-root local/results/experiments \
  --run-id adaptive_tvt_seed101 \
  --seed 101
```

CMU-MOSEI:

```bash
python scripts/stage3_train.py \
  --config configs/cmu_mosei.yaml \
  --stage1-dir local/results/partition/cmu_mosei/text_10clients_audio_10clients_visual_10clients__official_video_disjoint_tvt_v1 \
  --stage2-dir local/results/cluster/cmu_mosei/text_10clients_audio_10clients_visual_10clients__official_video_disjoint_tvt_v1/adaptive_isodata \
  --output-root local/results/experiments \
  --run-id adaptive_tvt_seed101 \
  --seed 101
```

IEMOCAP true-cluster Oracle/debug comparison:

```bash
python scripts/stage3_train.py \
  --config configs/iemocap.yaml \
  --stage1-dir local/results/partition/iemocap/audio_10clients_video_10clients_text_10clients__session_disjoint_123_4_5_v1 \
  --stage2-dir local/results/cluster/iemocap/audio_10clients_video_10clients_text_10clients__session_disjoint_123_4_5_v1/adaptive_isodata \
  --output-root local/results/experiments \
  --run-id true_cluster_weighted_seed101 \
  --seed 101
```

Stage 3 writes directly under `local/results/experiments/<dataset>/<run_id>/`:

```text
resolved_config.yaml
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
  -m semantic_split_multimodal.evaluation.plot_training_curves \
  --run-dir local/results/experiments/<dataset>/<run_id>
```

The five formal Stage 3 seeds are `101`, `202`, `303`, `404`, and `505`. Give every run a distinct `run_id`, for example:

```bash
python scripts/stage3_train.py \
  --config configs/uci_har.yaml \
  --stage1-dir local/results/partition/uci_har/acc_10clients_gyro_10clients__subject_disjoint_tvt_v1 \
  --stage2-dir local/results/cluster/uci_har/acc_10clients_gyro_10clients__subject_disjoint_tvt_v1/adaptive_isodata \
  --output-root local/results/experiments \
  --run-id adaptive_tvt_seed202 \
  --seed 202
```

The four pre-existing datasets have independent launchers that run Stage 1, Stage 2, and all five Stage 3 seeds:

```bash
nohup bash local/tools/launch_uci_har_formal.sh \
  > "local/tools/uci_har_formal_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &

nohup bash local/tools/launch_mhealth_formal.sh \
  > "local/tools/mhealth_formal_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &

nohup bash local/tools/launch_pamap2_formal.sh \
  > "local/tools/pamap2_formal_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &

nohup bash local/tools/launch_cmu_mosei_formal.sh \
  > "local/tools/cmu_mosei_formal_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &
```

`launch_stage3_formal.sh` is retained as the aggregate launcher that sequentially starts formal experiments for those four datasets:

```bash
nohup bash local/tools/launch_stage3_formal.sh \
  > "local/tools/formal_all_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &
```

It uses `adaptive_tvt_seed<N>` and therefore does not overwrite old train/test runs. The entire `local/` tree, including this launcher and its logs, is ignored by Git.

## Testing

```bash
PYTHONPATH=src python -m pytest tests -q
```
