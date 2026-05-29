# semantic_split_multimodal

`semantic_split_multimodal` is a **single-process simulation of distributed split multimodal learning**.

## Important Protocol Notes
- This project is **NOT Federated Learning**.
- FedAvg is **not implemented**.
- No client parameter aggregation is used.
- Clients only update local encoders through returned activation gradients.
- Server performs projection/alignment/fusion/classification and backward.

## Training Protocol
- **Global Round**: one client scheduling cycle.
- **Local Step**: one batch-level split-learning update.
- Current v1 main path:
  - client encoder
  - feature upload
  - shared semantic projector
  - supervised contrastive semantic alignment
  - concat fusion
  - classifier
  - server backward
  - gradient return
  - client backward update

## Data Pairing Rules
- Training: label-guided virtual multimodal pairing is used.
- Training must **not** use sample_id for cross-modality pairing.
- Testing: paired multimodal test set is used directly.
- Testing must **not** use label for pairing or filtering.

## Implemented in v1
- KMeans clustering with known K
- Fair random full-modality scheduling
- Label-guided virtual multimodal pairing
- Shared semantic projection
- Supervised contrastive semantic alignment
- Concat fusion
- Paired multimodal test evaluation

## Stubs Only (not implemented in this phase)
- ISODATA clustering
- D2D offloading
- PairedFullModalityScheduler
- GlobalRandomScheduler
- AttentionFusion training logic

## Device Selection
Set `device` in yaml:
- `auto` (default): CUDA if available, else CPU
- `cuda` / `gpu`: force CUDA
- `cpu`: force CPU
- custom strings like `cuda:1`

## Run Commands (Smoke)
1. Synthetic main training:
```bash
python experiments/run_stage2_training.py --config configs/default.yaml --experiment_name s2_default
```

2. No-alignment ablation:
```bash
python experiments/run_stage2_training.py --config configs/default.yaml --lambda_align 0.0 --experiment_name s2_no_align
```

3. KMeans cluster-map training:
```bash
python experiments/run_stage2_training.py --config configs/default.yaml --use_oracle_clusters_for_training false --experiment_name s2_kmeans_map
```

4. UCI-HAR:
```bash
python experiments/run_stage2_training.py --config configs/uci_har.yaml --experiment_name s2_uci_har
```

## Result Logs
- Round-level log:
  - `experiments/results/<experiment_name>_train_log.json`
- Final summary:
  - `experiments/results/<experiment_name>_summary.json`
