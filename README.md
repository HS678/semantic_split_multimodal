# semantic_split_multimodal

`semantic_split_multimodal` is a **distributed split multimodal learning** simulation on a single machine.

## Important
- This project is **NOT Federated Learning**.
- No FedAvg, no client parameter aggregation.
- Clients only own and update local encoders.
- Server owns semantic projectors, alignment/fusion/classifier and runs loss/backward.

## Run
```bash
python experiments/run_stage2_training.py --config configs/default.yaml
```

## Dataset Sources
- `dataset.type: synthetic` uses built-in synthetic paired multimodal data.
- `dataset.type: real` loads paired multimodal `.npz` files from `dataset.root`.

### Real dataset `.npz` format
Each `.npz` file must contain:
- `labels` (or `y`): shape `[N]`, integer labels
- `mod_0`, `mod_1`, ..., `mod_{M-1}`: each shape `[N, input_dim]`

By default:
- `train_file: train_paired.npz`
- `test_file: test_paired.npz`

Optional:
- Set `full_file` to one `.npz` and it will be split by `train_split_ratio`.

## Implemented in v1
- Synthetic paired multimodal dataset
- Controlled modality-to-client partition
- Controlled label-skew partition
- Avoid single-label clients
- Class-balanced batch sampling
- KMeans clustering (known K = num_modalities)
- FairRandomFullModalityScheduler
- Label-guided virtual semantic batch (label-only pairing)
- Semantic projectors + supervised contrastive alignment
- Concat fusion + classifier
- Server-side loss/backward + gradient routing
- Client-side backward/update
- Paired multimodal test evaluation

## Stubs in v1
- ISODATA clustering
- PairedFullModalityScheduler
- GlobalRandomScheduler
- AttentionFusion
- PrototypeMemory/PrototypeLoss (lambda_proto default 0)
- D2D offloading
