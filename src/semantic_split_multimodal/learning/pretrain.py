import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from semantic_split_multimodal.discovery.clustering import run_hdbscan, run_isodata, run_kmeans
from semantic_split_multimodal.discovery.fingerprint import build_fingerprints
from semantic_split_multimodal.data.client import Client
from semantic_split_multimodal.data.partitioner import resolve_project_path
from semantic_split_multimodal.evaluation.metrics import discovery_metrics
from semantic_split_multimodal.learning.models import create_client_encoder, flattened_dim


def _load_clients(partition_dir: Path):
    train_dir = partition_dir / "train_clients"
    paths = sorted(train_dir.glob("client_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No client_*.pt files found in {train_dir}. Run Stage 1 first.")
    return [Client.from_payload(torch.load(path, map_location="cpu")) for path in paths]


class _AutoEncoder(nn.Module):
    def __init__(self, client: Client, cfg: dict):
        super().__init__()
        self.encoder = create_client_encoder(cfg, input_shape=client.input_shape, encoder_type=client.encoder_type)
        self.decoder = nn.Linear(int(cfg.get("encoder_hidden_dim", 128)), flattened_dim(client.input_shape))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def _pretrain_client_encoder(client: Client, cfg: dict, device):
    pre_cfg = cfg.get("pretrain", {})
    model = _AutoEncoder(client, cfg).to(device)
    epochs = int(pre_cfg.get("epochs", 5))
    batch_size = int(pre_cfg.get("batch_size", cfg.get("batch_size", 64)))
    lr = float(pre_cfg.get("lr", cfg.get("learning_rate", 1e-3)))
    max_samples = pre_cfg.get("max_samples")
    x = client.samples
    if max_samples is not None:
        x = x[: min(int(max_samples), int(x.shape[0]))]
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=float(pre_cfg.get("weight_decay", 0.0)))
    loss_fn = nn.MSELoss()
    total = 0.0
    steps = 0
    model.train()
    for _ in range(max(0, epochs)):
        for (xb,) in loader:
            xb = xb.to(device)
            recon, _ = model(xb)
            loss = loss_fn(recon, xb.reshape(xb.shape[0], -1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
    return model.encoder, (total / steps if steps else None)


def run_stage2_discovery(cfg: dict, project_root: Path, device: torch.device):
    partition_dir = resolve_project_path(project_root, cfg.get("partition", {}).get("output_dir", "local/results/data_partition"))
    cluster_dir = resolve_project_path(project_root, cfg.get("cluster", {}).get("output_dir", "local/results/cluster"))
    result_dir = resolve_project_path(project_root, cfg.get("result", {}).get("output_dir", "local/logs"))
    encoder_dir = cluster_dir / "pretrained_encoders"
    encoder_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    clients = _load_clients(partition_dir)
    encoders = {}
    losses = {}
    for client in clients:
        encoder, avg_loss = _pretrain_client_encoder(client, cfg, device)
        encoders[client.client_id] = encoder
        losses[client.client_id] = avg_loss
        torch.save(
            {
                "client_id": client.client_id,
                "encoder_type": client.encoder_type,
                "input_shape": [int(v) for v in client.input_shape],
                "hidden_dim": int(cfg.get("encoder_hidden_dim", 128)),
                "state_dict": encoder.cpu().state_dict(),
            },
            encoder_dir / f"{client.client_id}_encoder.pt",
        )
        encoder.to(device)

    fingerprints = build_fingerprints(clients, encoders, cfg, device)
    np.save(cluster_dir / "fingerprints.npy", fingerprints)

    cluster_cfg = cfg.get("cluster", {})
    method = str(cluster_cfg.get("method", "isodata")).lower()
    raw_k = cluster_cfg.get("known_k")
    known_k = None if raw_k is None or str(raw_k).lower() in {"auto", "none", "null"} else int(raw_k)
    seed = int(cfg.get("seed", 42))
    if method == "kmeans":
        pred = run_kmeans(fingerprints, known_k, seed=seed)
    elif method == "isodata":
        pred = run_isodata(fingerprints, known_k, seed=seed, **dict(cluster_cfg.get("isodata", {})))
    elif method == "hdbscan":
        pred = run_hdbscan(fingerprints, seed=seed, **dict(cluster_cfg.get("hdbscan", {})))
    else:
        raise ValueError("cluster.method must be 'kmeans', 'hdbscan', or 'isodata'.")

    true = np.array([client.hidden_modality_id for client in clients], dtype=int)
    pred = np.asarray(pred, dtype=int)
    metrics = discovery_metrics(true, pred)
    metrics.update(
        {
            "method": method,
            "known_k": known_k,
            "true_num_modalities": int(len(np.unique(true))),
            "pretrain_reconstruction_loss": losses,
            "fingerprint_type": str(cfg.get("fingerprint", {}).get("type", "hybrid")),
            "uses_input_dimension_hint": False,
        }
    )

    rows = []
    for client, pred_cluster in zip(clients, pred):
        rows.append(
            {
                "client_id": client.client_id,
                "hidden_modality_id": int(client.hidden_modality_id),
                "pred_cluster": int(pred_cluster),
            }
        )
    with (cluster_dir / "cluster_assignments.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["client_id", "hidden_modality_id", "pred_cluster"])
        writer.writeheader()
        writer.writerows(rows)
    for path in [cluster_dir / "cluster_metrics.json", result_dir / "cluster_metrics.json"]:
        with path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    with (cluster_dir / "cluster_config.json").open("w", encoding="utf-8") as f:
        json.dump({"cluster": cluster_cfg, "pretrain": cfg.get("pretrain", {}), "fingerprint": cfg.get("fingerprint", {})}, f, indent=2)
    with (result_dir / "cluster_result.txt").open("w", encoding="utf-8") as f:
        f.write(f"method: {method}\n")
        f.write(f"known_k: {known_k}\n")
        f.write(f"true_Q: {metrics['true_Q']}\n")
        f.write(f"estimated_Q: {metrics['estimated_Q']}\n")
        f.write(f"abs_Q_error: {metrics['abs_Q_error']}\n")
        f.write(f"estimated_num_clusters: {metrics['estimated_num_clusters']}\n")
        f.write(f"ACC: {metrics['ACC']:.6f}\n")
        f.write(f"NMI: {metrics['NMI']:.6f}\n")
        f.write(f"ARI: {metrics['ARI']:.6f}\n")
    return metrics
