import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from clustering.cluster import evaluate_clustering, run_isodata, run_kmeans
from data.partitioner import resolve_project_path
from models.modules import ClientEncoder


def _load_clients(partition_dir: Path):
    train_dir = partition_dir / "train_clients"
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train client directory: {train_dir}. Run Stage 1 first.")
    paths = sorted(train_dir.glob("client_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No client_*.pt files found in {train_dir}. Run Stage 1 first.")
    return [torch.load(path, map_location="cpu") for path in paths]


class _AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = ClientEncoder(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def _pretrain_client_encoder(client, cfg, device):
    pre_cfg = cfg.get("pretrain", {})
    input_dim = int(client["input_dim"])
    hidden_dim = int(cfg.get("encoder_hidden_dim", 128))
    model = _AutoEncoder(input_dim, hidden_dim).to(device)

    epochs = int(pre_cfg.get("epochs", 5))
    batch_size = int(pre_cfg.get("batch_size", cfg.get("batch_size", 64)))
    lr = float(pre_cfg.get("lr", cfg.get("learning_rate", 1e-3)))
    weight_decay = float(pre_cfg.get("weight_decay", 0.0))
    max_samples = pre_cfg.get("max_samples")

    x = client["x"]
    if max_samples is not None:
        x = x[: min(int(max_samples), int(x.shape[0]))]
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    total = 0.0
    steps = 0
    model.train()
    for _ in range(max(0, epochs)):
        for (xb,) in loader:
            xb = xb.to(device)
            recon, _ = model(xb)
            loss = loss_fn(recon, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
    return model.encoder, (total / steps if steps else None)


def _fingerprint(encoder, x, cfg, device):
    fp_cfg = cfg.get("fingerprint", {})
    batch_size = int(fp_cfg.get("batch_size", cfg.get("batch_size", 64)))
    max_batches = fp_cfg.get("max_batches", 4)
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
    zs = []
    encoder.eval()
    with torch.no_grad():
        for i, (xb,) in enumerate(loader):
            if max_batches is not None and i >= int(max_batches):
                break
            zs.append(encoder(xb.to(device)).detach().cpu())
    if not zs:
        raise RuntimeError("Cannot extract fingerprint from an empty client dataset.")
    return torch.cat(zs, dim=0).mean(dim=0).numpy()


def run_stage2_pretrain_cluster(cfg: dict, project_root: Path, device: torch.device):
    partition_dir = resolve_project_path(project_root, cfg.get("partition", {}).get("output_dir", "data_partition"))
    cluster_dir = resolve_project_path(project_root, cfg.get("cluster", {}).get("output_dir", "cluster"))
    result_dir = resolve_project_path(project_root, cfg.get("result", {}).get("output_dir", "result"))
    encoder_dir = cluster_dir / "pretrained_encoders"
    encoder_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    clients = _load_clients(partition_dir)
    fingerprints = []
    losses = {}
    for client in clients:
        encoder, avg_loss = _pretrain_client_encoder(client, cfg, device)
        fingerprints.append(_fingerprint(encoder, client["x"], cfg, device))
        losses[client["client_id"]] = avg_loss
        torch.save(
            {
                "client_id": client["client_id"],
                "modality_id": int(client["modality_id"]),
                "modality_name": client["modality_name"],
                "input_dim": int(client["input_dim"]),
                "hidden_dim": int(cfg.get("encoder_hidden_dim", 128)),
                "state_dict": encoder.cpu().state_dict(),
            },
            encoder_dir / f"{client['client_id']}_encoder.pt",
        )
        encoder.to(device)

    fingerprints_np = np.stack(fingerprints)
    np.save(cluster_dir / "fingerprints.npy", fingerprints_np)

    cluster_cfg = cfg.get("cluster", {})
    method = str(cluster_cfg.get("method", "kmeans")).lower()
    known_k = int(cluster_cfg.get("known_k", cfg.get("num_modalities", 2)))
    seed = int(cfg.get("seed", 42))
    if method == "kmeans":
        pred = run_kmeans(fingerprints, known_k, seed=seed)
    elif method == "isodata":
        iso_kwargs = dict(cluster_cfg.get("isodata", {}))
        pred = run_isodata(fingerprints, known_k, seed=seed, **iso_kwargs)
    else:
        raise ValueError(f"Unsupported cluster.method: {method}. Expected 'kmeans' or 'isodata'.")

    true = np.array([int(c["modality_id"]) for c in clients])
    mapping, cm, acc, nmi, ari = evaluate_clustering(true, pred, known_k)
    metrics = {
        "clustering_accuracy": acc,
        "NMI": nmi,
        "ARI": ari,
        "confusion_matrix": cm.tolist(),
        "cluster_to_true_modality_majority": {str(k): int(v) for k, v in mapping.items()},
        "method": method,
        "known_k": known_k,
        "pretrain_reconstruction_loss": losses,
    }

    rows = []
    for client, pred_cluster in zip(clients, pred):
        rows.append(
            {
                "client_id": client["client_id"],
                "true_modality": client["modality_name"],
                "pred_cluster": int(pred_cluster),
            }
        )
    with (cluster_dir / "cluster_assignments.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["client_id", "true_modality", "pred_cluster"])
        writer.writeheader()
        writer.writerows(rows)

    with (cluster_dir / "cluster_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (cluster_dir / "cluster_config.json").open("w", encoding="utf-8") as f:
        json.dump({"cluster": cluster_cfg, "pretrain": cfg.get("pretrain", {}), "fingerprint": cfg.get("fingerprint", {})}, f, indent=2)

    with (result_dir / "cluster_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (result_dir / "cluster_result.txt").open("w", encoding="utf-8") as f:
        f.write(f"method: {method}\n")
        f.write(f"known_k: {known_k}\n")
        f.write(f"clustering_accuracy: {acc:.6f}\n")
        f.write(f"NMI: {nmi:.6f}\n")
        f.write(f"ARI: {ari:.6f}\n")
        for row in rows:
            f.write(f"{row['client_id']}, true={row['true_modality']}, pred_cluster={row['pred_cluster']}\n")

    return metrics
