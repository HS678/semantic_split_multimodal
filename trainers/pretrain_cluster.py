import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from clustering.cluster import evaluate_clustering, run_isodata, run_kmeans
from data.partitioner import resolve_project_path
from models.encoders import create_client_encoder, resolve_encoder_type


def _load_clients(partition_dir: Path):
    train_dir = partition_dir / "train_clients"
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train client directory: {train_dir}. Run Stage 1 first.")
    paths = sorted(train_dir.glob("client_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No client_*.pt files found in {train_dir}. Run Stage 1 first.")
    return [torch.load(path, map_location="cpu") for path in paths]


class _AutoEncoder(nn.Module):
    def __init__(self, client: dict, cfg: dict):
        super().__init__()
        input_dim = int(client["input_dim"])
        hidden_dim = int(cfg.get("encoder_hidden_dim", 128))
        self.encoder = create_client_encoder(
            cfg,
            input_dim=input_dim,
            input_shape=client.get("input_shape"),
            modality_name=client.get("modality_name"),
        )
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def _pretrain_client_encoder(client, cfg, device):
    pre_cfg = cfg.get("pretrain", {})
    input_dim = int(client["input_dim"])
    model = _AutoEncoder(client, cfg).to(device)

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
            loss = loss_fn(recon, xb.reshape(xb.shape[0], -1))
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


def _cluster_by_input_dim_if_available(clients, known_k):
    input_dims = [int(client["input_dim"]) for client in clients]
    unique_dims = sorted(set(input_dims))
    if len(unique_dims) != int(known_k):
        return None
    dim_to_cluster = {dim: idx for idx, dim in enumerate(unique_dims)}
    return np.array([dim_to_cluster[dim] for dim in input_dims], dtype=int), dim_to_cluster


def _append_input_dim_hint(fingerprints, clients, repeat):
    repeat = max(1, int(repeat))
    dims = np.array([float(client["input_dim"]) for client in clients], dtype=np.float32)
    dims = (dims - dims.mean()) / (dims.std() + 1e-6)
    hint = np.repeat(dims[:, None], repeat, axis=1)
    return [row for row in np.concatenate([np.stack(fingerprints), hint], axis=1)]


def run_stage2_pretrain_cluster(cfg: dict, project_root: Path, device: torch.device):
    partition_dir = resolve_project_path(project_root, cfg.get("partition", {}).get("output_dir", "results/data_partition"))
    cluster_dir = resolve_project_path(project_root, cfg.get("cluster", {}).get("output_dir", "results/cluster"))
    result_dir = resolve_project_path(project_root, cfg.get("result", {}).get("output_dir", "results/logs"))
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
                "input_shape": [int(v) for v in client.get("input_shape", [int(client["input_dim"])])],
                "hidden_dim": int(cfg.get("encoder_hidden_dim", 128)),
                "encoder_type": resolve_encoder_type(cfg, client.get("modality_name")),
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
    input_dim_hint = {
        "enabled": bool(cluster_cfg.get("use_input_dim_hint", False)),
        "mode": "disabled",
        "dim_to_cluster": None,
    }
    clustering_features = fingerprints
    if input_dim_hint["enabled"]:
        exact = _cluster_by_input_dim_if_available(clients, known_k)
        if exact is not None:
            pred, dim_to_cluster = exact
            input_dim_hint["mode"] = "unique_input_dim"
            input_dim_hint["dim_to_cluster"] = {str(k): int(v) for k, v in dim_to_cluster.items()}
        else:
            repeat = int(cluster_cfg.get("input_dim_hint_repeat", 16))
            clustering_features = _append_input_dim_hint(fingerprints, clients, repeat)
            input_dim_hint["mode"] = "augmented_fingerprint"
            input_dim_hint["repeat"] = repeat

    if input_dim_hint["mode"] != "unique_input_dim":
        if method == "kmeans":
            pred = run_kmeans(clustering_features, known_k, seed=seed)
        elif method == "isodata":
            iso_kwargs = dict(cluster_cfg.get("isodata", {}))
            pred = run_isodata(clustering_features, known_k, seed=seed, **iso_kwargs)
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
        "input_dim_hint": input_dim_hint,
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

