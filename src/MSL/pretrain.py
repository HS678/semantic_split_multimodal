# 模态发现前的单模态 encoder 预训练、fingerprint 提取和聚类产物保存。
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from MSL.discovery import adaptive_isodata, run_kmeans
from MSL.discovery import build_fingerprints
from MSL.data import Client
from MSL.data import resolve_project_path
from MSL.evaluation import discovery_metrics
from tools.plot_fingerprint_embedding import write_fingerprint_pca_figure
from MSL.models import create_client_encoder, flattened_dim


def _load_clients(partition_dir: Path):
    train_dir = partition_dir / "train_clients"
    paths = sorted(train_dir.glob("client_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No client_*.pt files found in {train_dir}. Run client preparation first.")
    return [Client.from_payload(torch.load(path, map_location="cpu")) for path in paths]


class _ClientPretrainModel(nn.Module):
    def __init__(self, client: Client, cfg: dict, objective: str):
        super().__init__()
        self.encoder = create_client_encoder(cfg, input_shape=client.input_shape, encoder_type=client.encoder_type)
        hidden_dim = int(cfg.get("encoder_hidden_dim", 128))
        self.objective = str(objective)
        if self.objective == "reconstruction":
            self.head = nn.Linear(hidden_dim, flattened_dim(client.input_shape))
        elif self.objective == "classification":
            self.head = nn.Linear(hidden_dim, int(cfg.get("num_classes", 2)))
        else:
            raise ValueError("pretrain.objective must be 'reconstruction' or 'classification'.")

    def forward(self, x, lengths=None):
        z = self.encoder(x, lengths)
        return self.head(z), z


def _inverse_sqrt_class_weights(labels, num_classes: int, device):
    counts = torch.bincount(labels.detach().cpu().long(), minlength=int(num_classes)).float()
    weights = torch.zeros_like(counts)
    present = counts > 0
    weights[present] = counts[present].rsqrt()
    if bool(present.any()):
        weights[present] = weights[present] / weights[present].mean()
    return weights.to(device)


def _pretrain_client_encoder(client: Client, cfg: dict, device):
    pre_cfg = cfg.get("pretrain", {})
    objective = str(pre_cfg.get("objective", "reconstruction")).strip().lower()
    model = _ClientPretrainModel(client, cfg, objective).to(device)
    epochs = int(pre_cfg.get("epochs", 5))
    batch_size = int(pre_cfg.get("batch_size", cfg.get("batch_size", 64)))
    lr = float(pre_cfg.get("lr", cfg.get("learning_rate", 1e-3)))
    max_samples = pre_cfg.get("max_samples")
    x = client.samples
    if max_samples is not None:
        x = x[: min(int(max_samples), int(x.shape[0]))]
    labels = client.labels[: x.shape[0]].long()
    lengths = client.sequence_lengths
    if lengths is not None:
        lengths = lengths[: x.shape[0]]
        loader = DataLoader(TensorDataset(x, lengths, labels), batch_size=batch_size, shuffle=True)
    else:
        loader = DataLoader(TensorDataset(x, labels), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=float(pre_cfg.get("weight_decay", 0.0)))
    if objective == "classification":
        class_weighting = str(pre_cfg.get("class_weighting", "none")).strip().lower()
        if class_weighting not in {"none", "inverse_sqrt"}:
            raise ValueError("pretrain.class_weighting must be 'none' or 'inverse_sqrt'.")
        class_weights = (
            _inverse_sqrt_class_weights(labels, int(cfg.get("num_classes", 2)), device)
            if class_weighting == "inverse_sqrt"
            else None
        )
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = nn.MSELoss()
    total = 0.0
    steps = 0
    correct = 0
    total_examples = 0
    model.train()
    for _ in range(max(0, epochs)):
        for batch in loader:
            xb = batch[0]
            if lengths is None:
                length_batch = None
                label_batch = batch[1].to(device)
            else:
                length_batch = batch[1].to(device)
                label_batch = batch[2].to(device)
            xb = xb.to(device)
            output, _ = model(xb, length_batch)
            target = label_batch if objective == "classification" else xb.reshape(xb.shape[0], -1)
            loss = loss_fn(output, target)
            opt.zero_grad()
            loss.backward()
            max_grad_norm = pre_cfg.get("max_grad_norm")
            if max_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
            opt.step()
            total += float(loss.item())
            steps += 1
            if objective == "classification":
                correct += int((output.argmax(dim=1) == label_batch).sum().item())
                total_examples += int(label_batch.numel())
    return model.encoder, {
        "objective": objective,
        "loss": total / steps if steps else None,
        "accuracy": None if objective != "classification" else float(correct / max(1, total_examples)),
        "epochs": epochs,
    }


def discover_modalities(cfg: dict, project_root: Path, device: torch.device):
    partition_dir = resolve_project_path(project_root, cfg.get("partition", {}).get("output_dir", "results/pipeline/clients"))
    cluster_dir = resolve_project_path(project_root, cfg.get("cluster", {}).get("output_dir", "results/pipeline/discovery"))
    encoder_dir = cluster_dir / "pretrained_encoders"
    encoder_dir.mkdir(parents=True, exist_ok=True)

    clients = _load_clients(partition_dir)
    encoders = {}
    losses = {}
    for client in clients:
        encoder, pretrain_metrics = _pretrain_client_encoder(client, cfg, device)
        encoders[client.client_id] = encoder
        losses[client.client_id] = pretrain_metrics
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

    cluster_cfg = cfg.get("cluster", {})
    method = str(cluster_cfg.get("method", "adaptive_isodata")).lower()
    if method == "adaptive":
        method = "adaptive_isodata"
    raw_k = cluster_cfg.get("known_k")
    known_k = None if raw_k is None or str(raw_k).lower() in {"auto", "none", "null"} else int(raw_k)
    seed = int(cfg.get("seed", 42))
    adaptive_diagnostics = None
    if method == "kmeans":
        pred = run_kmeans(fingerprints, known_k, seed=seed)
    elif method == "adaptive_isodata":
        if known_k is not None:
            raise ValueError("cluster.known_k must be null when cluster.method is adaptive_isodata.")
        pred, adaptive_diagnostics = adaptive_isodata(
            fingerprints,
            seed=seed,
            **dict(cluster_cfg.get("adaptive", {})),
        )
    else:
        raise ValueError("cluster.method must be 'kmeans' or 'adaptive_isodata'.")

    true = np.array([client.hidden_modality_id for client in clients], dtype=int)
    pred = np.asarray(pred, dtype=int)
    metrics = discovery_metrics(true, pred)
    metrics.update(
        {
            "method": method,
            "known_k": known_k,
            "true_num_modalities": int(len(np.unique(true))),
            "pretrain_metrics": losses,
            "fingerprint_type": str(cfg.get("fingerprint", {}).get("type", "hybrid")),
            "uses_input_dimension_hint": False,
        }
    )
    if adaptive_diagnostics is not None:
        metrics.update(
            {
                "q_source": adaptive_diagnostics["q_source"],
                "adaptive_estimated_Q": adaptive_diagnostics["estimated_Q"],
                "cluster_sizes": adaptive_diagnostics["cluster_sizes"],
                "pca_diagnostics": adaptive_diagnostics["preprocessing"],
                "split_history": adaptive_diagnostics["split_history"],
                "merge_history": adaptive_diagnostics["merge_history"],
                "split_count": adaptive_diagnostics["split_count"],
                "merge_count": adaptive_diagnostics["merge_count"],
                "convergence_reason": adaptive_diagnostics["convergence_reason"],
                "per_seed_estimated_Q": adaptive_diagnostics["per_seed_estimated_Q"],
                "q_stability": adaptive_diagnostics["q_stability"],
                "assignment_stability": adaptive_diagnostics["assignment_stability"],
                "selection_confidence": adaptive_diagnostics["selection_confidence"],
                "boundary_saturation": adaptive_diagnostics["boundary_saturation"],
                "minimum_cluster_size": adaptive_diagnostics["minimum_cluster_size"],
                "small_cluster_present": adaptive_diagnostics["small_cluster_present"],
                "silhouette": adaptive_diagnostics["silhouette"],
                "DBI": adaptive_diagnostics["DBI"],
                "CH": adaptive_diagnostics["CH"],
                "adaptive_algorithm_config": adaptive_diagnostics["algorithm_config"],
            }
        )

    true_rows = []
    pred_rows = []
    for client, pred_cluster in zip(clients, pred):
        true_rows.append(
            {
                "client_id": client.client_id,
                "true_cluster": int(client.hidden_modality_id),
            }
        )
        pred_rows.append(
            {
                "client_id": client.client_id,
                "pred_cluster": int(pred_cluster),
            }
        )
    with (cluster_dir / "true_cluster.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["client_id", "true_cluster"])
        writer.writeheader()
        writer.writerows(true_rows)
    with (cluster_dir / "pred_cluster.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["client_id", "pred_cluster"])
        writer.writeheader()
        writer.writerows(pred_rows)
    visualization_cfg = cfg.get("fingerprint_visualization", {})
    if bool(visualization_cfg.get("enabled", True)):
        visualization_dir = cluster_dir / "visualization"
        visualization_dir.mkdir(parents=True, exist_ok=True)
        write_fingerprint_pca_figure(
            fingerprints,
            [client.client_id for client in clients],
            true,
            pred,
            visualization_dir,
            cfg.get("dataset", {}).get("name", cfg.get("dataset", {}).get("type", "dataset")),
            visualization_cfg=visualization_cfg,
        )
    return metrics
