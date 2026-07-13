import csv
import json
import random
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml
from sklearn.metrics import confusion_matrix, f1_score

from data.partitioner import resolve_project_path
from models.encoders import create_client_encoder


class Stage3Client:
    def __init__(self, payload, pred_cluster: int, encoder_state: dict, cfg: dict, device: torch.device):
        self.client_id = payload["client_id"]
        self.pred_cluster = int(pred_cluster)
        self.input_dim = int(payload["input_dim"])
        self.input_shape = [int(v) for v in payload.get("input_shape", [self.input_dim])]
        self.modality_name = payload.get("modality_name")
        self.x = payload["x"].to(device)
        self.y = payload["y"].to(device)
        self.device = device
        self.batch_size = int(cfg.get("training", {}).get("batch_size", cfg.get("batch_size", 32)))
        self.encoder = create_client_encoder(
            cfg,
            input_dim=self.input_dim,
            input_shape=self.input_shape,
            modality_name=self.modality_name,
        ).to(device)
        self.encoder.load_state_dict(encoder_state)
        lr = float(cfg.get("training", {}).get("client_lr", cfg.get("learning_rate", 1e-3)))
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)

    def sample_batch(self):
        replace = int(self.x.shape[0]) < self.batch_size
        idx = torch.randint(0, int(self.x.shape[0]), (self.batch_size,), device=self.device) if replace else torch.randperm(int(self.x.shape[0]), device=self.device)[: self.batch_size]
        return self.x[idx], self.y[idx]

    def forward_detached(self, x):
        z_client = self.encoder(x)
        z_server = z_client.detach().requires_grad_(True)
        return z_client, z_server

    def backward_from_server(self, z_client, grad):
        self.optimizer.zero_grad()
        z_client.backward(grad)
        self.optimizer.step()


class ServerFusionClassifier(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_clusters: int,
        num_classes: int,
        server_hidden_dim: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        fusion_hidden_dim = int(server_hidden_dim or hidden_dim)
        num_layers = max(1, int(num_layers))
        layers = []
        in_dim = int(hidden_dim) * int(num_clusters)
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(in_dim, fusion_hidden_dim),
                    nn.ReLU(),
                ]
            )
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            in_dim = fusion_hidden_dim
        layers.append(nn.Linear(fusion_hidden_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, features):
        return self.classifier(torch.cat(features, dim=1))


def _read_assignments(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing cluster assignments: {path}. Run Stage 2 first.")
    result = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result[row["client_id"]] = int(row["pred_cluster"])
    return result


def _load_stage3_clients(cfg, partition_dir: Path, cluster_dir: Path, device: torch.device):
    assignments = _read_assignments(cluster_dir / "cluster_assignments.csv")
    train_dir = partition_dir / "train_clients"
    encoder_dir = cluster_dir / "pretrained_encoders"
    clients = {}
    for path in sorted(train_dir.glob("client_*.pt")):
        payload = torch.load(path, map_location="cpu")
        client_id = payload["client_id"]
        if client_id not in assignments:
            raise KeyError(f"Client {client_id} is missing in cluster_assignments.csv")
        enc_path = encoder_dir / f"{client_id}_encoder.pt"
        if not enc_path.exists():
            raise FileNotFoundError(f"Missing pretrained encoder for {client_id}: {enc_path}. Run Stage 2 first.")
        enc_payload = torch.load(enc_path, map_location="cpu")
        clients[client_id] = Stage3Client(payload, assignments[client_id], enc_payload["state_dict"], cfg, device)
    if not clients:
        raise FileNotFoundError(f"No Stage 3 clients found under {train_dir}. Run Stage 1 first.")
    return clients


def build_cluster_to_clients(clients: dict):
    cluster_to_clients = {}
    for client in clients.values():
        cluster_to_clients.setdefault(client.pred_cluster, []).append(client)
    return {cluster_id: members for cluster_id, members in sorted(cluster_to_clients.items())}


def balanced_cluster_schedule(cluster_to_clients: dict, r: int, rng: random.Random):
    selected = {}
    for cluster_id, members in cluster_to_clients.items():
        if len(members) >= r:
            picks = rng.sample(members, r)
        else:
            picks = [rng.choice(members) for _ in range(r)]
        selected[int(cluster_id)] = {group_id: picks[group_id] for group_id in range(r)}
    return selected


def _paired_group_batch(group_clients, batch_size: int, device: torch.device):
    labels_per_client = [set(c.y.detach().cpu().tolist()) for c in group_clients]
    common = sorted(set.intersection(*labels_per_client)) if labels_per_client else []
    if not common:
        raise RuntimeError("Selected clients do not share any label; cannot build a modality-complete group batch.")
    y_out = []
    xs = {c.client_id: [] for c in group_clients}
    for _ in range(batch_size):
        label = random.choice(common)
        y_out.append(label)
        for c in group_clients:
            idxs = (c.y == label).nonzero(as_tuple=False).squeeze(1)
            take = idxs[torch.randint(0, int(idxs.numel()), (1,), device=device).item()]
            xs[c.client_id].append(c.x[take])
    x_out = {cid: torch.stack(vals, dim=0) for cid, vals in xs.items()}
    y = torch.tensor(y_out, dtype=torch.long, device=device)
    return x_out, y


def _mse_alignment_loss(features):
    if len(features) < 2:
        return features[0].new_tensor(0.0)
    losses = []
    mse = nn.MSELoss()
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            losses.append(mse(features[i], features[j]))
    return torch.stack(losses).mean()


def _supervised_contrastive_alignment_loss(features, labels, temperature):
    if len(features) < 2:
        return features[0].new_tensor(0.0)
    views = torch.stack(features, dim=1)
    bsz, num_views, dim = views.shape
    z = F.normalize(views.reshape(bsz * num_views, dim), dim=1)
    y = labels.repeat_interleave(num_views)
    logits = torch.matmul(z, z.T) / float(temperature)
    self_mask = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    positive_mask = (y[:, None] == y[None, :]) & (~self_mask)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * (~self_mask).float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not bool(valid.any()):
        return z.new_tensor(0.0)
    mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1)[valid] / positive_count[valid].float()
    return -mean_log_prob_pos.mean()


def _classwise_prototype_alignment_loss(features, labels):
    if len(features) < 2:
        return features[0].new_tensor(0.0)
    losses = []
    mse = nn.MSELoss()
    for label in torch.unique(labels):
        mask = labels == label
        if int(mask.sum().item()) == 0:
            continue
        prototypes = [feature[mask].mean(dim=0) for feature in features]
        for i in range(len(prototypes)):
            for j in range(i + 1, len(prototypes)):
                losses.append(mse(prototypes[i], prototypes[j]))
    if not losses:
        return features[0].new_tensor(0.0)
    return torch.stack(losses).mean()


def _alignment_loss(features, labels, align_cfg):
    loss_type = str(align_cfg.get("type", "mse")).lower()
    if loss_type == "mse":
        return _mse_alignment_loss(features)
    if loss_type == "classwise_mse":
        return _classwise_prototype_alignment_loss(features, labels)
    if loss_type in {"supervised_contrastive", "supcon"}:
        return _supervised_contrastive_alignment_loss(
            features,
            labels,
            temperature=float(align_cfg.get("temperature", 0.2)),
        )
    raise ValueError(
        f"Unsupported alignment.type: {loss_type}. Expected 'mse', 'classwise_mse', or 'supervised_contrastive'."
    )


def _train_round(server, server_optimizer, clients, selected, cfg, device):
    r = int(cfg.get("training", {}).get("clients_per_cluster_per_round", 1))
    batch_size = int(cfg.get("training", {}).get("batch_size", cfg.get("batch_size", 32)))
    align_cfg = cfg.get("alignment", {})
    lambda_align = float(align_cfg.get("lambda_align", 0.0)) if bool(align_cfg.get("enabled", False)) else 0.0
    cluster_ids = sorted(selected.keys())
    ce = nn.CrossEntropyLoss()
    group_losses = []
    group_cls_losses = []
    group_align_losses = []
    group_correct = 0
    group_total = 0
    feature_map = {}
    client_paths = {}

    server_optimizer.zero_grad()
    for group_id in range(r):
        group_clients = [selected[cluster_id][group_id] for cluster_id in cluster_ids]
        x_by_client, y = _paired_group_batch(group_clients, batch_size, device)
        features = []
        for cluster_id, client in zip(cluster_ids, group_clients):
            z_client, z_server = client.forward_detached(x_by_client[client.client_id])
            feature_map[(int(cluster_id), int(group_id))] = client.client_id
            client_paths[(int(cluster_id), int(group_id))] = (client, z_client, z_server)
            features.append(z_server)
        logits = server(features)
        loss_cls = ce(logits, y)
        loss_align = _alignment_loss(features, y, align_cfg)
        loss = loss_cls + lambda_align * loss_align
        group_losses.append(loss)
        group_cls_losses.append(loss_cls)
        group_align_losses.append(loss_align)
        group_correct += int((logits.argmax(dim=1) == y).sum().item())
        group_total += int(y.numel())

    total_loss = torch.stack(group_losses).mean()
    total_cls_loss = torch.stack(group_cls_losses).mean()
    total_align_loss = torch.stack(group_align_losses).mean()
    total_loss.backward()
    server_optimizer.step()

    grad_return_count = 0
    for key, (client, z_client, z_server) in client_paths.items():
        if z_server.grad is None:
            raise RuntimeError(f"Missing server gradient for feature_map key {key}, client {client.client_id}")
        client.backward_from_server(z_client, z_server.grad.detach())
        grad_return_count += 1

    return {
        "loss": float(total_loss.item()),
        "loss_cls": float(total_cls_loss.item()),
        "loss_align": float(total_align_loss.item()),
        "lambda_align": float(lambda_align),
        "accuracy": float(group_correct / max(1, group_total)),
        "feature_map": {f"{k[0]}:{k[1]}": v for k, v in feature_map.items()},
        "grad_return_count": int(grad_return_count),
        "K_t": int(len(cluster_ids) * r),
        "Q_star": int(len(cluster_ids)),
    }


def _evaluate(server, cluster_ids, cluster_to_clients, cfg, test_payload, device):
    cluster_metrics_path = cfg.get("_cluster_metrics_path")
    with open(cluster_metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    # Evaluation maps predicted clusters to paired test modalities via Stage 2 majority metadata.
    # This mapping is not used by the training scheduler.
    cluster_to_true = {int(k): int(v) for k, v in metrics.get("cluster_to_true_modality_majority", {}).items()}

    modality_names = list(test_payload.get("modality_names", []))
    if not modality_names:
        modality_names = list(test_payload.get("modalities", {}).keys())
    if not modality_names:
        raise RuntimeError("test_multimodal.pt must contain modality_names or a modalities mapping.")
    modality_tensors = [test_payload["modalities"][name] for name in modality_names]
    modality_dims = {idx: int(x.reshape(int(test_payload["label"].shape[0]), -1).shape[1]) for idx, x in enumerate(modality_tensors)}
    encoders = {}
    for cid in cluster_ids:
        true_modality = int(cluster_to_true.get(cid, cid))
        if true_modality not in modality_dims:
            raise RuntimeError(
                f"Cluster {cid} maps to modality id {true_modality}, but test data only has "
                f"{len(modality_names)} modalities."
            )
        expected_dim = modality_dims[true_modality]
        candidates = [client for client in cluster_to_clients[int(cid)] if int(client.input_dim) == expected_dim]
        if not candidates:
            raise RuntimeError(f"No trained client encoder with input_dim={expected_dim} for eval cluster {cid}.")
        encoders[int(cid)] = candidates[0].encoder

    batch_size = int(cfg.get("training", {}).get("eval_batch_size", cfg.get("training", {}).get("batch_size", 64)))
    label = test_payload["label"]
    loader = DataLoader(TensorDataset(*modality_tensors, label), batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    loss_total = 0.0
    y_true = []
    y_pred = []
    ce = nn.CrossEntropyLoss(reduction="sum")
    server.eval()
    for encoder in encoders.values():
        encoder.eval()
    with torch.no_grad():
        for batch in loader:
            *modality_batches, y_b = batch
            modality_batches = [x.to(device) for x in modality_batches]
            y_b = y_b.to(device)
            features = []
            for cid in cluster_ids:
                true_modality = cluster_to_true.get(cid, cid)
                xb = modality_batches[int(true_modality)]
                features.append(encoders[int(cid)](xb))
            logits = server(features)
            pred = logits.argmax(dim=1)
            loss_total += float(ce(logits, y_b).item())
            correct += int((pred == y_b).sum().item())
            total += int(y_b.numel())
            y_true.extend(y_b.detach().cpu().tolist())
            y_pred.extend(pred.detach().cpu().tolist())
    server.train()
    labels = list(range(int(cfg.get("num_classes", 6))))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_total = cm.sum(axis=1)
    per_class_correct = cm.diagonal()
    per_class_accuracy = {
        str(label): (float(per_class_correct[label] / per_class_total[label]) if per_class_total[label] > 0 else 0.0)
        for label in labels
    }
    return {
        "loss": float(loss_total / max(1, total)),
        "accuracy": float(correct / max(1, total)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "per_class_accuracy": per_class_accuracy,
    }


def run_stage3_split_training(cfg: dict, project_root: Path, device: torch.device):
    partition_dir = resolve_project_path(project_root, cfg.get("partition", {}).get("output_dir", "results/data_partition"))
    cluster_dir = resolve_project_path(project_root, cfg.get("cluster", {}).get("output_dir", "results/cluster"))
    result_dir = resolve_project_path(project_root, cfg.get("result", {}).get("output_dir", "results/logs"))
    model_dir = resolve_project_path(project_root, cfg.get("result_model", {}).get("output_dir", "results/models"))
    best_client_dir = model_dir / "best_client_encoders"
    result_dir.mkdir(parents=True, exist_ok=True)
    best_client_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(cfg)
    cfg["_cluster_dir"] = str(cluster_dir)
    cfg["_cluster_metrics_path"] = str(cluster_dir / "cluster_metrics.json")

    clients = _load_stage3_clients(cfg, partition_dir, cluster_dir, device)
    cluster_to_clients = build_cluster_to_clients(clients)
    cluster_ids = sorted(cluster_to_clients.keys())
    r = int(cfg.get("training", {}).get("clients_per_cluster_per_round", 1))
    hidden_dim = int(cfg.get("encoder_hidden_dim", 128))
    num_classes = int(cfg.get("num_classes", 6))
    server_cfg = cfg.get("model", {}).get("server", {})
    server = ServerFusionClassifier(
        hidden_dim,
        len(cluster_ids),
        num_classes,
        server_hidden_dim=server_cfg.get("hidden_dim"),
        num_layers=int(server_cfg.get("num_layers", 1)),
        dropout=float(server_cfg.get("dropout", 0.0)),
    ).to(device)
    server_optimizer = torch.optim.Adam(server.parameters(), lr=float(cfg.get("training", {}).get("server_lr", cfg.get("learning_rate", 1e-3))))
    rng = random.Random(int(cfg.get("seed", 42)))

    cluster_input_dims = {cid: cluster_to_clients[cid][0].input_dim for cid in cluster_ids}
    test_path = partition_dir / "test_multimodal.pt"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing paired multimodal test data: {test_path}. Run Stage 1 first.")
    test_payload = torch.load(test_path, map_location="cpu")

    train_log_path = result_dir / "train_log.csv"
    eval_log_path = result_dir / "eval_log.csv"
    train_fields = [
        "round",
        "loss",
        "loss_cls",
        "loss_align",
        "lambda_align",
        "accuracy",
        "Q_star",
        "r",
        "K_t",
        "grad_return_count",
        "feature_map_json",
    ]
    eval_fields = ["round", "loss", "accuracy", "macro_f1", "weighted_f1"]
    best_metrics = {"macro_f1": -1.0, "accuracy": None, "loss": None, "round": None}

    rounds = int(cfg.get("training", {}).get("global_rounds", cfg.get("global_rounds", 3)))
    eval_every = int(cfg.get("training", {}).get("eval_every", 1))
    with train_log_path.open("w", newline="", encoding="utf-8") as train_f, eval_log_path.open("w", newline="", encoding="utf-8") as eval_f:
        train_writer = csv.DictWriter(train_f, fieldnames=train_fields)
        eval_writer = csv.DictWriter(eval_f, fieldnames=eval_fields)
        train_writer.writeheader()
        eval_writer.writeheader()

        final_eval = None
        for round_idx in range(1, rounds + 1):
            selected = balanced_cluster_schedule(cluster_to_clients, r, rng)
            train_metrics = _train_round(server, server_optimizer, clients, selected, cfg, device)
            train_writer.writerow(
                {
                    "round": round_idx,
                    "loss": train_metrics["loss"],
                    "loss_cls": train_metrics["loss_cls"],
                    "loss_align": train_metrics["loss_align"],
                    "lambda_align": train_metrics["lambda_align"],
                    "accuracy": train_metrics["accuracy"],
                    "Q_star": train_metrics["Q_star"],
                    "r": r,
                    "K_t": train_metrics["K_t"],
                    "grad_return_count": train_metrics["grad_return_count"],
                    "feature_map_json": json.dumps(train_metrics["feature_map"], sort_keys=True),
                }
            )

            if round_idx % eval_every == 0 or round_idx == rounds:
                final_eval = _evaluate(server, cluster_ids, cluster_to_clients, cfg, test_payload, device)
                eval_writer.writerow(
                    {
                        "round": round_idx,
                        "loss": final_eval["loss"],
                        "accuracy": final_eval["accuracy"],
                        "macro_f1": final_eval["macro_f1"],
                        "weighted_f1": final_eval["weighted_f1"],
                    }
                )
                if final_eval["macro_f1"] > best_metrics["macro_f1"]:
                    best_metrics = {"round": round_idx, **final_eval}
                    torch.save(server.state_dict(), model_dir / "best_server_model.pt")
                    for client in clients.values():
                        torch.save(
                            {
                                "client_id": client.client_id,
                                "pred_cluster": int(client.pred_cluster),
                                "input_dim": int(client.input_dim),
                                "input_shape": [int(v) for v in client.input_shape],
                                "modality_name": client.modality_name,
                                "state_dict": client.encoder.cpu().state_dict(),
                            },
                            best_client_dir / f"{client.client_id}_encoder.pt",
                        )
                        client.encoder.to(device)

    final_metrics = {
        "final_eval": final_eval,
        "Q_star": len(cluster_ids),
        "r": r,
        "K_t": len(cluster_ids) * r,
        "cluster_ids": cluster_ids,
    }
    with (result_dir / "final_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    with (result_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)
    with (result_dir / "config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with (model_dir / "best_model_info.json").open("w", encoding="utf-8") as f:
        json.dump({"best_metrics": best_metrics, "cluster_ids": cluster_ids, "r": r}, f, indent=2)

    return final_metrics


