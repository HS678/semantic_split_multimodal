import csv
import json
import random
import shutil
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from data.partitioner import resolve_project_path
from models.modules import ClientEncoder


class Stage3Client:
    def __init__(self, payload, pred_cluster: int, encoder_state: dict, cfg: dict, device: torch.device):
        self.client_id = payload["client_id"]
        self.pred_cluster = int(pred_cluster)
        self.input_dim = int(payload["input_dim"])
        self.x = payload["x"].to(device)
        self.y = payload["y"].to(device)
        self.device = device
        self.batch_size = int(cfg.get("training", {}).get("batch_size", cfg.get("batch_size", 32)))
        hidden_dim = int(cfg.get("encoder_hidden_dim", 128))
        self.encoder = ClientEncoder(self.input_dim, hidden_dim).to(device)
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
    def __init__(self, hidden_dim: int, num_clusters: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_clusters, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

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


def _train_round(server, server_optimizer, clients, selected, cfg, device):
    r = int(cfg.get("training", {}).get("clients_per_cluster_per_round", 1))
    batch_size = int(cfg.get("training", {}).get("batch_size", cfg.get("batch_size", 32)))
    cluster_ids = sorted(selected.keys())
    ce = nn.CrossEntropyLoss()
    group_losses = []
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
        loss = ce(logits, y)
        group_losses.append(loss)
        group_correct += int((logits.argmax(dim=1) == y).sum().item())
        group_total += int(y.numel())

    total_loss = torch.stack(group_losses).mean()
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
        "accuracy": float(group_correct / max(1, group_total)),
        "feature_map": {f"{k[0]}:{k[1]}": v for k, v in feature_map.items()},
        "grad_return_count": int(grad_return_count),
        "K_t": int(len(cluster_ids) * r),
        "Q_star": int(len(cluster_ids)),
    }


def _evaluate(server, cluster_ids, cluster_input_dims, cfg, test_payload, device):
    hidden_dim = int(cfg.get("encoder_hidden_dim", 128))
    cluster_metrics_path = cfg.get("_cluster_metrics_path")
    with open(cluster_metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    # Evaluation maps predicted clusters to paired test modalities via Stage 2 majority metadata.
    # This mapping is not used by the training scheduler.
    cluster_to_true = {int(k): int(v) for k, v in metrics.get("cluster_to_true_modality_majority", {}).items()}

    encoder_dir = Path(cfg["_cluster_dir"]) / "pretrained_encoders"
    assignment_path = Path(cfg["_cluster_dir"]) / "cluster_assignments.csv"
    with assignment_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    modality_dims = {0: int(test_payload["acc"].shape[1]), 1: int(test_payload["gyro"].shape[1])}
    encoders = {}
    for cid in cluster_ids:
        true_modality = int(cluster_to_true.get(cid, cid))
        expected_dim = modality_dims[true_modality]
        candidate_rows = [row for row in rows if int(row["pred_cluster"]) == cid]
        chosen_payload = None
        for row in candidate_rows:
            payload = torch.load(encoder_dir / f"{row['client_id']}_encoder.pt", map_location="cpu")
            if int(payload["input_dim"]) == expected_dim:
                chosen_payload = payload
                break
        if chosen_payload is None:
            for row in rows:
                payload = torch.load(encoder_dir / f"{row['client_id']}_encoder.pt", map_location="cpu")
                if int(payload["input_dim"]) == expected_dim:
                    chosen_payload = payload
                    break
        if chosen_payload is None:
            raise RuntimeError(f"No pretrained encoder with input_dim={expected_dim} for eval cluster {cid}.")
        encoder = ClientEncoder(expected_dim, hidden_dim).to(device)
        encoder.load_state_dict(chosen_payload["state_dict"])
        encoders[int(cid)] = encoder

    batch_size = int(cfg.get("training", {}).get("eval_batch_size", cfg.get("training", {}).get("batch_size", 64)))
    label = test_payload["label"]
    acc = test_payload["acc"]
    gyro = test_payload["gyro"]
    loader = DataLoader(TensorDataset(acc, gyro, label), batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    loss_total = 0.0
    ce = nn.CrossEntropyLoss(reduction="sum")
    server.eval()
    for encoder in encoders.values():
        encoder.eval()
    with torch.no_grad():
        for acc_b, gyro_b, y_b in loader:
            acc_b = acc_b.to(device)
            gyro_b = gyro_b.to(device)
            y_b = y_b.to(device)
            features = []
            for cid in cluster_ids:
                true_modality = cluster_to_true.get(cid, cid)
                xb = acc_b if true_modality == 0 else gyro_b
                features.append(encoders[int(cid)](xb))
            logits = server(features)
            loss_total += float(ce(logits, y_b).item())
            correct += int((logits.argmax(dim=1) == y_b).sum().item())
            total += int(y_b.numel())
    server.train()
    return {"loss": float(loss_total / max(1, total)), "accuracy": float(correct / max(1, total))}


def run_stage3_split_training(cfg: dict, project_root: Path, device: torch.device):
    partition_dir = resolve_project_path(project_root, cfg.get("partition", {}).get("output_dir", "data_partition"))
    cluster_dir = resolve_project_path(project_root, cfg.get("cluster", {}).get("output_dir", "cluster"))
    result_dir = resolve_project_path(project_root, cfg.get("result", {}).get("output_dir", "result"))
    model_dir = resolve_project_path(project_root, cfg.get("result_model", {}).get("output_dir", "result_model"))
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
    server = ServerFusionClassifier(hidden_dim, len(cluster_ids), num_classes).to(device)
    server_optimizer = torch.optim.Adam(server.parameters(), lr=float(cfg.get("training", {}).get("server_lr", cfg.get("learning_rate", 1e-3))))
    rng = random.Random(int(cfg.get("seed", 42)))

    cluster_input_dims = {cid: cluster_to_clients[cid][0].input_dim for cid in cluster_ids}
    test_path = partition_dir / "test_multimodal.pt"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing paired multimodal test data: {test_path}. Run Stage 1 first.")
    test_payload = torch.load(test_path, map_location="cpu")

    train_log_path = result_dir / "train_log.csv"
    eval_log_path = result_dir / "eval_log.csv"
    train_fields = ["round", "loss", "accuracy", "Q_star", "r", "K_t", "grad_return_count", "feature_map_json"]
    eval_fields = ["round", "loss", "accuracy"]
    best_metrics = {"accuracy": -1.0, "loss": None, "round": None}

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
                    "accuracy": train_metrics["accuracy"],
                    "Q_star": train_metrics["Q_star"],
                    "r": r,
                    "K_t": train_metrics["K_t"],
                    "grad_return_count": train_metrics["grad_return_count"],
                    "feature_map_json": json.dumps(train_metrics["feature_map"], sort_keys=True),
                }
            )

            if round_idx % eval_every == 0 or round_idx == rounds:
                final_eval = _evaluate(server, cluster_ids, cluster_input_dims, cfg, test_payload, device)
                eval_writer.writerow({"round": round_idx, "loss": final_eval["loss"], "accuracy": final_eval["accuracy"]})
                if final_eval["accuracy"] > best_metrics["accuracy"]:
                    best_metrics = {"round": round_idx, **final_eval}
                    torch.save(server.state_dict(), model_dir / "best_server_model.pt")
                    for client in clients.values():
                        torch.save(
                            {
                                "client_id": client.client_id,
                                "pred_cluster": int(client.pred_cluster),
                                "input_dim": int(client.input_dim),
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

