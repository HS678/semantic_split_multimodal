import csv
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from semantic_split_multimodal.data.client import Client
from semantic_split_multimodal.data.partitioner import resolve_project_path
from semantic_split_multimodal.evaluation.baseline_eval import evaluate_naturally_paired_unpaired_baseline
from semantic_split_multimodal.evaluation.metrics import d2d_metrics, learning_metrics
from semantic_split_multimodal.evaluation.oracle_mapping import build_oracle_eval_mapping
from semantic_split_multimodal.learning.models import (
    ClassifierHead,
    ClusterAdapter,
    SharedSemanticBackbone,
    create_client_encoder,
)
from semantic_split_multimodal.learning.scheduling import build_scheduler


class PrototypeBank:
    def __init__(self, momentum=0.9):
        self.momentum = float(momentum)
        self.bank = {}

    def alignment_loss(self, semantics, labels, cluster_ids):
        if semantics.numel() == 0:
            return semantics.new_tensor(0.0)
        current = {}
        for cluster_id in torch.unique(cluster_ids).detach().cpu().tolist():
            c_mask = cluster_ids == int(cluster_id)
            for class_id in torch.unique(labels[c_mask]).detach().cpu().tolist():
                mask = c_mask & (labels == int(class_id))
                current[(int(cluster_id), int(class_id))] = semantics[mask].mean(dim=0)

        targets = []
        preds = []
        for (cluster_id, class_id), proto in current.items():
            other = [
                value.to(proto.device)
                for (c, y), value in self.bank.items()
                if int(y) == int(class_id) and int(c) != int(cluster_id)
            ]
            if other:
                preds.append(proto)
                targets.append(torch.stack(other, dim=0).mean(dim=0).detach())
        if not preds:
            return semantics.new_tensor(0.0)
        return nn.functional.mse_loss(torch.stack(preds, dim=0), torch.stack(targets, dim=0))

    def update(self, semantics, labels, cluster_ids):
        with torch.no_grad():
            for cluster_id in torch.unique(cluster_ids).detach().cpu().tolist():
                c_mask = cluster_ids == int(cluster_id)
                for class_id in torch.unique(labels[c_mask]).detach().cpu().tolist():
                    mask = c_mask & (labels == int(class_id))
                    proto = semantics[mask].mean(dim=0).detach().cpu()
                    key = (int(cluster_id), int(class_id))
                    if key in self.bank:
                        self.bank[key] = self.momentum * self.bank[key] + (1.0 - self.momentum) * proto
                    else:
                        self.bank[key] = proto


class SplitClient:
    def __init__(self, client: Client, encoder_state: dict, cfg: dict, device):
        self.client = client
        self.client_id = client.client_id
        self.pred_cluster = int(client.pred_cluster)
        self.device = device
        self.samples = client.samples.to(device)
        self.labels = client.labels.to(device)
        self.batch_size = int(cfg.get("training", {}).get("batch_size", cfg.get("batch_size", 32)))
        self.encoder = create_client_encoder(cfg, input_shape=client.input_shape, encoder_type=client.encoder_type).to(device)
        self.encoder.load_state_dict(encoder_state)
        self.optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=float(cfg.get("training", {}).get("client_lr", cfg.get("learning_rate", 1e-3))),
        )

    @property
    def hidden_modality_id(self):
        return self.client.hidden_modality_id

    def sample_batch(self):
        n = int(self.samples.shape[0])
        replace = n < self.batch_size
        idx = torch.randint(0, n, (self.batch_size,), device=self.device) if replace else torch.randperm(n, device=self.device)[: self.batch_size]
        return self.samples[idx], self.labels[idx]

    def forward_detached(self, x):
        z_client = self.encoder(x)
        z_server = z_client.detach().requires_grad_(True)
        return z_client, z_server

    def backward_from_server(self, z_client, grad):
        self.optimizer.zero_grad()
        z_client.backward(grad)
        self.optimizer.step()


class SharedSemanticServer(nn.Module):
    def __init__(self, cluster_ids, feature_dim, num_classes, cfg):
        super().__init__()
        server_cfg = cfg.get("model", {}).get("server", {})
        adapter_dim = int(server_cfg.get("adapter_dim", feature_dim))
        semantic_dim = int(server_cfg.get("semantic_dim", adapter_dim))
        self.adapters = nn.ModuleDict(
            {
                str(int(cid)): ClusterAdapter(
                    feature_dim,
                    adapter_dim,
                    hidden_dim=server_cfg.get("adapter_hidden_dim"),
                    dropout=float(server_cfg.get("dropout", 0.0)),
                )
                for cid in cluster_ids
            }
        )
        self.semantic_backbone = SharedSemanticBackbone(
            adapter_dim,
            out_dim=semantic_dim,
            hidden_dim=server_cfg.get("hidden_dim", semantic_dim),
            num_layers=int(server_cfg.get("num_layers", 1)),
            dropout=float(server_cfg.get("dropout", 0.0)),
        )
        self.classifier = ClassifierHead(semantic_dim, num_classes)

    def forward_cluster(self, cluster_id, activation):
        adapted = self.adapters[str(int(cluster_id))](activation)
        semantic = self.semantic_backbone(adapted)
        return self.classifier(semantic), semantic


def _read_assignments(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing cluster assignments: {path}. Run Stage 2 first.")
    rows = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["client_id"]] = int(row["pred_cluster"])
    return rows


def _load_clients(cfg, partition_dir: Path, cluster_dir: Path, device):
    assignments = _read_assignments(cluster_dir / "cluster_assignments.csv")
    encoder_dir = cluster_dir / "pretrained_encoders"
    clients = []
    for path in sorted((partition_dir / "train_clients").glob("client_*.pt")):
        client = Client.from_payload(torch.load(path, map_location="cpu"), pred_cluster=assignments[path.stem])
        enc_payload = torch.load(encoder_dir / f"{client.client_id}_encoder.pt", map_location="cpu")
        clients.append(SplitClient(client, enc_payload["state_dict"], cfg, device))
    if not clients:
        raise FileNotFoundError(f"No client_*.pt files found under {partition_dir / 'train_clients'}.")
    return clients


def _train_round(server, server_optimizer, prototype_bank, selected, cfg):
    ce = nn.CrossEntropyLoss()
    align_cfg = cfg.get("alignment", {})
    lambda_align = float(align_cfg.get("lambda_align", 0.0)) if bool(align_cfg.get("enabled", True)) else 0.0
    client_paths = []
    logits_parts = []
    semantics_parts = []
    label_parts = []
    cluster_parts = []
    correct = 0
    total = 0
    server_optimizer.zero_grad()
    for client in selected:
        x, y = client.sample_batch()
        z_client, z_server = client.forward_detached(x)
        logits, semantic = server.forward_cluster(client.pred_cluster, z_server)
        logits_parts.append(logits)
        semantics_parts.append(semantic)
        label_parts.append(y)
        cluster_parts.append(torch.full_like(y, int(client.pred_cluster)))
        client_paths.append((client, z_client, z_server))
        correct += int((logits.argmax(dim=1) == y).sum().item())
        total += int(y.numel())

    logits = torch.cat(logits_parts, dim=0)
    semantics = torch.cat(semantics_parts, dim=0)
    labels = torch.cat(label_parts, dim=0)
    cluster_ids = torch.cat(cluster_parts, dim=0)
    loss_cls = ce(logits, labels)
    loss_align = prototype_bank.alignment_loss(semantics, labels, cluster_ids) if lambda_align > 0 else loss_cls.new_tensor(0.0)
    loss = loss_cls + lambda_align * loss_align
    loss.backward()
    server_optimizer.step()
    for client, z_client, z_server in client_paths:
        if z_server.grad is None:
            raise RuntimeError(f"Missing server gradient for client {client.client_id}")
        client.backward_from_server(z_client, z_server.grad.detach())
    prototype_bank.update(semantics.detach(), labels.detach(), cluster_ids.detach())
    return {
        "loss": float(loss.item()),
        "loss_cls": float(loss_cls.item()),
        "loss_align": float(loss_align.item()),
        "lambda_align": float(lambda_align),
        "accuracy": float(correct / max(1, total)),
        "K_t": int(len(selected)),
    }


def _evaluate(server, clients, cfg, device):
    batch_size = int(cfg.get("training", {}).get("eval_batch_size", cfg.get("training", {}).get("batch_size", 64)))
    y_true, y_pred, modality_ids = [], [], []
    ce = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total = 0
    server.eval()
    for client in clients:
        client.encoder.eval()
    with torch.no_grad():
        for client in clients:
            loader = DataLoader(TensorDataset(client.samples, client.labels), batch_size=batch_size, shuffle=False)
            for xb, yb in loader:
                logits, _ = server.forward_cluster(client.pred_cluster, client.encoder(xb.to(device)))
                pred = logits.argmax(dim=1)
                y_true.extend(yb.detach().cpu().tolist())
                y_pred.extend(pred.detach().cpu().tolist())
                modality_ids.extend([int(client.hidden_modality_id)] * int(yb.numel()))
                total_loss += float(ce(logits, yb.to(device)).item())
                total += int(yb.numel())
    server.train()
    for client in clients:
        client.encoder.train()
    metrics = learning_metrics(y_true, y_pred, modality_ids)
    metrics["loss"] = float(total_loss / max(1, total))
    return metrics


def run_unpaired_stage3_split_training(cfg: dict, project_root: Path, device: torch.device):
    partition_dir = resolve_project_path(project_root, cfg.get("partition", {}).get("output_dir", "local/results/data_partition"))
    cluster_dir = resolve_project_path(project_root, cfg.get("cluster", {}).get("output_dir", "local/results/cluster"))
    result_dir = resolve_project_path(project_root, cfg.get("result", {}).get("output_dir", "local/logs"))
    model_dir = resolve_project_path(project_root, cfg.get("result_model", {}).get("output_dir", "local/checkpoints"))
    best_client_dir = model_dir / "best_client_encoders"
    result_dir.mkdir(parents=True, exist_ok=True)
    best_client_dir.mkdir(parents=True, exist_ok=True)

    clients = _load_clients(cfg, partition_dir, cluster_dir, device)
    cluster_ids = sorted({int(client.pred_cluster) for client in clients})
    clients_per_round = int(cfg.get("training", {}).get("clients_per_round", cfg.get("training", {}).get("K_t", len(cluster_ids))))
    scheduler = build_scheduler(
        cfg.get("training", {}).get("scheduler", "proposed_cluster_coverage"),
        clients,
        clients_per_round=clients_per_round,
        seed=int(cfg.get("seed", 42)),
    )
    server = SharedSemanticServer(cluster_ids, int(cfg.get("encoder_hidden_dim", 128)), int(cfg.get("num_classes", 6)), cfg).to(device)
    server_optimizer = torch.optim.Adam(
        server.parameters(),
        lr=float(cfg.get("training", {}).get("server_lr", cfg.get("learning_rate", 1e-3))),
    )
    prototype_bank = PrototypeBank(momentum=float(cfg.get("alignment", {}).get("ema_momentum", 0.9)))
    rounds = int(cfg.get("training", {}).get("global_rounds", cfg.get("global_rounds", 3)))
    eval_every = int(cfg.get("training", {}).get("eval_every", 1))

    train_fields = [
        "round",
        "loss",
        "loss_cls",
        "loss_align",
        "lambda_align",
        "accuracy",
        "K_t",
        "coverage",
        "participation_fairness",
        "latency",
        "speedup",
        "selected_clients_json",
    ]
    eval_fields = [
        "round",
        "eval_status",
        "eval_failure_reason",
        "loss",
        "accuracy",
        "macro_f1",
        "diagnostic_client_loss",
        "diagnostic_client_accuracy",
        "diagnostic_client_macro_f1",
        "diagnostic_client_per_modality_accuracy_json",
    ]
    best_metrics = {"macro_f1": -1.0}
    final_eval = None
    diagnostic_eval = None
    oracle_mapping = None
    clients_by_id = None
    with (result_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as train_f, (
        result_dir / "eval_log.csv"
    ).open("w", newline="", encoding="utf-8") as eval_f:
        train_writer = csv.DictWriter(train_f, fieldnames=train_fields)
        eval_writer = csv.DictWriter(eval_f, fieldnames=eval_fields)
        train_writer.writeheader()
        eval_writer.writeheader()
        for round_idx in range(1, rounds + 1):
            selected = scheduler.sample_round()
            start = time.perf_counter()
            train_metrics = _train_round(server, server_optimizer, prototype_bank, selected, cfg)
            latency = time.perf_counter() - start
            d2d = d2d_metrics(latency_without_d2d=latency, latency_with_d2d=latency)
            sched_metrics = scheduler.metrics(selected)
            train_writer.writerow(
                {
                    "round": round_idx,
                    **train_metrics,
                    "coverage": sched_metrics["coverage"],
                    "participation_fairness": sched_metrics["participation_fairness"],
                    "latency": d2d["latency"],
                    "speedup": d2d["speedup"],
                    "selected_clients_json": json.dumps([c.client_id for c in selected]),
                }
            )
            if round_idx % eval_every == 0 or round_idx == rounds:
                diagnostic_eval = _evaluate(server, clients, cfg, device)
                oracle_mapping = build_oracle_eval_mapping(
                    partition_dir / "client_meta.csv",
                    cluster_dir / "cluster_assignments.csv",
                    model_dir / "oracle_eval_modality_to_cluster.json",
                )
                if clients_by_id is None:
                    clients_by_id = {client.client_id: client for client in clients}
                final_eval = evaluate_naturally_paired_unpaired_baseline(
                    server,
                    clients_by_id,
                    partition_dir / "test_multimodal.pt",
                    oracle_mapping,
                    cfg,
                    device,
                )
                eval_writer.writerow(
                    {
                        "round": round_idx,
                        "eval_status": final_eval["eval_status"],
                        "eval_failure_reason": final_eval["eval_failure_reason"],
                        "loss": final_eval["loss"],
                        "accuracy": final_eval["accuracy"],
                        "macro_f1": final_eval["macro_f1"],
                        "diagnostic_client_loss": diagnostic_eval["loss"],
                        "diagnostic_client_accuracy": diagnostic_eval["accuracy"],
                        "diagnostic_client_macro_f1": diagnostic_eval["macro_f1"],
                        "diagnostic_client_per_modality_accuracy_json": json.dumps(
                            diagnostic_eval["per_modality_accuracy"],
                            sort_keys=True,
                        ),
                    }
                )
                if final_eval["eval_status"] == "success" and final_eval["macro_f1"] > best_metrics["macro_f1"]:
                    best_metrics = {"round": round_idx, **final_eval}
                    torch.save(server.state_dict(), model_dir / "best_server_model.pt")
                    for client in clients:
                        torch.save(
                            {
                                "client_id": client.client_id,
                                "pred_cluster": int(client.pred_cluster),
                                "encoder_type": client.client.encoder_type,
                                "input_shape": client.client.input_shape,
                                "state_dict": client.encoder.cpu().state_dict(),
                            },
                            best_client_dir / f"{client.client_id}_encoder.pt",
                        )
                        client.encoder.to(device)

    final_metrics = {
        "final_eval": final_eval,
        "diagnostic_client_eval": diagnostic_eval,
        "eval_status": None if final_eval is None else final_eval["eval_status"],
        "eval_failure_reason": None if final_eval is None else final_eval["eval_failure_reason"],
        "oracle_eval_mapping": oracle_mapping,
        "cluster_ids": cluster_ids,
        "estimated_num_clusters": len(cluster_ids),
        "clients_per_round": clients_per_round,
        "scheduler": cfg.get("training", {}).get("scheduler", "proposed_cluster_coverage"),
        "multimodal_mode": "unpaired_split_learning",
    }
    with (result_dir / "final_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    with (result_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)
    with (result_dir / "config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with (model_dir / "best_model_info.json").open("w", encoding="utf-8") as f:
        json.dump({"best_metrics": best_metrics, "cluster_ids": cluster_ids}, f, indent=2)
    return final_metrics
