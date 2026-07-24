import csv
import itertools
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml
from sklearn.metrics import confusion_matrix, f1_score

from data.partitioner import resolve_project_path
from models.modules import ClassifierHead, ClusterAdapter, SharedSemanticBackbone
from trainers.split_multimodal_trainer import (
    ClusterCyclicScheduler,
    Stage3Client,
    build_cluster_to_clients,
    _read_assignments,
)


class UnpairedSharedSemanticServer(nn.Module):
    def __init__(self, cluster_ids, hidden_dim, num_classes, server_cfg):
        super().__init__()
        adapter_dim = int(server_cfg.get("adapter_dim", hidden_dim))
        semantic_dim = int(server_cfg.get("semantic_dim", adapter_dim))
        dropout = float(server_cfg.get("dropout", 0.0))
        adapter_hidden_dim = server_cfg.get("adapter_hidden_dim")
        self.adapters = nn.ModuleDict(
            {
                str(int(cluster_id)): ClusterAdapter(
                    hidden_dim,
                    adapter_dim,
                    hidden_dim=adapter_hidden_dim,
                    dropout=dropout,
                )
                for cluster_id in cluster_ids
            }
        )
        self.semantic_backbone = SharedSemanticBackbone(
            adapter_dim,
            out_dim=semantic_dim,
            hidden_dim=server_cfg.get("hidden_dim", semantic_dim),
            num_layers=int(server_cfg.get("num_layers", 1)),
            dropout=dropout,
        )
        self.classifier = ClassifierHead(semantic_dim, num_classes)

    def forward_cluster(self, cluster_id, activation):
        adapted = self.adapters[str(int(cluster_id))](activation)
        semantic = self.semantic_backbone(adapted)
        logits = self.classifier(semantic)
        return logits, semantic


def _load_stage3_clients(cfg, partition_dir: Path, cluster_dir: Path, device: torch.device):
    assignments = _read_assignments(cluster_dir / "cluster_assignments.csv")
    train_dir = partition_dir / "train_clients"
    encoder_dir = cluster_dir / "pretrained_encoders"
    use_oracle = str(cfg.get("training", {}).get("cluster_assignment", "predicted")).lower() == "oracle"
    use_oracle = use_oracle or bool(cfg.get("training", {}).get("use_oracle_modality_cluster", False))
    clients = {}
    for path in sorted(train_dir.glob("client_*.pt")):
        payload = torch.load(path, map_location="cpu")
        client_id = payload["client_id"]
        if client_id not in assignments:
            raise KeyError(f"Client {client_id} is missing in cluster_assignments.csv")
        pred_cluster = int(payload["modality_id"]) if use_oracle else int(assignments[client_id])
        enc_path = encoder_dir / f"{client_id}_encoder.pt"
        if not enc_path.exists():
            raise FileNotFoundError(f"Missing pretrained encoder for {client_id}: {enc_path}. Run Stage 2 first.")
        enc_payload = torch.load(enc_path, map_location="cpu")
        clients[client_id] = Stage3Client(payload, pred_cluster, enc_payload["state_dict"], cfg, device)
    if not clients:
        raise FileNotFoundError(f"No Stage 3 clients found under {train_dir}. Run Stage 1 first.")
    return clients


def classwise_prototype_alignment_loss(features_by_cluster, labels_by_cluster):
    cluster_ids = sorted(features_by_cluster.keys())
    if len(cluster_ids) < 2:
        first = features_by_cluster[cluster_ids[0]]
        return first.new_tensor(0.0)

    common_labels = None
    for cluster_id in cluster_ids:
        labels = set(labels_by_cluster[cluster_id].detach().cpu().tolist())
        common_labels = labels if common_labels is None else common_labels & labels
    if not common_labels:
        return features_by_cluster[cluster_ids[0]].new_tensor(0.0)

    losses = []
    mse = nn.MSELoss()
    for label in sorted(common_labels):
        prototypes = []
        for cluster_id in cluster_ids:
            labels = labels_by_cluster[cluster_id]
            mask = labels == int(label)
            if bool(mask.any()):
                prototypes.append(features_by_cluster[cluster_id][mask].mean(dim=0))
        for i in range(len(prototypes)):
            for j in range(i + 1, len(prototypes)):
                losses.append(mse(prototypes[i], prototypes[j]))
    if not losses:
        return features_by_cluster[cluster_ids[0]].new_tensor(0.0)
    return torch.stack(losses).mean()


def _train_round(server, server_optimizer, selected, cfg, device):
    align_cfg = cfg.get("alignment", {})
    lambda_align = float(align_cfg.get("lambda_align", 0.0)) if bool(align_cfg.get("enabled", False)) else 0.0
    align_type = str(align_cfg.get("type", "classwise_prototype")).lower()
    ce = nn.CrossEntropyLoss()
    client_paths = {}
    features_by_cluster = {}
    labels_by_cluster = {}
    cls_losses = []
    correct = 0
    total = 0
    feature_map = {}

    server_optimizer.zero_grad()
    for cluster_id in sorted(selected.keys()):
        cluster_semantics = []
        cluster_labels = []
        for group_id, client in sorted(selected[int(cluster_id)].items()):
            x, y = client.sample_batch()
            z_client, z_server = client.forward_detached(x)
            logits, semantic = server.forward_cluster(cluster_id, z_server)
            loss_cls = ce(logits, y)
            cls_losses.append(loss_cls)
            cluster_semantics.append(semantic)
            cluster_labels.append(y)
            client_paths[(int(cluster_id), int(group_id))] = (client, z_client, z_server)
            feature_map[f"{int(cluster_id)}:{int(group_id)}"] = client.client_id
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(y.numel())
        features_by_cluster[int(cluster_id)] = torch.cat(cluster_semantics, dim=0)
        labels_by_cluster[int(cluster_id)] = torch.cat(cluster_labels, dim=0)

    loss_cls = torch.stack(cls_losses).mean()
    if lambda_align > 0.0:
        if align_type not in {"classwise_prototype", "classwise_mse", "prototype"}:
            raise ValueError(
                "unpaired_shared_semantic supports alignment.type "
                "'classwise_prototype' or 'classwise_mse'."
            )
        loss_align = classwise_prototype_alignment_loss(features_by_cluster, labels_by_cluster)
    else:
        loss_align = loss_cls.new_tensor(0.0)
    loss = loss_cls + lambda_align * loss_align
    loss.backward()
    server_optimizer.step()

    grad_return_count = 0
    for key, (client, z_client, z_server) in client_paths.items():
        if z_server.grad is None:
            raise RuntimeError(f"Missing server gradient for feature_map key {key}, client {client.client_id}")
        client.backward_from_server(z_client, z_server.grad.detach())
        grad_return_count += 1

    return {
        "loss": float(loss.item()),
        "loss_cls": float(loss_cls.item()),
        "loss_align": float(loss_align.item()),
        "lambda_align": float(lambda_align),
        "accuracy": float(correct / max(1, total)),
        "feature_map": feature_map,
        "grad_return_count": int(grad_return_count),
        "K_t": int(grad_return_count),
        "Q_star": int(len(selected)),
    }


def _cluster_to_test_modality(cfg, cluster_ids):
    cluster_metrics_path = cfg.get("_cluster_metrics_path")
    with open(cluster_metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    if str(cfg.get("training", {}).get("cluster_assignment", "predicted")).lower() == "oracle":
        return {int(cluster_id): int(cluster_id) for cluster_id in cluster_ids}
    return {int(k): int(v) for k, v in metrics.get("cluster_to_true_modality_majority", {}).items()}


def _prediction_metrics(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_total = cm.sum(axis=1)
    per_class_correct = cm.diagonal()
    per_class_accuracy = {
        str(label): (float(per_class_correct[idx] / per_class_total[idx]) if per_class_total[idx] > 0 else 0.0)
        for idx, label in enumerate(labels)
    }
    return {
        "accuracy": float(sum(int(a == b) for a, b in zip(y_true, y_pred)) / max(1, len(y_true))),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "per_class_accuracy": per_class_accuracy,
    }


def _fuse_logits(logits_list, strategy):
    stacked = torch.stack(logits_list, dim=0)
    strategy = str(strategy).lower()
    if strategy == "mean":
        return stacked.mean(dim=0)
    if strategy in {"confidence_weighted", "entropy_weighted"}:
        probs = torch.softmax(stacked, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
        max_entropy = torch.log(torch.tensor(float(stacked.shape[-1]), dtype=stacked.dtype, device=stacked.device))
        confidence = 1.0 - entropy / max_entropy.clamp_min(1e-12)
        weights = confidence / confidence.sum(dim=0, keepdim=True).clamp_min(1e-12)
        return (weights.unsqueeze(-1) * stacked).sum(dim=0)
    raise ValueError("evaluation.fusion_strategy must be 'mean' or 'confidence_weighted'.")


def _evaluate(server, cluster_ids, cluster_to_clients, cfg, test_payload, device):
    cluster_to_true = _cluster_to_test_modality(cfg, cluster_ids)
    modality_names = list(test_payload.get("modality_names", []))
    if not modality_names:
        modality_names = list(test_payload.get("modalities", {}).keys())
    if not modality_names:
        raise RuntimeError("test_multimodal.pt must contain modality_names or a modalities mapping.")
    modality_tensors = [test_payload["modalities"][name] for name in modality_names]
    modality_dims = {idx: int(x.reshape(int(test_payload["label"].shape[0]), -1).shape[1]) for idx, x in enumerate(modality_tensors)}

    encoders = {}
    valid_cluster_ids = []
    for cid in cluster_ids:
        true_modality = int(cluster_to_true.get(cid, cid))
        if true_modality not in modality_dims:
            continue
        expected_dim = modality_dims[true_modality]
        candidates = [client for client in cluster_to_clients[int(cid)] if int(client.input_dim) == expected_dim]
        if not candidates:
            continue
        encoders[int(cid)] = candidates[0].encoder
        valid_cluster_ids.append(int(cid))
    if not valid_cluster_ids:
        raise RuntimeError("No predicted clusters can be mapped to paired test modalities for evaluation.")

    batch_size = int(cfg.get("training", {}).get("eval_batch_size", cfg.get("training", {}).get("batch_size", 64)))
    label = test_payload["label"]
    loader = DataLoader(TensorDataset(*modality_tensors, label), batch_size=batch_size, shuffle=False)
    labels = list(range(int(cfg.get("num_classes", 6))))
    fusion_cfg = cfg.get("evaluation", {})
    requested = set(fusion_cfg.get("fusion_modes", ["single", "pairwise", "all"]))
    fusion_strategy = str(fusion_cfg.get("fusion_strategy", "mean")).lower()
    all_outputs = {"single": {}, "pairwise": {}, "all": None}

    server.eval()
    for encoder in encoders.values():
        encoder.eval()
    with torch.no_grad():
        logits_by_cluster = {cid: [] for cid in valid_cluster_ids}
        y_true = []
        total_loss_by_cluster = {cid: 0.0 for cid in valid_cluster_ids}
        ce = nn.CrossEntropyLoss(reduction="sum")
        for batch in loader:
            *modality_batches, y_b = batch
            modality_batches = [x.to(device) for x in modality_batches]
            y_b = y_b.to(device)
            y_true.extend(y_b.detach().cpu().tolist())
            for cid in valid_cluster_ids:
                true_modality = int(cluster_to_true.get(cid, cid))
                xb = modality_batches[true_modality]
                activation = encoders[cid](xb)
                logits, _ = server.forward_cluster(cid, activation)
                logits_by_cluster[cid].append(logits.detach().cpu())
                total_loss_by_cluster[cid] += float(ce(logits, y_b).item())

    logits_by_cluster = {cid: torch.cat(parts, dim=0) for cid, parts in logits_by_cluster.items()}
    y_true_tensor = torch.tensor(y_true, dtype=torch.long)
    ce_cpu = nn.CrossEntropyLoss(reduction="sum")
    if "single" in requested:
        for cid in valid_cluster_ids:
            pred = logits_by_cluster[cid].argmax(dim=1).tolist()
            metric = _prediction_metrics(y_true, pred, labels)
            metric["loss"] = float(total_loss_by_cluster[cid] / max(1, len(y_true)))
            all_outputs["single"][str(cid)] = metric
    if "pairwise" in requested and len(valid_cluster_ids) >= 2:
        for a, b in itertools.combinations(valid_cluster_ids, 2):
            logits = _fuse_logits([logits_by_cluster[a], logits_by_cluster[b]], fusion_strategy)
            metric = _prediction_metrics(y_true, logits.argmax(dim=1).tolist(), labels)
            metric["loss"] = float(ce_cpu(logits, y_true_tensor).item() / max(1, len(y_true)))
            all_outputs["pairwise"][f"{a},{b}"] = metric
    if "all" in requested:
        logits = _fuse_logits([logits_by_cluster[cid] for cid in valid_cluster_ids], fusion_strategy)
        metric = _prediction_metrics(y_true, logits.argmax(dim=1).tolist(), labels)
        metric["loss"] = float(ce_cpu(logits, y_true_tensor).item() / max(1, len(y_true)))
        all_outputs["all"] = metric
    if all_outputs["all"] is None:
        cid = valid_cluster_ids[0]
        all_outputs["all"] = all_outputs["single"].get(str(cid))
        if all_outputs["all"] is None:
            metric = _prediction_metrics(y_true, logits_by_cluster[cid].argmax(dim=1).tolist(), labels)
            metric["loss"] = float(total_loss_by_cluster[cid] / max(1, len(y_true)))
            all_outputs["all"] = metric

    server.train()
    for encoder in encoders.values():
        encoder.train()
    primary = dict(all_outputs["all"])
    primary["single"] = all_outputs["single"]
    primary["pairwise"] = all_outputs["pairwise"]
    primary["evaluated_cluster_ids"] = valid_cluster_ids
    primary["fusion_strategy"] = fusion_strategy
    return primary


def run_unpaired_stage3_split_training(cfg: dict, project_root: Path, device: torch.device):
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
    server = UnpairedSharedSemanticServer(cluster_ids, hidden_dim, num_classes, server_cfg).to(device)
    server_optimizer = torch.optim.Adam(
        server.parameters(),
        lr=float(cfg.get("training", {}).get("server_lr", cfg.get("learning_rate", 1e-3))),
    )
    rng = random.Random(int(cfg.get("seed", 42)))
    scheduler = ClusterCyclicScheduler(cluster_to_clients, r, rng)

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
            selected = scheduler.sample_round()
            train_metrics = _train_round(server, server_optimizer, selected, cfg, device)
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
                        "loss": final_eval.get("loss", 0.0),
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
        "multimodal_mode": "unpaired_shared_semantic",
    }
    with (result_dir / "final_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    with (result_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)
    with (result_dir / "config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with (model_dir / "best_model_info.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_metrics": best_metrics,
                "cluster_ids": cluster_ids,
                "r": r,
                "multimodal_mode": "unpaired_shared_semantic",
            },
            f,
            indent=2,
        )

    return final_metrics
