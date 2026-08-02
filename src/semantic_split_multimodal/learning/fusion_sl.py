import csv
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from semantic_split_multimodal.learning.binding import ClientActivationBatch, build_label_random_pseudo_batch, common_labels_for_clusters
from semantic_split_multimodal.data.partitioner import resolve_project_path
from semantic_split_multimodal.evaluation.fusion_eval import evaluate_naturally_paired_fusion
from semantic_split_multimodal.evaluation.oracle_mapping import build_oracle_eval_mapping
from semantic_split_multimodal.learning.models import create_client_encoder
from semantic_split_multimodal.learning.models import ConcatMLPFusionServer
from semantic_split_multimodal.learning.scheduling import build_scheduler


class FusionSplitClient:
    def __init__(self, payload: dict, pred_cluster: int, encoder_state: dict, cfg: dict, device):
        self.client_id = str(payload["client_id"])
        self.pred_cluster = int(pred_cluster)
        self.device = device
        self.samples = payload["samples"].to(device)
        self.labels = payload["labels"].to(device)
        sequence_lengths = payload.get("sequence_lengths")
        self.sequence_lengths = None if sequence_lengths is None else sequence_lengths.to(device)
        self.batch_size = int(cfg.get("training", {}).get("batch_size", cfg.get("batch_size", 32)))
        input_shape = [int(v) for v in payload.get("input_shape", list(payload["samples"].shape[1:]))]
        encoder_type = str(payload.get("encoder_type", "time_series"))
        self.encoder = create_client_encoder(cfg, input_shape=input_shape, encoder_type=encoder_type).to(device)
        self.encoder.load_state_dict(encoder_state)
        self.optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=float(cfg.get("training", {}).get("client_lr", cfg.get("learning_rate", 1e-3))),
        )

    def sample_batch(self):
        n = int(self.samples.shape[0])
        replace = n < self.batch_size
        idx = torch.randint(0, n, (self.batch_size,), device=self.device) if replace else torch.randperm(n, device=self.device)[: self.batch_size]
        lengths = None if self.sequence_lengths is None else self.sequence_lengths[idx]
        return self.samples[idx], self.labels[idx], lengths

    def forward_detached(self, x, lengths=None):
        z_client = self.encoder(x, lengths)
        z_server = z_client.detach().requires_grad_(True)
        return z_client, z_server

    def backward_from_server(self, z_client, grad):
        self.optimizer.zero_grad()
        z_client.backward(grad)
        self.optimizer.step()


def _cluster_assignment_spec(cfg: dict, cluster_dir: Path):
    source = str(cfg.get("training", {}).get("cluster_assignment_source", "pred_cluster")).strip().lower()
    if source not in {"pred_cluster", "true_cluster"}:
        raise ValueError(
            "training.cluster_assignment_source must be 'pred_cluster' or 'true_cluster', "
            f"got {source!r}."
        )
    return source, cluster_dir / f"{source}.csv", source


def _read_assignments(path: Path, assignment_column: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing cluster assignments: {path}. Run Stage 2 first.")
    rows = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["client_id"]] = int(row[assignment_column])
    return rows


def _load_clients(cfg, partition_dir: Path, cluster_dir: Path, device):
    _, assignment_path, assignment_column = _cluster_assignment_spec(cfg, cluster_dir)
    assignments = _read_assignments(assignment_path, assignment_column)
    encoder_dir = cluster_dir / "pretrained_encoders"
    clients = []
    for path in sorted((partition_dir / "train_clients").glob("client_*.pt")):
        payload = torch.load(path, map_location="cpu")
        client_id = str(payload["client_id"])
        enc_payload = torch.load(encoder_dir / f"{client_id}_encoder.pt", map_location="cpu")
        clients.append(FusionSplitClient(payload, assignments[path.stem], enc_payload["state_dict"], cfg, device))
    if not clients:
        raise FileNotFoundError(f"No client_*.pt files found under {partition_dir / 'train_clients'}.")
    return clients


def _collect_selected_activations(selected):
    activation_batches = []
    client_paths = {}
    for client in selected:
        sampled = client.sample_batch()
        if len(sampled) == 2:
            x, y = sampled
            lengths = None
        else:
            x, y, lengths = sampled
        if lengths is None:
            z_client, z_server = client.forward_detached(x)
        else:
            z_client, z_server = client.forward_detached(x, lengths)
        activation_batches.append(
            ClientActivationBatch(
                client_id=client.client_id,
                pred_cluster=client.pred_cluster,
                activations=z_server,
                labels=y,
            )
        )
        client_paths.setdefault(int(client.pred_cluster), {}).setdefault(client.client_id, []).append((client, z_client, z_server))
    return activation_batches, client_paths


def _backward_to_clients(pseudo, client_paths):
    used = set()
    for cluster_id in pseudo.cluster_ids:
        source_ids = pseudo.source_client_ids[cluster_id]
        for client_id in source_ids:
            used.add((int(cluster_id), client_id))

    for cluster_id, clients_by_id in client_paths.items():
        for client_id, paths in clients_by_id.items():
            if (int(cluster_id), client_id) not in used:
                continue
            for client, z_client, z_server in paths:
                if z_server.grad is None:
                    raise RuntimeError(f"Missing server gradient for client {client.client_id}")
                client.backward_from_server(z_client, z_server.grad.detach())


def _param_l1_norm(module):
    total = 0.0
    with torch.no_grad():
        for param in module.parameters():
            total += float(param.detach().abs().sum().item())
    return total


def _client_param_l1_norm(clients):
    return sum(_param_l1_norm(client.encoder) for client in clients)


def _fusion_training_spec(cfg):
    fusion_cfg = cfg.get("fusion", {})
    objective = str(fusion_cfg.get("training_objective", "label_random_ce")).strip().lower()
    supported = {"label_random_ce", "mmbind_weighted_contrastive"}
    if objective not in supported:
        raise ValueError(
            "fusion.training_objective must be one of "
            f"{sorted(supported)}, got {objective!r}."
        )

    mmbind_cfg = fusion_cfg.get("mmbind", {})
    spec = {
        "objective": objective,
        "temperature": float(mmbind_cfg.get("temperature", 0.1)),
        "contrastive_weight": float(mmbind_cfg.get("contrastive_weight", 0.1)),
        "heterogeneous_ce_weight": float(mmbind_cfg.get("heterogeneous_ce_weight", 0.5)),
    }
    if objective == "mmbind_weighted_contrastive":
        if spec["temperature"] <= 0.0:
            raise ValueError("fusion.mmbind.temperature must be positive.")
        if spec["contrastive_weight"] < 0.0:
            raise ValueError("fusion.mmbind.contrastive_weight must be non-negative.")
        if spec["heterogeneous_ce_weight"] < 0.0:
            raise ValueError("fusion.mmbind.heterogeneous_ce_weight must be non-negative.")
    return spec


def _weighted_cross_cluster_contrastive_loss(adapted_slots, labels, confidences, temperature):
    """MMBind-style label-shared group contrastive loss across predicted clusters."""

    cluster_ids = sorted(int(cluster_id) for cluster_id in adapted_slots)
    if len(cluster_ids) < 2 or int(torch.unique(labels).numel()) < 2:
        return sum(value.sum() * 0.0 for value in adapted_slots.values())

    labels = labels.reshape(-1)
    confidences = confidences.reshape(-1).clamp_min(0.0)
    same_label = labels[:, None].eq(labels[None, :])
    pair_confidence = torch.sqrt(confidences[:, None] * confidences[None, :])
    positive_weights = same_label.to(pair_confidence.dtype) * pair_confidence
    losses = []

    def directional_loss(similarity, weights):
        log_prob = F.log_softmax(similarity / float(temperature), dim=1)
        positive_mass = weights.sum(dim=1)
        valid = positive_mass > 0
        if not bool(valid.any()):
            return similarity.sum() * 0.0
        per_anchor = -(weights * log_prob).sum(dim=1) / positive_mass.clamp_min(1.0e-12)
        return per_anchor[valid].mean()

    for left_pos, left_cluster in enumerate(cluster_ids):
        left = F.normalize(adapted_slots[left_cluster], dim=1)
        for right_cluster in cluster_ids[left_pos + 1 :]:
            right = F.normalize(adapted_slots[right_cluster], dim=1)
            similarity = left @ right.transpose(0, 1)
            losses.append(directional_loss(similarity, positive_weights))
            losses.append(directional_loss(similarity.transpose(0, 1), positive_weights.transpose(0, 1)))
    return torch.stack(losses).mean()


def _mmbind_fusion_losses(server, pseudo, spec):
    if not hasattr(server, "adapt_slots") or not hasattr(server, "classify_adapted"):
        raise TypeError(
            "mmbind_weighted_contrastive requires a fusion server with "
            "adapt_slots() and classify_adapted()."
        )

    adapted_slots = server.adapt_slots(pseudo.slot_activations)
    logits, _ = server.classify_adapted(adapted_slots)
    labels = pseudo.labels.to(logits.device)
    classification_loss = nn.CrossEntropyLoss()(logits, labels)
    contrastive_loss = _weighted_cross_cluster_contrastive_loss(
        adapted_slots,
        labels,
        pseudo.binding_confidences.to(logits.device),
        spec["temperature"],
    )

    heterogeneous_losses = []
    for retained_cluster in sorted(adapted_slots):
        heterogeneous_slots = {
            cluster_id: (
                value if cluster_id == retained_cluster else torch.zeros_like(value)
            )
            for cluster_id, value in adapted_slots.items()
        }
        heterogeneous_logits, _ = server.classify_adapted(heterogeneous_slots)
        heterogeneous_losses.append(nn.CrossEntropyLoss()(heterogeneous_logits, labels))
    heterogeneous_loss = torch.stack(heterogeneous_losses).mean()
    loss = (
        classification_loss
        + spec["contrastive_weight"] * contrastive_loss
        + spec["heterogeneous_ce_weight"] * heterogeneous_loss
    )
    return loss, logits, classification_loss, contrastive_loss, heterogeneous_loss


def _train_local_step(server, server_optimizer, selected, required_clusters, cfg):
    expected_clusters = sorted(int(cluster_id) for cluster_id in required_clusters)
    selected_clusters = sorted({int(client.pred_cluster) for client in selected})
    activation_batches, client_paths = _collect_selected_activations(selected)
    binding_cfg = cfg.get("binding", {})
    pseudo_batch_size = int(binding_cfg.get("batch_size", cfg.get("training", {}).get("batch_size", 32)))
    common_labels = common_labels_for_clusters(activation_batches, expected_clusters)
    fusion_training = _fusion_training_spec(cfg)
    pseudo = build_label_random_pseudo_batch(
        activation_batches,
        required_clusters=expected_clusters,
        batch_size=pseudo_batch_size,
    )
    if pseudo is None:
        return {
            "loss": 0.0,
            "classification_loss": 0.0,
            "contrastive_loss": 0.0,
            "heterogeneous_loss": 0.0,
            "accuracy": 0.0,
            "pseudo_batch_size": 0,
            "common_labels": common_labels,
            "binding_success": 0.0,
            "empty_binding_local_step": 1,
            "cluster_slot_coverage": float(len(selected_clusters) / max(1, len(expected_clusters))),
            "server_update_l1": 0.0,
            "client_update_l1": 0.0,
            "fusion_training_objective": fusion_training["objective"],
            "binding_confidence_mean": 0.0,
        }

    server_norm_before = _param_l1_norm(server)
    bound_clients = sorted(
        {
            client
            for cluster_id in pseudo.cluster_ids
            for client in pseudo.source_client_ids[cluster_id]
        }
    )
    clients_by_id = {
        client.client_id: client
        for cluster_paths in client_paths.values()
        for client_paths_for_id in cluster_paths.values()
        for client, _, _ in client_paths_for_id
    }
    client_norm_before = _client_param_l1_norm([clients_by_id[client_id] for client_id in bound_clients])
    server_optimizer.zero_grad()
    if fusion_training["objective"] == "label_random_ce":
        logits, _ = server(pseudo.slot_activations)
        classification_loss = nn.CrossEntropyLoss()(logits, pseudo.labels.to(logits.device))
        contrastive_loss = classification_loss.detach() * 0.0
        heterogeneous_loss = classification_loss.detach() * 0.0
        loss = classification_loss
    else:
        loss, logits, classification_loss, contrastive_loss, heterogeneous_loss = _mmbind_fusion_losses(
            server,
            pseudo,
            fusion_training,
        )
    loss.backward()
    server_optimizer.step()
    _backward_to_clients(pseudo, client_paths)
    server_update_l1 = abs(_param_l1_norm(server) - server_norm_before)
    client_update_l1 = abs(_client_param_l1_norm([clients_by_id[client_id] for client_id in bound_clients]) - client_norm_before)

    correct = int((logits.argmax(dim=1) == pseudo.labels.to(logits.device)).sum().item())
    total = int(pseudo.labels.numel())
    return {
        "loss": float(loss.item()),
        "classification_loss": float(classification_loss.item()),
        "contrastive_loss": float(contrastive_loss.item()),
        "heterogeneous_loss": float(heterogeneous_loss.item()),
        "accuracy": float(correct / max(1, total)),
        "pseudo_batch_size": int(total),
        "common_labels": common_labels,
        "binding_success": 1.0,
        "empty_binding_local_step": 0,
        "cluster_slot_coverage": float(len(pseudo.cluster_ids) / max(1, len(expected_clusters))),
        "server_update_l1": float(server_update_l1),
        "client_update_l1": float(client_update_l1),
        "fusion_training_objective": fusion_training["objective"],
        "binding_confidence_mean": float(pseudo.binding_confidences.mean().item()),
    }


def _train_round(server, server_optimizer, selected, required_clusters, cfg):
    expected_clusters = sorted(int(cluster_id) for cluster_id in required_clusters)
    selected_clusters = sorted({int(client.pred_cluster) for client in selected})
    missing_clusters = sorted(set(expected_clusters) - set(selected_clusters))
    if missing_clusters:
        raise RuntimeError(
            "Scheduler failed to cover all predicted clusters: "
            f"expected={expected_clusters}, selected={selected_clusters}, missing={missing_clusters}"
        )

    configured_local_steps = int(cfg.get("training", {}).get("local_steps", cfg.get("local_steps", 1)))
    if configured_local_steps <= 0:
        raise ValueError("training.local_steps must be positive.")

    losses = []
    classification_losses = []
    contrastive_losses = []
    heterogeneous_losses = []
    binding_confidences = []
    correct_weighted_sum = 0.0
    total_pseudo_samples = 0
    effective_local_steps = 0
    empty_binding_local_steps = 0
    server_update_l1 = 0.0
    client_update_l1 = 0.0
    common_labels_by_step = []
    common_labels_seen = set()
    cluster_slot_coverages = []

    for _ in range(configured_local_steps):
        local_metrics = _train_local_step(server, server_optimizer, selected, expected_clusters, cfg)
        common_labels_by_step.append(local_metrics["common_labels"])
        common_labels_seen.update(int(label) for label in local_metrics["common_labels"])
        cluster_slot_coverages.append(float(local_metrics["cluster_slot_coverage"]))
        if int(local_metrics["empty_binding_local_step"]):
            empty_binding_local_steps += 1
            continue

        effective_local_steps += 1
        pseudo_batch_size = int(local_metrics["pseudo_batch_size"])
        total_pseudo_samples += pseudo_batch_size
        losses.append(float(local_metrics["loss"]))
        classification_losses.append(float(local_metrics["classification_loss"]))
        contrastive_losses.append(float(local_metrics["contrastive_loss"]))
        heterogeneous_losses.append(float(local_metrics["heterogeneous_loss"]))
        binding_confidences.append(float(local_metrics["binding_confidence_mean"]))
        correct_weighted_sum += float(local_metrics["accuracy"]) * pseudo_batch_size
        server_update_l1 += float(local_metrics["server_update_l1"])
        client_update_l1 += float(local_metrics["client_update_l1"])

    attempted_local_steps = configured_local_steps
    empty_binding_round = int(effective_local_steps == 0)
    mean_loss = float(sum(losses) / max(1, len(losses))) if losses else 0.0
    mean_pseudo_batch_size = float(total_pseudo_samples / max(1, effective_local_steps)) if effective_local_steps else 0.0
    accuracy = float(correct_weighted_sum / max(1, total_pseudo_samples)) if total_pseudo_samples else 0.0
    binding_success_rate = float(effective_local_steps / max(1, attempted_local_steps))
    round_status = "empty_binding_round" if empty_binding_round else "effective"

    return {
        "loss": mean_loss,
        "mean_loss": mean_loss,
        "classification_loss": float(sum(classification_losses) / max(1, len(classification_losses))) if classification_losses else 0.0,
        "contrastive_loss": float(sum(contrastive_losses) / max(1, len(contrastive_losses))) if contrastive_losses else 0.0,
        "heterogeneous_loss": float(sum(heterogeneous_losses) / max(1, len(heterogeneous_losses))) if heterogeneous_losses else 0.0,
        "accuracy": accuracy,
        "K_t": int(len(selected)),
        "pseudo_batch_size": int(total_pseudo_samples),
        "total_pseudo_samples": int(total_pseudo_samples),
        "mean_pseudo_batch_size": mean_pseudo_batch_size,
        "common_labels": sorted(common_labels_seen),
        "local_step_common_labels": common_labels_by_step,
        "binding_success": float(effective_local_steps > 0),
        "empty_binding_round": empty_binding_round,
        "round_status": round_status,
        "configured_local_steps": int(configured_local_steps),
        "attempted_local_steps": int(attempted_local_steps),
        "effective_local_steps": int(effective_local_steps),
        "empty_binding_local_steps": int(empty_binding_local_steps),
        "round_binding_success_rate": binding_success_rate,
        "cluster_slot_coverage": float(sum(cluster_slot_coverages) / max(1, len(cluster_slot_coverages))),
        "server_update_l1": float(server_update_l1),
        "client_update_l1": float(client_update_l1),
        "fusion_training_objective": _fusion_training_spec(cfg)["objective"],
        "binding_confidence_mean": float(sum(binding_confidences) / max(1, len(binding_confidences))) if binding_confidences else 0.0,
    }


def run_mmbind_fusion_stage3_split_training(cfg: dict, project_root: Path, device: torch.device):
    partition_dir = resolve_project_path(project_root, cfg.get("partition", {}).get("output_dir", "local/results/data_partition"))
    cluster_dir = resolve_project_path(project_root, cfg.get("cluster", {}).get("output_dir", "local/results/cluster"))
    result_dir = resolve_project_path(project_root, cfg.get("result", {}).get("output_dir", "local/logs"))
    model_dir = resolve_project_path(project_root, cfg.get("result_model", {}).get("output_dir", "local/checkpoints"))
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    cluster_assignment_source, cluster_assignment_path, cluster_assignment_column = _cluster_assignment_spec(
        cfg, cluster_dir
    )
    clients = _load_clients(cfg, partition_dir, cluster_dir, device)
    clients_by_id = {client.client_id: client for client in clients}
    cluster_ids = sorted({int(client.pred_cluster) for client in clients})
    cluster_to_slot = {int(cluster_id): int(slot) for slot, cluster_id in enumerate(cluster_ids)}
    training_cfg = cfg.get("training", {})
    clients_per_cluster = training_cfg.get("clients_per_cluster_per_round")
    if clients_per_cluster is None:
        raise ValueError("training.clients_per_cluster_per_round is required for balanced cluster scheduling.")
    clients_per_cluster = int(clients_per_cluster)
    clients_per_round = clients_per_cluster * len(cluster_ids)
    scheduler = build_scheduler(
        training_cfg.get("scheduler", "balanced_cluster_round_robin"),
        clients,
        clients_per_cluster_per_round=clients_per_cluster,
        seed=int(cfg.get("seed", 42)),
    )
    server = ConcatMLPFusionServer(
        cluster_ids,
        int(cfg.get("encoder_hidden_dim", 128)),
        int(cfg.get("num_classes", 6)),
        cfg,
        cluster_to_slot=cluster_to_slot,
    ).to(device)
    server_optimizer = torch.optim.Adam(
        server.parameters(),
        lr=float(cfg.get("training", {}).get("server_lr", cfg.get("learning_rate", 1e-3))),
    )
    rounds = int(cfg.get("training", {}).get("global_rounds", cfg.get("global_rounds", 3)))
    validation_every = int(training_cfg.get("validation_every", 10))
    early_stopping_cfg = training_cfg.get("early_stopping", {})
    patience = int(early_stopping_cfg.get("patience", 3))
    min_rounds = int(early_stopping_cfg.get("min_rounds", 50))
    min_delta = float(early_stopping_cfg.get("min_delta", 0.001))
    if rounds <= 0:
        raise ValueError("training.global_rounds must be positive.")
    if validation_every <= 0:
        raise ValueError("training.validation_every must be positive.")
    if patience <= 0:
        raise ValueError("training.early_stopping.patience must be positive.")
    if min_rounds <= 0 or min_rounds > rounds:
        raise ValueError("training.early_stopping.min_rounds must be in [1, training.global_rounds].")
    if min_delta < 0.0:
        raise ValueError("training.early_stopping.min_delta must be non-negative.")

    train_fields = [
        "global_round",
        "round",
        "loss",
        "mean_loss",
        "classification_loss",
        "contrastive_loss",
        "heterogeneous_loss",
        "accuracy",
        "K_t",
        "pseudo_batch_size",
        "total_pseudo_samples",
        "mean_pseudo_batch_size",
        "common_labels_json",
        "local_step_common_labels_json",
        "binding_success",
        "empty_binding_round",
        "empty_binding_rounds",
        "binding_success_rate",
        "configured_local_steps",
        "attempted_local_steps",
        "effective_local_steps",
        "empty_binding_local_steps",
        "round_binding_success_rate",
        "coverage",
        "cluster_slot_coverage",
        "participation_fairness",
        "latency",
        "server_update_l1",
        "client_update_l1",
        "fusion_training_objective",
        "binding_confidence_mean",
        "selected_client_ids",
        "selected_cluster_ids",
        "expected_cluster_ids",
        "clients_per_cluster_per_round",
        "per_cluster_selected_json",
        "round_status",
    ]
    validation_fields = [
        "round",
        "eval_status",
        "eval_failure_reason",
        "loss",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "is_best",
        "checks_without_improvement",
    ]
    best_metrics = None
    test_eval = None
    oracle_mapping = None
    checks_without_improvement = 0
    executed_rounds = 0
    stop_reason = "max_global_rounds"
    empty_binding_rounds = 0
    successful_binding_rounds = 0
    total_attempted_local_steps = 0
    total_effective_local_steps = 0
    total_empty_binding_local_steps = 0
    with (result_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as train_f, (
        result_dir / "validation_log.csv"
    ).open("w", newline="", encoding="utf-8") as validation_f:
        train_writer = csv.DictWriter(train_f, fieldnames=train_fields)
        validation_writer = csv.DictWriter(validation_f, fieldnames=validation_fields)
        train_writer.writeheader()
        validation_writer.writeheader()
        for round_idx in range(1, rounds + 1):
            executed_rounds = round_idx
            selected = scheduler.sample_round()
            start = time.perf_counter()
            train_metrics = _train_round(server, server_optimizer, selected, cluster_ids, cfg)
            latency = time.perf_counter() - start
            sched_metrics = scheduler.metrics(selected)
            empty_binding_rounds += int(train_metrics["empty_binding_round"])
            successful_binding_rounds += int(train_metrics["binding_success"] > 0.0)
            total_attempted_local_steps += int(train_metrics["attempted_local_steps"])
            total_effective_local_steps += int(train_metrics["effective_local_steps"])
            total_empty_binding_local_steps += int(train_metrics["empty_binding_local_steps"])
            binding_success_rate = successful_binding_rounds / max(1, round_idx)
            train_writer.writerow(
                {
                    "global_round": round_idx,
                    "round": round_idx,
                    "loss": train_metrics["loss"],
                    "mean_loss": train_metrics["mean_loss"],
                    "classification_loss": train_metrics.get("classification_loss", 0.0),
                    "contrastive_loss": train_metrics.get("contrastive_loss", 0.0),
                    "heterogeneous_loss": train_metrics.get("heterogeneous_loss", 0.0),
                    "accuracy": train_metrics["accuracy"],
                    "K_t": train_metrics["K_t"],
                    "pseudo_batch_size": train_metrics["pseudo_batch_size"],
                    "total_pseudo_samples": train_metrics["total_pseudo_samples"],
                    "mean_pseudo_batch_size": train_metrics["mean_pseudo_batch_size"],
                    "common_labels_json": json.dumps(train_metrics["common_labels"]),
                    "local_step_common_labels_json": json.dumps(train_metrics["local_step_common_labels"]),
                    "binding_success": train_metrics["binding_success"],
                    "empty_binding_round": train_metrics["empty_binding_round"],
                    "empty_binding_rounds": empty_binding_rounds,
                    "binding_success_rate": float(binding_success_rate),
                    "configured_local_steps": train_metrics["configured_local_steps"],
                    "attempted_local_steps": train_metrics["attempted_local_steps"],
                    "effective_local_steps": train_metrics["effective_local_steps"],
                    "empty_binding_local_steps": train_metrics["empty_binding_local_steps"],
                    "round_binding_success_rate": train_metrics["round_binding_success_rate"],
                    "coverage": sched_metrics["coverage"],
                    "cluster_slot_coverage": train_metrics["cluster_slot_coverage"],
                    "participation_fairness": sched_metrics["participation_fairness"],
                    "latency": float(latency),
                    "server_update_l1": train_metrics["server_update_l1"],
                    "client_update_l1": train_metrics["client_update_l1"],
                    "fusion_training_objective": train_metrics.get(
                        "fusion_training_objective",
                        _fusion_training_spec(cfg)["objective"],
                    ),
                    "binding_confidence_mean": train_metrics.get("binding_confidence_mean", 0.0),
                    "selected_client_ids": json.dumps([c.client_id for c in selected]),
                    "selected_cluster_ids": json.dumps(sorted({int(c.pred_cluster) for c in selected})),
                    "expected_cluster_ids": json.dumps(cluster_ids),
                    "clients_per_cluster_per_round": sched_metrics["clients_per_cluster_per_round"],
                    "per_cluster_selected_json": json.dumps(sched_metrics["per_cluster_selected"]),
                    "round_status": train_metrics["round_status"],
                }
            )
            if round_idx % validation_every == 0 or round_idx == rounds:
                if oracle_mapping is None:
                    oracle_mapping = build_oracle_eval_mapping(
                        partition_dir / "client_meta.csv",
                        cluster_assignment_path,
                        None,
                        cluster_assignment_column,
                    )
                validation_eval = evaluate_naturally_paired_fusion(
                    server,
                    clients_by_id,
                    partition_dir / "validation_multimodal.pt",
                    oracle_mapping,
                    cfg,
                    device,
                )
                is_best = bool(
                    validation_eval["eval_status"] == "success"
                    and (
                        best_metrics is None
                        or float(validation_eval["weighted_f1"])
                        > float(best_metrics["weighted_f1"]) + min_delta
                    )
                )
                if is_best:
                    checks_without_improvement = 0
                    best_metrics = {
                        "best_round": round_idx,
                        "selected_by": "validation_weighted_f1",
                        "min_delta": min_delta,
                        **validation_eval,
                    }
                    _save_checkpoint(
                        model_dir / "best_model.pt",
                        server,
                        clients,
                        cfg,
                        cluster_ids,
                        cluster_to_slot,
                        best_metrics,
                    )
                else:
                    checks_without_improvement += 1
                validation_writer.writerow(
                    {
                        "round": round_idx,
                        "eval_status": validation_eval["eval_status"],
                        "eval_failure_reason": validation_eval["eval_failure_reason"],
                        "loss": validation_eval["loss"],
                        "accuracy": validation_eval["accuracy"],
                        "macro_f1": validation_eval["macro_f1"],
                        "weighted_f1": validation_eval["weighted_f1"],
                        "is_best": int(is_best),
                        "checks_without_improvement": checks_without_improvement,
                    }
                )
                train_f.flush()
                validation_f.flush()
                if validation_eval["eval_status"] != "success":
                    stop_reason = "validation_failed"
                    break
                if round_idx >= min_rounds and checks_without_improvement >= patience:
                    stop_reason = "early_stopping"
                    break

    last_metrics = {
        "checkpoint_role": "last_training_state",
        "stop_round": executed_rounds,
        "stop_reason": stop_reason,
    }
    _save_checkpoint(
        model_dir / "last_model.pt",
        server,
        clients,
        cfg,
        cluster_ids,
        cluster_to_slot,
        last_metrics,
    )

    test_evaluation_count = 0
    if best_metrics is not None:
        _load_checkpoint(
            model_dir / "best_model.pt",
            server,
            clients,
            cluster_ids,
            cluster_to_slot,
            device,
        )
        test_eval = evaluate_naturally_paired_fusion(
            server,
            clients_by_id,
            partition_dir / "test_multimodal.pt",
            oracle_mapping,
            cfg,
            device,
        )
        test_evaluation_count = 1
    else:
        test_eval = {
            "eval_status": "failed",
            "eval_failure_reason": "no_successful_validation_checkpoint",
            "loss": None,
            "accuracy": None,
            "macro_f1": None,
            "weighted_f1": None,
        }

    final_metrics = {
        "checkpoint": "best_model.pt",
        "selected_by": "validation_weighted_f1",
        "best_round": None if best_metrics is None else int(best_metrics["best_round"]),
        "test_eval_status": test_eval["eval_status"],
        "test_eval_failure_reason": test_eval["eval_failure_reason"],
        "test_loss": test_eval["loss"],
        "test_accuracy": test_eval["accuracy"],
        "test_macro_f1": test_eval["macro_f1"],
        "test_weighted_f1": test_eval["weighted_f1"],
        "test_num_eval_samples": test_eval.get("num_eval_samples"),
        "test_num_eval_batches": test_eval.get("num_eval_batches"),
        "oracle_eval_mapping": oracle_mapping,
        "cluster_ids": cluster_ids,
        "cluster_to_slot": cluster_to_slot,
        "estimated_num_clusters": len(cluster_ids),
        "cluster_assignment_source": cluster_assignment_source,
        "cluster_assignment_path": str(cluster_assignment_path),
        "clients_per_round": clients_per_round,
        "clients_per_cluster_per_round": clients_per_cluster,
        "configured_local_steps": int(cfg.get("training", {}).get("local_steps", cfg.get("local_steps", 1))),
        "configured_global_rounds": int(rounds),
        "executed_global_rounds": int(executed_rounds),
        "effective_global_rounds": int(successful_binding_rounds),
        "empty_binding_rounds": int(empty_binding_rounds),
        "total_attempted_local_steps": int(total_attempted_local_steps),
        "total_effective_local_steps": int(total_effective_local_steps),
        "total_empty_binding_local_steps": int(total_empty_binding_local_steps),
        "local_step_binding_success_rate": float(total_effective_local_steps / max(1, total_attempted_local_steps)),
        "binding_success_rate": float(successful_binding_rounds / max(1, executed_rounds)),
        "scheduler": training_cfg.get("scheduler", "balanced_cluster_round_robin"),
        "binding": str(cfg.get("binding", {}).get("type", "label_random")),
        "fusion": "concat_mlp",
        "fusion_training": _fusion_training_spec(cfg),
        "device": str(device),
        "validation_protocol": "naturally_paired_evaluation_only_oracle_mapping",
        "validation_every": validation_every,
        "early_stopping": {
            "patience": patience,
            "min_rounds": min_rounds,
            "min_delta": min_delta,
        },
        "stop_round": int(executed_rounds),
        "stop_reason": stop_reason,
        "test_evaluation_count": int(test_evaluation_count),
        "official_result": {
            "selection": "best_validation_weighted_f1",
            "metrics_file": "final_metrics.json",
            "model_file": "best_model.pt",
        },
    }
    with (result_dir / "final_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    with (result_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)
    return final_metrics


def _save_checkpoint(path: Path, server, clients, cfg, cluster_ids, cluster_to_slot, metrics):
    torch.save(
        {
            "server_state_dict": server.state_dict(),
            "client_encoder_states": {
                client.client_id: {
                    "pred_cluster": int(client.pred_cluster),
                    "state_dict": client.encoder.cpu().state_dict(),
                }
                for client in clients
            },
            "pred_cluster_assignments": {
                client.client_id: int(client.pred_cluster)
                for client in clients
            },
            "cluster_ids": [int(cluster_id) for cluster_id in cluster_ids],
            "cluster_to_slot": {
                int(cluster_id): int(slot)
                for cluster_id, slot in cluster_to_slot.items()
            },
            "resolved_config": cfg,
            "metrics": metrics,
        },
        path,
    )
    for client in clients:
        client.encoder.to(client.device)


def _load_checkpoint(path, server, clients, cluster_ids, cluster_to_slot, device):
    checkpoint = torch.load(path, map_location="cpu")
    saved_cluster_ids = [int(v) for v in checkpoint["cluster_ids"]]
    saved_cluster_to_slot = {int(k): int(v) for k, v in checkpoint["cluster_to_slot"].items()}
    if saved_cluster_ids != [int(v) for v in cluster_ids]:
        raise ValueError("Checkpoint cluster_ids do not match the current Stage3 cluster IDs.")
    if saved_cluster_to_slot != {int(k): int(v) for k, v in cluster_to_slot.items()}:
        raise ValueError("Checkpoint cluster_to_slot does not match the current Stage3 mapping.")
    server.load_state_dict(checkpoint["server_state_dict"])
    client_states = checkpoint["client_encoder_states"]
    current_ids = {client.client_id for client in clients}
    if set(client_states) != current_ids:
        raise ValueError("Checkpoint client encoder IDs do not match the current Stage3 clients.")
    for client in clients:
        saved = client_states[client.client_id]
        if int(saved["pred_cluster"]) != int(client.pred_cluster):
            raise ValueError(f"Checkpoint pred_cluster mismatch for client {client.client_id}.")
        client.encoder.load_state_dict(saved["state_dict"])
        client.encoder.to(device)
