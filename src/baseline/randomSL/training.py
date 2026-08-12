"""randomSL Stage 3 training entry.

randomSL keeps the entire MSL training protocol unchanged and replaces only
the cluster-aware scheduler with uniform random client sampling. Every other
component (client encoders, label-guided semantic pseudo binding,
``ClusterAdapter`` + concat fusion, split-learning forward/backward, naturally
paired evaluation, checkpointing) is reused from the MSL mainline package.

Private helpers from :mod:`MSL.learning.fusion_sl` are intentionally imported
so that the baseline behaves exactly like the mainline training step; keep
this module in sync if the mainline internals change.
"""

import csv
import json
import time
from pathlib import Path

import torch
from sklearn.metrics import f1_score

from baseline.randomSL.scheduling import RandomScheduler
from MSL.data.partitioner import resolve_project_path
from MSL.evaluation.fusion_eval import evaluate_naturally_paired_fusion
from MSL.evaluation.oracle_mapping import build_oracle_eval_mapping
from MSL.learning.fusion_sl import (
    _cluster_assignment_spec,
    _fusion_training_spec,
    _load_checkpoint,
    _load_clients,
    _save_checkpoint,
    _train_local_step,
    _training_class_weights,
)
from MSL.learning.models import ConcatMLPFusionServer


def _random_sl_round(server, server_optimizer, selected, required_clusters, cfg, class_weights=None):
    """One randomSL training round.

    Aggregation matches the mainline ``_train_round``, except that a round is
    allowed to proceed when the random selection does not cover every expected
    predicted cluster. In that case the label binder returns ``None``, the
    local step is recorded as an empty binding step, and the round is reported
    as an ``empty_binding_round`` instead of raising.
    """

    expected_clusters = sorted(int(cluster_id) for cluster_id in required_clusters)
    selected_clusters = sorted({int(client.pred_cluster) for client in selected})
    missing_clusters = sorted(set(expected_clusters) - set(selected_clusters))

    configured_local_steps = int(cfg.get("training", {}).get("local_steps", cfg.get("local_steps", 1)))
    if configured_local_steps <= 0:
        raise ValueError("training.local_steps must be positive.")

    losses = []
    classification_losses = []
    contrastive_losses = []
    heterogeneous_losses = []
    binding_confidences = []
    all_preds = []
    all_labels = []
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
        local_metrics = _train_local_step(
            server,
            server_optimizer,
            selected,
            expected_clusters,
            cfg,
            class_weights=class_weights,
        )
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
        all_preds.append(local_metrics["preds"])
        all_labels.append(local_metrics["labels"])
        correct_weighted_sum += float(local_metrics["accuracy"]) * pseudo_batch_size
        server_update_l1 += float(local_metrics["server_update_l1"])
        client_update_l1 += float(local_metrics["client_update_l1"])

    attempted_local_steps = configured_local_steps
    empty_binding_round = int(effective_local_steps == 0)
    mean_loss = float(sum(losses) / max(1, len(losses))) if losses else 0.0
    mean_pseudo_batch_size = float(total_pseudo_samples / max(1, effective_local_steps)) if effective_local_steps else 0.0
    accuracy = float(correct_weighted_sum / max(1, total_pseudo_samples)) if total_pseudo_samples else 0.0
    if all_preds:
        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(labels, preds, average="weighted", zero_division=0))
    else:
        macro_f1 = 0.0
        weighted_f1 = 0.0
    binding_success_rate = float(effective_local_steps / max(1, attempted_local_steps))
    round_status = "empty_binding_round" if empty_binding_round else "effective"
    if missing_clusters:
        empty_binding_reason = "missing_cluster"
    elif empty_binding_round:
        empty_binding_reason = "no_common_label"
    else:
        empty_binding_reason = None

    return {
        "loss": mean_loss,
        "mean_loss": mean_loss,
        "classification_loss": float(sum(classification_losses) / max(1, len(classification_losses))) if classification_losses else 0.0,
        "contrastive_loss": float(sum(contrastive_losses) / max(1, len(contrastive_losses))) if contrastive_losses else 0.0,
        "heterogeneous_loss": float(sum(heterogeneous_losses) / max(1, len(heterogeneous_losses))) if heterogeneous_losses else 0.0,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
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
        "missing_cluster_ids": missing_clusters,
        "empty_binding_reason": empty_binding_reason,
    }


def run_random_sl_stage3_split_training(cfg: dict, project_root: Path, device: torch.device):
    partition_dir = resolve_project_path(project_root, cfg.get("partition", {}).get("output_dir", "results/MSL/partition"))
    cluster_dir = resolve_project_path(project_root, cfg.get("cluster", {}).get("output_dir", "results/MSL/cluster"))
    result_dir = resolve_project_path(project_root, cfg.get("result", {}).get("output_dir", "local/logs"))
    model_dir = resolve_project_path(project_root, cfg.get("result_model", {}).get("output_dir", "local/checkpoints"))
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    cluster_assignment_source, cluster_assignment_path, cluster_assignment_column = _cluster_assignment_spec(
        cfg, cluster_dir
    )
    clients = _load_clients(cfg, partition_dir, cluster_dir, device)
    class_weights = _training_class_weights(clients, cfg, device)
    clients_by_id = {client.client_id: client for client in clients}
    cluster_ids = sorted({int(client.pred_cluster) for client in clients})
    cluster_to_slot = {int(cluster_id): int(slot) for slot, cluster_id in enumerate(cluster_ids)}
    training_cfg = cfg.get("training", {})
    clients_per_cluster = training_cfg.get("clients_per_cluster_per_round")
    if clients_per_cluster is None:
        raise ValueError(
            "training.clients_per_cluster_per_round is required to keep the per-round "
            "client budget identical to the mainline."
        )
    clients_per_cluster = int(clients_per_cluster)
    clients_per_round = clients_per_cluster * len(cluster_ids)
    scheduler = RandomScheduler(
        clients,
        clients_per_round=clients_per_round,
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
        weight_decay=float(cfg.get("training", {}).get("server_weight_decay", 0.0)),
    )
    rounds = int(cfg.get("training", {}).get("global_rounds", cfg.get("global_rounds", 3)))
    if rounds <= 0:
        raise ValueError("training.global_rounds must be positive.")

    train_fields = [
        "global_round",
        "round",
        "loss",
        "mean_loss",
        "classification_loss",
        "contrastive_loss",
        "heterogeneous_loss",
        "accuracy",
        "macro_f1",
        "weighted_f1",
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
        "clients_per_round",
        "clients_per_cluster_per_round",
        "per_cluster_selected_json",
        "round_status",
        "missing_cluster_ids_json",
        "empty_binding_reason",
    ]
    test_eval = None
    oracle_mapping = None
    executed_rounds = 0
    stop_reason = "max_global_rounds"
    empty_binding_rounds = 0
    missing_cluster_rounds = 0
    no_common_label_rounds = 0
    successful_binding_rounds = 0
    total_attempted_local_steps = 0
    total_effective_local_steps = 0
    total_empty_binding_local_steps = 0
    with (result_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as train_f:
        train_writer = csv.DictWriter(train_f, fieldnames=train_fields)
        train_writer.writeheader()
        for round_idx in range(1, rounds + 1):
            executed_rounds = round_idx
            selected = scheduler.sample_round()
            start = time.perf_counter()
            train_metrics = _random_sl_round(
                server,
                server_optimizer,
                selected,
                cluster_ids,
                cfg,
                class_weights=class_weights,
            )
            latency = time.perf_counter() - start
            sched_metrics = scheduler.metrics(selected)
            empty_binding_rounds += int(train_metrics["empty_binding_round"])
            missing_cluster_rounds += int(bool(train_metrics["missing_cluster_ids"]))
            no_common_label_rounds += int(train_metrics["empty_binding_reason"] == "no_common_label")
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
                    "macro_f1": train_metrics.get("macro_f1", 0.0),
                    "weighted_f1": train_metrics.get("weighted_f1", 0.0),
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
                    "clients_per_round": int(clients_per_round),
                    "clients_per_cluster_per_round": int(clients_per_cluster),
                    "per_cluster_selected_json": json.dumps(sched_metrics["per_cluster_selected"]),
                    "round_status": train_metrics["round_status"],
                    "missing_cluster_ids_json": json.dumps(train_metrics["missing_cluster_ids"]),
                    "empty_binding_reason": train_metrics["empty_binding_reason"],
                }
            )
            train_f.flush()

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

    run_test = bool(cfg.get("evaluation", {}).get("run_test", True))
    test_evaluation_count = 0
    if oracle_mapping is None:
        oracle_mapping = build_oracle_eval_mapping(
            partition_dir / "client_meta.csv",
            cluster_assignment_path,
            None,
            cluster_assignment_column,
        )
    _load_checkpoint(
        model_dir / "last_model.pt",
        server,
        clients,
        cluster_ids,
        cluster_to_slot,
        device,
    )
    if run_test:
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
            "eval_status": "deferred",
            "eval_failure_reason": None,
            "loss": None,
            "accuracy": None,
            "balanced_accuracy": None,
            "macro_recall": None,
            "macro_f1": None,
            "weighted_f1": None,
            "binary_f1": None,
            "confusion_matrix": None,
        }

    final_metrics = {
        "method": "random_sl",
        "baseline": "randomSL",
        "scheduler": "random",
        "training_mode": "random_sl_split_learning",
        "checkpoint": "last_model.pt",
        "selected_by": "fixed_rounds_no_validation",
        "best_round": int(executed_rounds),
        "test_eval_status": test_eval["eval_status"],
        "test_eval_failure_reason": test_eval["eval_failure_reason"],
        "test_loss": test_eval["loss"],
        "test_accuracy": test_eval["accuracy"],
        "test_balanced_accuracy": test_eval.get("balanced_accuracy"),
        "test_macro_recall": test_eval.get("macro_recall"),
        "test_macro_f1": test_eval["macro_f1"],
        "test_weighted_f1": test_eval["weighted_f1"],
        "test_binary_f1": test_eval.get("binary_f1"),
        "test_confusion_matrix": test_eval.get("confusion_matrix"),
        "test_per_class_precision": test_eval.get("per_class_precision"),
        "test_per_class_recall": test_eval.get("per_class_recall"),
        "test_per_class_f1": test_eval.get("per_class_f1"),
        "test_per_class_support": test_eval.get("per_class_support"),
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
        "missing_cluster_rounds": int(missing_cluster_rounds),
        "no_common_label_rounds": int(no_common_label_rounds),
        "total_attempted_local_steps": int(total_attempted_local_steps),
        "total_effective_local_steps": int(total_effective_local_steps),
        "total_empty_binding_local_steps": int(total_empty_binding_local_steps),
        "local_step_binding_success_rate": float(total_effective_local_steps / max(1, total_attempted_local_steps)),
        "binding_success_rate": float(successful_binding_rounds / max(1, executed_rounds)),
        "scheduler": training_cfg.get("scheduler", "random"),
        "binding": str(cfg.get("binding", {}).get("type", "label_random")),
        "fusion": "concat_mlp",
        "fusion_training": _fusion_training_spec(cfg),
        "class_weighting": training_cfg.get("class_weighting", "none"),
        "class_weights": None if class_weights is None else class_weights.detach().cpu().tolist(),
        "device": str(device),
        "stop_round": int(executed_rounds),
        "stop_reason": stop_reason,
        "test_evaluation_count": int(test_evaluation_count),
        "evaluation_mode": "formal_test" if run_test else "test_deferred",
        "official_result": (
            {
                "selection": "fixed_rounds_no_validation",
                "metrics_file": "final_metrics.json",
                "model_file": "last_model.pt",
            }
            if run_test
            else None
        ),
    }
    with (result_dir / "final_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    return final_metrics
