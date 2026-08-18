# 多模态评估、tolerant routing 和 discovery metrics 计算逻辑。
import csv
from dataclasses import dataclass
from pathlib import Path

import torch


SUCCESS = "success"


@dataclass
class EvaluationRouting:
    true_modalities: list[int]
    pred_clusters: list[int]
    n_mq: dict[int, dict[int, int]]
    p_mq: dict[int, dict[int, float]]
    activation_ensemble_groups: dict[tuple[int, int], list[object]]
    client_rows: list[dict]

    # 返回可写入 JSON 的 routing 元数据。
    def to_metadata(self) -> dict:
        return {
            "status": SUCCESS,
            "failure_reason": None,
            "mapping_type": "tolerant_evaluation_only",
            "evaluation_encoder_aggregation": "activation_mean",
            "uses_hidden_modality_id": True,
            "true_modalities": [int(value) for value in self.true_modalities],
            "pred_clusters": [int(value) for value in self.pred_clusters],
            "true_Q": int(len(self.true_modalities)),
            "estimated_Q": int(len(self.pred_clusters)),
            "N_mq": {
                str(modality_id): {str(cluster_id): int(count) for cluster_id, count in row.items()}
                for modality_id, row in self.n_mq.items()
            },
            "P_mq": {
                str(modality_id): {str(cluster_id): float(weight) for cluster_id, weight in row.items()}
                    for modality_id, row in self.p_mq.items()
            },
            "activation_ensemble_keys": [
                {
                    "true_modality": int(modality_id),
                    "pred_cluster": int(cluster_id),
                    "num_client_encoders": int(len(clients)),
                    "client_ids": [str(getattr(client, "client_id", "")) for client in clients],
                }
                for (modality_id, cluster_id), clients in sorted(self.activation_ensemble_groups.items())
            ],
            "client_rows": self.client_rows,
        }


# 读取 client preparation 保存的 client metadata。
def read_client_meta(path: Path) -> dict[str, dict]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing client metadata: {path}")
    rows = {}
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[str(row["client_id"])] = {
                "hidden_modality_id": int(row["hidden_modality_id"]),
                "hidden_modality_name": row.get("hidden_modality_name"),
                "encoder_type": row.get("encoder_type"),
            }
    return rows


# 读取 modality discovery 保存的预测簇或 oracle 簇 assignment。
def read_cluster_assignments(path: Path, assignment_column: str = "pred_cluster") -> dict[str, int]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing cluster assignments: {path}")
    rows = {}
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[str(row["client_id"])] = int(row[assignment_column])
    return rows


# 构建 true modality × predicted cluster 的计数矩阵。
def build_count_matrix(client_rows: list[dict]) -> tuple[list[int], list[int], dict[int, dict[int, int]]]:
    true_modalities = sorted({int(row["hidden_modality_id"]) for row in client_rows})
    pred_clusters = sorted({int(row["pred_cluster"]) for row in client_rows})
    n_mq = {modality_id: {cluster_id: 0 for cluster_id in pred_clusters} for modality_id in true_modalities}
    for row in client_rows:
        n_mq[int(row["hidden_modality_id"])][int(row["pred_cluster"])] += 1
    return true_modalities, pred_clusters, n_mq


# 按 predicted cluster 列归一化得到 P_mq。
def normalize_count_matrix(n_mq: dict[int, dict[int, int]], pred_clusters: list[int]) -> dict[int, dict[int, float]]:
    p_mq = {modality_id: {} for modality_id in n_mq}
    for cluster_id in pred_clusters:
        denominator = sum(int(row.get(cluster_id, 0)) for row in n_mq.values())
        if denominator <= 0:
            continue
        for modality_id, row in n_mq.items():
            p_mq[modality_id][cluster_id] = float(int(row.get(cluster_id, 0)) / denominator)
    return p_mq


# 校验 P_mq 的每个非空 predicted cluster 列归一化。
def validate_probability_matrix(p_mq: dict[int, dict[int, float]], pred_clusters: list[int]) -> None:
    for cluster_id in pred_clusters:
        column_sum = sum(float(row.get(cluster_id, 0.0)) for row in p_mq.values())
        if abs(column_sum - 1.0) > 1e-6:
            raise ValueError(f"P_mq column {cluster_id} is not normalized: sum={column_sum}")


# 为每个非空 (true modality, predicted cluster) 组合保存训练后的 client objects。
def build_activation_ensemble_groups(client_rows: list[dict], clients_by_id: dict[str, object]) -> dict[tuple[int, int], list[object]]:
    grouped: dict[tuple[int, int], list[object]] = {}
    for row in sorted(client_rows, key=lambda item: str(item["client_id"])):
        client_id = str(row["client_id"])
        if client_id not in clients_by_id:
            raise KeyError(f"Missing trained client object for {client_id}.")
        key = (int(row["hidden_modality_id"]), int(row["pred_cluster"]))
        grouped.setdefault(key, []).append(clients_by_id[client_id])

    for clients in grouped.values():
        for client in clients:
            client.encoder.eval()
    return grouped


# 从 metadata、assignment 和训练后 client 构建 tolerant evaluation routing。
def build_tolerant_evaluation_routing(
    client_meta_path: Path,
    cluster_assignments_path: Path,
    clients_by_id: dict[str, object],
    assignment_column: str = "pred_cluster",
) -> EvaluationRouting:
    client_meta = read_client_meta(Path(client_meta_path))
    assignments = read_cluster_assignments(Path(cluster_assignments_path), assignment_column)
    client_rows = []
    for client_id in sorted(client_meta):
        if client_id not in assignments:
            raise ValueError(f"Missing cluster assignment for client {client_id}.")
        client_rows.append(
            {
                "client_id": client_id,
                "hidden_modality_id": int(client_meta[client_id]["hidden_modality_id"]),
                "pred_cluster": int(assignments[client_id]),
                "hidden_modality_name": client_meta[client_id].get("hidden_modality_name"),
                "encoder_type": client_meta[client_id].get("encoder_type"),
            }
        )
    true_modalities, pred_clusters, n_mq = build_count_matrix(client_rows)
    p_mq = normalize_count_matrix(n_mq, pred_clusters)
    validate_probability_matrix(p_mq, pred_clusters)
    activation_ensemble_groups = build_activation_ensemble_groups(client_rows, clients_by_id)
    return EvaluationRouting(
        true_modalities=true_modalities,
        pred_clusters=pred_clusters,
        n_mq=n_mq,
        p_mq=p_mq,
        activation_ensemble_groups=activation_ensemble_groups,
        client_rows=client_rows,
    )


# 按 tolerant routing 公式为一个 batch 构造所有 predicted slot 表征。
def route_paired_batch(xs, batch_lengths, routing: EvaluationRouting, device) -> dict[int, torch.Tensor]:
    slot_activations = {}
    for cluster_id in routing.pred_clusters:
        weighted_parts = []
        for modality_id in routing.true_modalities:
            count = int(routing.n_mq[modality_id].get(cluster_id, 0))
            weight = float(routing.p_mq[modality_id].get(cluster_id, 0.0))
            if count <= 0 or weight <= 0.0:
                continue
            clients = routing.activation_ensemble_groups[(modality_id, cluster_id)]
            xb = xs[modality_id].to(device)
            lengths = batch_lengths[modality_id]
            if lengths is not None:
                lengths = lengths.to(device)
            activations = []
            for client in clients:
                encoder = client.encoder
                z = encoder(xb) if lengths is None else encoder(xb, lengths)
                activations.append(z)
            z = torch.stack(activations, dim=0).mean(dim=0)
            weighted_parts.append(z * weight)
        if not weighted_parts:
            raise ValueError(f"No activation ensemble available for predicted cluster {cluster_id}.")
        slot_activations[int(cluster_id)] = torch.stack(weighted_parts, dim=0).sum(dim=0)
    return slot_activations


import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
)


def discovery_metrics(true_modality, pred_cluster):
    true = np.asarray(true_modality, dtype=int)
    pred = np.asarray(pred_cluster, dtype=int)
    true_values = sorted(np.unique(true))
    pred_values = sorted(np.unique(pred))
    mapping = {}
    mapped = np.zeros_like(pred)
    confusion = np.zeros((len(pred_values), len(true_values)), dtype=int)
    true_index = {int(value): idx for idx, value in enumerate(true_values)}
    pred_index = {int(value): idx for idx, value in enumerate(pred_values)}
    for cluster_id in pred_values:
        members = true[pred == cluster_id]
        values, counts = np.unique(members, return_counts=True)
        majority = int(values[np.argmax(counts)])
        mapping[int(cluster_id)] = majority
        mapped[pred == cluster_id] = majority
        for value, count in zip(values, counts):
            confusion[pred_index[int(cluster_id)], true_index[int(value)]] = int(count)
    hungarian_acc = _hungarian_accuracy(confusion, len(true))
    pred_purity = _pred_cluster_purity(confusion, pred_values, true_values)
    true_splits = _true_modality_splits(true, pred, true_values)
    pred_mixes = _pred_cluster_mixes(true, pred, pred_values)
    estimated_q = int(len(pred_values))
    true_q = int(len(true_values))
    q_correct = estimated_q == true_q
    partition_correct = (
        q_correct
        and all(item["num_pred_clusters"] == 1 for item in true_splits.values())
        and all(item["num_true_modalities"] == 1 for item in pred_mixes.values())
        and abs(float(hungarian_acc) - 1.0) < 1e-12
    )
    if q_correct and partition_correct:
        status = "discovery_success"
    elif q_correct:
        status = "correct_q_incorrect_partition"
    elif all(item["num_true_modalities"] == 1 for item in pred_mixes.values()):
        status = "incorrect_q_over_or_under_clustering"
    else:
        status = "discovery_failure"
    return {
        "ARI": float(adjusted_rand_score(true, pred)),
        "NMI": float(normalized_mutual_info_score(true, pred)),
        "ACC": float(accuracy_score(true, mapped)),
        "ACC_mapping_type": "many_to_one_majority",
        "hungarian_ACC": float(hungarian_acc),
        "hungarian_ACC_mapping_type": "one_to_one",
        "true_Q": true_q,
        "estimated_Q": estimated_q,
        "abs_Q_error": int(abs(estimated_q - true_q)),
        "estimated_num_clusters": estimated_q,
        "cluster_to_true_modality_majority": {str(k): int(v) for k, v in mapping.items()},
        "pred_cluster_hidden_modality_confusion": {
            "pred_clusters": [int(v) for v in pred_values],
            "hidden_modalities": [int(v) for v in true_values],
            "matrix": confusion.astype(int).tolist(),
        },
        "pred_cluster_purity": pred_purity,
        "true_modality_pred_cluster_splits": true_splits,
        "pred_cluster_true_modality_mixes": pred_mixes,
        "discovery_status": status,
    }


def _hungarian_accuracy(confusion, total):
    if int(total) == 0:
        return 0.0
    if confusion.size == 0:
        return 0.0
    cost = -confusion
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(confusion[row_ind, col_ind].sum() / int(total))


def _pred_cluster_purity(confusion, pred_values, true_values):
    out = {}
    for row, cluster_id in enumerate(pred_values):
        total = int(confusion[row].sum())
        if total == 0:
            majority = None
            purity = 0.0
        else:
            col = int(np.argmax(confusion[row]))
            majority = int(true_values[col])
            purity = float(confusion[row, col] / total)
        out[str(int(cluster_id))] = {
            "size": total,
            "majority_hidden_modality": majority,
            "purity": purity,
        }
    return out


def _true_modality_splits(true, pred, true_values):
    out = {}
    for modality_id in true_values:
        clusters = sorted(np.unique(pred[true == modality_id]).astype(int).tolist())
        out[str(int(modality_id))] = {
            "num_pred_clusters": int(len(clusters)),
            "pred_clusters": clusters,
        }
    return out


def _pred_cluster_mixes(true, pred, pred_values):
    out = {}
    for cluster_id in pred_values:
        modalities = sorted(np.unique(true[pred == cluster_id]).astype(int).tolist())
        out[str(int(cluster_id))] = {
            "num_true_modalities": int(len(modalities)),
            "hidden_modalities": modalities,
        }
    return out


def learning_metrics(y_true, y_pred, modality_ids=None):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "macro_recall": float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if len(y_true) else 0.0,
        "weighted_f1": (
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)) if len(y_true) else 0.0
        ),
    }
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    if labels:
        precision, recall, per_class_f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
        out["class_labels"] = [int(label) for label in labels]
        out["per_class_precision"] = [float(value) for value in precision]
        out["per_class_recall"] = [float(value) for value in recall]
        out["per_class_f1"] = [float(value) for value in per_class_f1]
        out["per_class_support"] = [int(value) for value in support]
        out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist()
    else:
        out.update(
            {
                "class_labels": [],
                "per_class_precision": [],
                "per_class_recall": [],
                "per_class_f1": [],
                "per_class_support": [],
                "confusion_matrix": [],
            }
        )
    out["binary_f1"] = (
        float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
        if len(y_true) and set(y_true.tolist()).issubset({0, 1})
        else None
    )
    if modality_ids is not None:
        modality_ids = np.asarray(modality_ids, dtype=int)
        out["per_modality_accuracy"] = {}
        for modality_id in sorted(np.unique(modality_ids)):
            mask = modality_ids == modality_id
            out["per_modality_accuracy"][str(int(modality_id))] = (
                float(accuracy_score(y_true[mask], y_pred[mask])) if mask.any() else 0.0
            )
    return out


import json
from pathlib import Path



EVALUATION_MAPPING_FAILURE = "evaluation_mapping_failure"


# 构建 evaluation-only tolerant routing 元数据，兼容旧函数名。
def build_oracle_eval_mapping(
    client_meta_path: Path,
    cluster_assignments_path: Path,
    output_path: Path | None = None,
    assignment_column: str = "pred_cluster",
):
    client_meta = read_client_meta(Path(client_meta_path))
    assignments = read_cluster_assignments(Path(cluster_assignments_path), assignment_column)
    rows = []
    for client_id in sorted(client_meta):
        if client_id not in assignments:
            result = _mapping_failure(
                f"Missing {assignment_column} assignment for client {client_id}.",
                rows,
                output_path,
            )
            return result
        rows.append(
            {
                "client_id": client_id,
                "hidden_modality_id": int(client_meta[client_id]["hidden_modality_id"]),
                "pred_cluster": int(assignments[client_id]),
                "hidden_modality_name": client_meta[client_id].get("hidden_modality_name"),
                "encoder_type": client_meta[client_id].get("encoder_type"),
            }
        )

    true_modalities, pred_clusters, n_mq = build_count_matrix(rows)
    p_mq = normalize_count_matrix(n_mq, pred_clusters)
    validate_probability_matrix(p_mq, pred_clusters)
    modality_to_clusters = {
        modality_id: sorted(
            cluster_id for cluster_id in pred_clusters if int(n_mq[modality_id].get(cluster_id, 0)) > 0
        )
        for modality_id in true_modalities
    }
    cluster_to_modalities = {
        cluster_id: sorted(
            modality_id for modality_id in true_modalities if int(n_mq[modality_id].get(cluster_id, 0)) > 0
        )
        for cluster_id in pred_clusters
    }
    result = {
        "status": SUCCESS,
        "failure_reason": None,
        "mapping_type": "tolerant_evaluation_only",
        "uses_hidden_modality_id": True,
        "true_modalities": [int(value) for value in true_modalities],
        "pred_clusters": [int(value) for value in pred_clusters],
        "true_Q": int(len(true_modalities)),
        "estimated_Q": int(len(pred_clusters)),
        "N_mq": {
            str(modality_id): {str(cluster_id): int(count) for cluster_id, count in row.items()}
            for modality_id, row in n_mq.items()
        },
        "P_mq": {
            str(modality_id): {str(cluster_id): float(weight) for cluster_id, weight in row.items()}
            for modality_id, row in p_mq.items()
        },
        "modality_to_clusters": {
            str(modality_id): [int(cluster_id) for cluster_id in clusters]
            for modality_id, clusters in modality_to_clusters.items()
        },
        "cluster_to_modalities": {
            str(cluster_id): [int(modality_id) for modality_id in modalities]
            for cluster_id, modalities in cluster_to_modalities.items()
        },
        "client_rows": rows,
    }
    _save_if_requested(result, output_path)
    return result


# 构建不涉及 split/merge 语义的 evaluation 映射失败结果。
def _mapping_failure(message, rows, output_path):
    result = {
        "status": "failed",
        "failure_reason": EVALUATION_MAPPING_FAILURE,
        "failure_message": message,
        "mapping_type": "tolerant_evaluation_only",
        "uses_hidden_modality_id": True,
        "true_modalities": sorted({int(row["hidden_modality_id"]) for row in rows}),
        "pred_clusters": sorted({int(row["pred_cluster"]) for row in rows}),
        "client_rows": rows,
    }
    _save_if_requested(result, output_path)
    return result


# 在请求时保存 routing metadata。
def _save_if_requested(result, output_path):
    if output_path is None:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)


from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset



# 使用 naturally paired test 数据执行 tolerant fusion evaluation。
def evaluate_naturally_paired_fusion(
    server,
    clients_by_id,
    multimodal_path: Path,
    oracle_mapping: dict,
    cfg: dict,
    device,
):
    if oracle_mapping.get("status") != SUCCESS:
        return {
            "eval_status": "failed",
            "eval_failure_reason": oracle_mapping.get("failure_reason", "evaluation_mapping_failure"),
            "loss": None,
            "classification_loss": None,
            "accuracy": None,
            "macro_f1": None,
            "weighted_f1": None,
        }

    payload = torch.load(Path(multimodal_path), map_location="cpu")
    labels = payload["label"].long()
    modality_names = list(payload["modality_names"])
    modalities = payload["modalities"]
    modality_lengths = payload.get("modality_lengths")
    routing = build_tolerant_evaluation_routing(
        Path(cfg["evaluation"]["client_meta_path"]),
        Path(cfg["evaluation"]["cluster_assignment_path"]),
        clients_by_id,
        cfg["evaluation"].get("cluster_assignment_column", "pred_cluster"),
    )

    tensors = [modalities[name] for name in modality_names]
    length_tensors = None
    if modality_lengths is not None:
        length_tensors = [modality_lengths[name] for name in modality_names]
        dataset = TensorDataset(*tensors, *length_tensors, labels)
    else:
        dataset = TensorDataset(*tensors, labels)
    batch_size = int(cfg.get("training", {}).get("eval_batch_size", cfg.get("training", {}).get("batch_size", 64)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    ce = nn.CrossEntropyLoss(reduction="sum")
    y_true, y_pred = [], []
    total_loss = 0.0
    total = 0
    num_eval_batches = 0
    eval_modality_ids = list(range(len(tensors)))
    eval_cluster_ids = routing.pred_clusters

    server.eval()
    used_clients = list(clients_by_id.values())
    for client in used_clients:
        client.encoder.eval()

    with torch.no_grad():
        for batch in loader:
            if length_tensors is None:
                xs = list(batch[:-1])
                batch_lengths = [None] * len(xs)
            else:
                xs = list(batch[: len(tensors)])
                batch_lengths = list(batch[len(tensors) : 2 * len(tensors)])
            yb = batch[-1]
            slot_activations = route_paired_batch(xs, batch_lengths, routing, device)
            logits, _ = server(slot_activations)
            pred = logits.argmax(dim=1)
            y_true.extend(yb.tolist())
            y_pred.extend(pred.detach().cpu().tolist())
            total_loss += float(ce(logits, yb.to(device)).item())
            total += int(yb.numel())
            num_eval_batches += 1

    server.train()
    for client in used_clients:
        client.encoder.train()

    metrics = learning_metrics(y_true, y_pred)
    metrics.update(
        {
            "eval_status": "success",
            "eval_failure_reason": None,
            "loss": float(total_loss / max(1, total)),
            "classification_loss": float(total_loss / max(1, total)),
            "loss_type": "classification_cross_entropy",
            "num_eval_samples": int(total),
            "num_eval_batches": int(num_eval_batches),
            "eval_modality_ids": eval_modality_ids,
            "eval_cluster_ids": eval_cluster_ids,
            "oracle_mapping_type": oracle_mapping.get("mapping_type", "tolerant_evaluation_only"),
            "tolerant_routing": routing.to_metadata(),
        }
    )
    return metrics
