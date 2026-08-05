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
