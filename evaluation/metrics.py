import numpy as np
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score, normalized_mutual_info_score


def discovery_metrics(true_modality, pred_cluster):
    true = np.asarray(true_modality, dtype=int)
    pred = np.asarray(pred_cluster, dtype=int)
    mapping = {}
    mapped = np.zeros_like(pred)
    for cluster_id in sorted(np.unique(pred)):
        members = true[pred == cluster_id]
        values, counts = np.unique(members, return_counts=True)
        majority = int(values[np.argmax(counts)])
        mapping[int(cluster_id)] = majority
        mapped[pred == cluster_id] = majority
    return {
        "ARI": float(adjusted_rand_score(true, pred)),
        "NMI": float(normalized_mutual_info_score(true, pred)),
        "ACC": float(accuracy_score(true, mapped)),
        "estimated_num_clusters": int(len(np.unique(pred))),
        "cluster_to_true_modality_majority": {str(k): int(v) for k, v in mapping.items()},
    }


def learning_metrics(y_true, y_pred, modality_ids=None):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if len(y_true) else 0.0,
    }
    if modality_ids is not None:
        modality_ids = np.asarray(modality_ids, dtype=int)
        out["per_modality_accuracy"] = {}
        for modality_id in sorted(np.unique(modality_ids)):
            mask = modality_ids == modality_id
            out["per_modality_accuracy"][str(int(modality_id))] = (
                float(accuracy_score(y_true[mask], y_pred[mask])) if mask.any() else 0.0
            )
    return out


def d2d_metrics(latency_without_d2d, latency_with_d2d):
    base = float(latency_without_d2d)
    current = float(latency_with_d2d)
    return {
        "latency": current,
        "speedup": float(base / current) if current > 0 else 0.0,
    }
