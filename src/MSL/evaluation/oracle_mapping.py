import csv
import json
from pathlib import Path


SUCCESS = "success"
SPLIT_TRUE_MODALITY_FAILURE = "split_true_modality_failure"
MERGED_TRUE_MODALITY_FAILURE = "merged_true_modality_failure"
EVALUATION_MAPPING_FAILURE = "evaluation_mapping_failure"


def build_oracle_eval_mapping(
    client_meta_path: Path,
    cluster_assignments_path: Path,
    output_path: Path | None = None,
    assignment_column: str = "pred_cluster",
):
    client_meta = _read_client_meta(Path(client_meta_path))
    assignments = _read_cluster_assignments(Path(cluster_assignments_path), assignment_column)

    rows = []
    for client_id, meta in client_meta.items():
        if client_id not in assignments:
            return _mapping_failure(
                EVALUATION_MAPPING_FAILURE,
                f"Missing pred_cluster assignment for client {client_id}.",
                rows,
                output_path,
            )
        rows.append(
            {
                "client_id": client_id,
                "hidden_modality_id": int(meta["hidden_modality_id"]),
                "pred_cluster": int(assignments[client_id]),
            }
        )

    true_modalities = sorted({row["hidden_modality_id"] for row in rows})
    pred_clusters = sorted({row["pred_cluster"] for row in rows})

    modality_to_clusters = {
        modality_id: sorted({row["pred_cluster"] for row in rows if row["hidden_modality_id"] == modality_id})
        for modality_id in true_modalities
    }
    split = {modality_id: clusters for modality_id, clusters in modality_to_clusters.items() if len(clusters) != 1}
    if split:
        return _mapping_failure(
            SPLIT_TRUE_MODALITY_FAILURE,
            "At least one true modality maps to multiple predicted clusters.",
            rows,
            output_path,
            modality_to_clusters=modality_to_clusters,
            pred_clusters=pred_clusters,
        )

    cluster_to_modalities = {
        cluster_id: sorted({row["hidden_modality_id"] for row in rows if row["pred_cluster"] == cluster_id})
        for cluster_id in pred_clusters
    }
    merged = {cluster_id: modalities for cluster_id, modalities in cluster_to_modalities.items() if len(modalities) != 1}
    if merged:
        return _mapping_failure(
            MERGED_TRUE_MODALITY_FAILURE,
            "At least one predicted cluster contains multiple true modalities.",
            rows,
            output_path,
            modality_to_clusters=modality_to_clusters,
            cluster_to_modalities=cluster_to_modalities,
            pred_clusters=pred_clusters,
        )

    if len(true_modalities) != len(pred_clusters):
        return _mapping_failure(
            EVALUATION_MAPPING_FAILURE,
            "The number of true modalities and predicted clusters does not match.",
            rows,
            output_path,
            modality_to_clusters=modality_to_clusters,
            cluster_to_modalities=cluster_to_modalities,
            pred_clusters=pred_clusters,
        )

    modality_to_cluster = {int(modality_id): int(clusters[0]) for modality_id, clusters in modality_to_clusters.items()}
    cluster_to_modality = {int(cluster_id): int(modalities[0]) for cluster_id, modalities in cluster_to_modalities.items()}
    representative_clients = {}
    for modality_id, cluster_id in modality_to_cluster.items():
        candidates = [
            row["client_id"]
            for row in rows
            if int(row["hidden_modality_id"]) == int(modality_id) and int(row["pred_cluster"]) == int(cluster_id)
        ]
        representative_clients[int(modality_id)] = sorted(candidates)[0]

    result = {
        "status": SUCCESS,
        "failure_reason": None,
        "mapping_type": "oracle_evaluation_only",
        "uses_hidden_modality_id": True,
        "true_modalities": [int(v) for v in true_modalities],
        "pred_clusters": [int(v) for v in pred_clusters],
        "true_Q": int(len(true_modalities)),
        "estimated_Q": int(len(pred_clusters)),
        "modality_to_cluster": {str(k): int(v) for k, v in modality_to_cluster.items()},
        "cluster_to_modality": {str(k): int(v) for k, v in cluster_to_modality.items()},
        "representative_clients": {str(k): v for k, v in representative_clients.items()},
        "client_rows": rows,
    }
    _save_if_requested(result, output_path)
    return result


def _mapping_failure(reason, message, rows, output_path, **extra):
    true_modalities = sorted({int(row["hidden_modality_id"]) for row in rows})
    pred_clusters = sorted({int(row["pred_cluster"]) for row in rows})
    result = {
        "status": "failed",
        "failure_reason": reason,
        "failure_message": message,
        "mapping_type": "oracle_evaluation_only",
        "uses_hidden_modality_id": True,
        "true_modalities": true_modalities,
        "pred_clusters": pred_clusters,
        "true_Q": int(len(true_modalities)),
        "estimated_Q": int(len(pred_clusters)),
        "modality_to_cluster": None,
        "cluster_to_modality": None,
        "representative_clients": None,
        "client_rows": rows,
    }
    result.update(_stringify_nested_int_keys(extra))
    _save_if_requested(result, output_path)
    return result


def _read_client_meta(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing client metadata: {path}")
    rows = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["client_id"]] = {"hidden_modality_id": int(row["hidden_modality_id"])}
    return rows


def _read_cluster_assignments(path: Path, assignment_column: str = "pred_cluster"):
    if not path.exists():
        raise FileNotFoundError(f"Missing cluster assignments: {path}")
    rows = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["client_id"]] = int(row[assignment_column])
    return rows


def _save_if_requested(result, output_path):
    if output_path is None:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)


def _stringify_nested_int_keys(value):
    if isinstance(value, dict):
        return {str(k): _stringify_nested_int_keys(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_stringify_nested_int_keys(v) for v in value]
    return value
