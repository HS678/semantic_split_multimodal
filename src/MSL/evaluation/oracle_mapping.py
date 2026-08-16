import json
from pathlib import Path

from MSL.evaluation.routing import (
    SUCCESS,
    build_count_matrix,
    normalize_count_matrix,
    read_client_meta,
    read_cluster_assignments,
    validate_probability_matrix,
)


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
