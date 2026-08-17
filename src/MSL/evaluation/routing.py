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


# 读取 Stage1 保存的 client metadata。
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


# 读取 Stage2 保存的预测簇或 oracle 簇 assignment。
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
