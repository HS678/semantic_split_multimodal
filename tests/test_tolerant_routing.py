import csv
from pathlib import Path

import torch
from torch import nn

from MSL.evaluation.routing import (
    build_count_matrix,
    build_tolerant_evaluation_routing,
    normalize_count_matrix,
    route_paired_batch,
    validate_probability_matrix,
)
from MSL.learning.models import ConcatMLPFusionServer


# 提供输出维度固定的最小 encoder。
class TinyEncoder(nn.Module):
    # 初始化线性 encoder。
    def __init__(self, value: float):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.fill_(float(value))

    # 将输入映射到 slot representation。
    def forward(self, x, lengths=None):
        return self.linear(x)


# 提供带 encoder 的最小 client 对象。
class TinyClient:
    # 保存 client id 和 encoder。
    def __init__(self, client_id: str, value: float):
        self.client_id = client_id
        self.encoder = TinyEncoder(value)


# 写入 routing 测试所需的 metadata 和 assignment。
def write_fixture_files(tmp_path: Path, rows):
    meta_path = tmp_path / "client_meta.csv"
    assign_path = tmp_path / "pred_cluster.csv"
    with meta_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["client_id", "hidden_modality_id", "hidden_modality_name", "num_samples", "encoder_type"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "client_id": row["client_id"],
                    "hidden_modality_id": row["m"],
                    "hidden_modality_name": f"m{row['m']}",
                    "num_samples": 1,
                    "encoder_type": "tiny",
                }
            )
    with assign_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["client_id", "pred_cluster"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"client_id": row["client_id"], "pred_cluster": row["q"]})
    return meta_path, assign_path


# 断言 routing 矩阵能归一化且 server forward shape 正确。
def assert_routing_case(tmp_path: Path, rows, expected_q):
    clients = {row["client_id"]: TinyClient(row["client_id"], row["value"]) for row in rows}
    meta_path, assign_path = write_fixture_files(tmp_path, rows)
    routing = build_tolerant_evaluation_routing(meta_path, assign_path, clients)
    assert routing.pred_clusters == expected_q
    validate_probability_matrix(routing.p_mq, routing.pred_clusters)
    xs = [torch.ones(3, 2) * (modality + 1) for modality in routing.true_modalities]
    slots = route_paired_batch(xs, [None] * len(xs), routing, torch.device("cpu"))
    assert sorted(slots) == expected_q
    server = ConcatMLPFusionServer(expected_q, 2, 2, {"fusion": {"adapter_dim": 2, "hidden_dim": 4}})
    logits, _ = server(slots)
    assert logits.shape == (3, 2)


# 验证 correct routing。
def test_routing_correct(tmp_path):
    rows = [
        {"client_id": "c0", "m": 0, "q": 0, "value": 1.0},
        {"client_id": "c1", "m": 1, "q": 1, "value": 2.0},
    ]
    assert_routing_case(tmp_path, rows, [0, 1])


# 验证 pure split routing。
def test_routing_split_q_hat_greater_than_m(tmp_path):
    rows = [
        {"client_id": "c0", "m": 0, "q": 0, "value": 1.0},
        {"client_id": "c1", "m": 0, "q": 1, "value": 2.0},
    ]
    assert_routing_case(tmp_path, rows, [0, 1])


# 验证 pure merge routing。
def test_routing_merge_q_hat_less_than_m(tmp_path):
    rows = [
        {"client_id": "c0", "m": 0, "q": 0, "value": 1.0},
        {"client_id": "c1", "m": 1, "q": 0, "value": 2.0},
    ]
    assert_routing_case(tmp_path, rows, [0])


# 验证 split + merge 同时存在的 routing。
def test_routing_split_and_merge(tmp_path):
    rows = [
        {"client_id": "c0", "m": 0, "q": 0, "value": 1.0},
        {"client_id": "c1", "m": 0, "q": 1, "value": 2.0},
        {"client_id": "c2", "m": 1, "q": 1, "value": 3.0},
    ]
    assert_routing_case(tmp_path, rows, [0, 1])


# 验证同一 (true modality, predicted cluster) 下使用 activation mean。
def test_route_paired_batch_uses_activation_mean_for_same_mq(tmp_path):
    rows = [
        {"client_id": "c0", "m": 0, "q": 0, "value": 1.0},
        {"client_id": "c1", "m": 0, "q": 0, "value": 3.0},
    ]
    clients = {row["client_id"]: TinyClient(row["client_id"], row["value"]) for row in rows}
    meta_path, assign_path = write_fixture_files(tmp_path, rows)
    routing = build_tolerant_evaluation_routing(meta_path, assign_path, clients)

    xs = [torch.ones(2, 2)]
    slots = route_paired_batch(xs, [None], routing, torch.device("cpu"))

    encoder1_output = torch.full((2, 2), 2.0)
    encoder2_output = torch.full((2, 2), 6.0)
    expected = torch.stack([encoder1_output, encoder2_output], dim=0).mean(dim=0)
    assert torch.allclose(slots[0], expected)

    metadata = routing.to_metadata()
    assert metadata["evaluation_encoder_aggregation"] == "activation_mean"
    assert metadata["activation_ensemble_keys"] == [
        {"true_modality": 0, "pred_cluster": 0, "num_client_encoders": 2, "client_ids": ["c0", "c1"]}
    ]


# 验证 N_mq 与 P_mq 的列归一化逻辑。
def test_count_and_probability_matrix():
    rows = [
        {"hidden_modality_id": 0, "pred_cluster": 0},
        {"hidden_modality_id": 1, "pred_cluster": 0},
        {"hidden_modality_id": 1, "pred_cluster": 1},
    ]
    _, pred_clusters, n_mq = build_count_matrix(rows)
    p_mq = normalize_count_matrix(n_mq, pred_clusters)
    validate_probability_matrix(p_mq, pred_clusters)
    assert p_mq[0][0] == 0.5
    assert p_mq[1][0] == 0.5
    assert p_mq[1][1] == 1.0
