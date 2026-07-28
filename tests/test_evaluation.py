import csv
from pathlib import Path

import torch
from torch import nn

from semantic_split_multimodal.evaluation.fusion_eval import evaluate_naturally_paired_fusion
from semantic_split_multimodal.evaluation.oracle_mapping import (
    MERGED_TRUE_MODALITY_FAILURE,
    SPLIT_TRUE_MODALITY_FAILURE,
    SUCCESS,
    build_oracle_eval_mapping,
)


class TraceEncoder(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.offset = float(offset)
        self.seen = []

    def forward(self, x):
        self.seen.append(x.detach().cpu().clone())
        value = x.reshape(x.shape[0], -1).mean(dim=1, keepdim=True) + self.offset
        return torch.cat([value, value], dim=1) * self.weight


class EvalClient:
    def __init__(self, client_id, encoder):
        self.client_id = client_id
        self.encoder = encoder


class TraceServer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
        self.seen_slots = []

    def forward(self, slot_activations):
        self.seen_slots.append({k: v.detach().cpu().clone() for k, v in slot_activations.items()})
        fused = torch.cat([slot_activations[5], slot_activations[9]], dim=1)
        return self.fc(fused), fused


def test_fusion_eval_uses_naturally_paired_indices_and_not_label_selection(tmp_path):
    test_path = tmp_path / "test_multimodal.pt"
    mod0 = torch.tensor([[1.0], [2.0], [3.0]])
    mod1 = torch.tensor([[10.0], [20.0], [30.0]])
    labels = torch.tensor([1, 0, 1])
    torch.save(
        {
            "label": labels,
            "modalities": {"m0": mod0, "m1": mod1},
            "modality_names": ["m0", "m1"],
            "modality_input_shapes": {"m0": [1], "m1": [1]},
        },
        test_path,
    )
    enc0 = TraceEncoder(offset=0.0)
    enc1 = TraceEncoder(offset=100.0)
    server = TraceServer()
    mapping = {
        "status": "success",
        "mapping_type": "oracle_evaluation_only",
        "modality_to_cluster": {"0": 5, "1": 9},
        "representative_clients": {"0": "client_001", "1": "client_010"},
    }

    metrics = evaluate_naturally_paired_fusion(
        server,
        {"client_001": EvalClient("client_001", enc0), "client_010": EvalClient("client_010", enc1)},
        test_path,
        mapping,
        {"training": {"eval_batch_size": 3}},
        torch.device("cpu"),
    )

    assert metrics["eval_status"] == "success"
    assert torch.equal(enc0.seen[0], mod0)
    assert torch.equal(enc1.seen[0], mod1)
    assert server.seen_slots[0][5].shape[0] == 3
    assert server.seen_slots[0][9].shape[0] == 3


def test_fusion_eval_returns_unavailable_metrics_when_mapping_fails(tmp_path):
    metrics = evaluate_naturally_paired_fusion(
        TraceServer(),
        {},
        tmp_path / "missing.pt",
        {"status": "failed", "failure_reason": "split_true_modality_failure"},
        {"training": {"eval_batch_size": 3}},
        torch.device("cpu"),
    )

    assert metrics["eval_status"] == "failed"
    assert metrics["eval_failure_reason"] == "split_true_modality_failure"
    assert metrics["accuracy"] is None
    assert metrics["macro_f1"] is None
    assert metrics["loss"] is None


def _write_csv(path: Path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_inputs(tmp_path: Path, meta_rows, assignment_rows):
    meta = tmp_path / "client_meta.csv"
    assignments = tmp_path / "pred_cluster.csv"
    _write_csv(meta, ["client_id", "hidden_modality_id"], meta_rows)
    _write_csv(assignments, ["client_id", "pred_cluster"], assignment_rows)
    return meta, assignments


def test_oracle_one_to_one_mapping_success(tmp_path):
    meta, assignments = _write_inputs(
        tmp_path,
        [
            {"client_id": "client_002", "hidden_modality_id": 0},
            {"client_id": "client_001", "hidden_modality_id": 0},
            {"client_id": "client_010", "hidden_modality_id": 1},
        ],
        [
            {"client_id": "client_002", "pred_cluster": 3},
            {"client_id": "client_001", "pred_cluster": 3},
            {"client_id": "client_010", "pred_cluster": 7},
        ],
    )

    result = build_oracle_eval_mapping(meta, assignments, tmp_path / "oracle_eval_modality_to_cluster.json")

    assert result["status"] == SUCCESS
    assert result["modality_to_cluster"] == {"0": 3, "1": 7}
    assert result["cluster_to_modality"] == {"3": 0, "7": 1}
    assert result["representative_clients"] == {"0": "client_001", "1": "client_010"}
    assert (tmp_path / "oracle_eval_modality_to_cluster.json").exists()


def test_oracle_mapping_fails_when_true_modality_is_split(tmp_path):
    meta, assignments = _write_inputs(
        tmp_path,
        [
            {"client_id": "client_001", "hidden_modality_id": 0},
            {"client_id": "client_002", "hidden_modality_id": 0},
            {"client_id": "client_010", "hidden_modality_id": 1},
        ],
        [
            {"client_id": "client_001", "pred_cluster": 3},
            {"client_id": "client_002", "pred_cluster": 4},
            {"client_id": "client_010", "pred_cluster": 7},
        ],
    )

    result = build_oracle_eval_mapping(meta, assignments)

    assert result["status"] == "failed"
    assert result["failure_reason"] == SPLIT_TRUE_MODALITY_FAILURE
    assert result["modality_to_cluster"] is None


def test_oracle_mapping_fails_when_cluster_merges_true_modalities(tmp_path):
    meta, assignments = _write_inputs(
        tmp_path,
        [
            {"client_id": "client_001", "hidden_modality_id": 0},
            {"client_id": "client_010", "hidden_modality_id": 1},
        ],
        [
            {"client_id": "client_001", "pred_cluster": 3},
            {"client_id": "client_010", "pred_cluster": 3},
        ],
    )

    result = build_oracle_eval_mapping(meta, assignments)

    assert result["status"] == "failed"
    assert result["failure_reason"] == MERGED_TRUE_MODALITY_FAILURE
    assert result["cluster_to_modality"] is None
