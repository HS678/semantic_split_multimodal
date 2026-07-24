from pathlib import Path

import torch
from torch import nn

from evaluation.fusion_eval import evaluate_naturally_paired_fusion


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
