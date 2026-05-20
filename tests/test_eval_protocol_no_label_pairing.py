import pytest
import torch

from server.evaluation import evaluate_paired_test
import server.server_core as server_core


class _DummyEncoder(torch.nn.Module):
    def __init__(self, in_dim=4, out_dim=8):
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


class _DummyClient:
    def __init__(self):
        self.encoder = _DummyEncoder()


class _DummyProjector(torch.nn.Module):
    def __init__(self, in_dim=8, out_dim=6):
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


class _DummyFusion(torch.nn.Module):
    def forward(self, projected_list):
        return torch.cat(projected_list, dim=1)


class _DummyServer(torch.nn.Module):
    def __init__(self, num_modalities=2, proj_dim=6, num_classes=3):
        super().__init__()
        self.projectors = torch.nn.ModuleDict({str(i): _DummyProjector(8, proj_dim) for i in range(num_modalities)})
        self.fusion = _DummyFusion()
        self.classifier = torch.nn.Linear(proj_dim * num_modalities, num_classes)


def test_eval_phase_does_not_use_label_guided_pairing(monkeypatch):
    # If eval accidentally uses training-time label-guided builder, this test must fail.
    def _forbidden(*args, **kwargs):
        raise RuntimeError("SemanticBatchBuilder.build must not be used in test evaluation")

    monkeypatch.setattr(server_core.SemanticBatchBuilder, "build", staticmethod(_forbidden))

    n = 12
    in_dim = 4
    num_modalities = 2
    num_classes = 3

    test_set = {
        "modalities": [torch.randn(n, in_dim), torch.randn(n, in_dim)],
        "labels": torch.randint(0, num_classes, (n,), dtype=torch.long),
    }

    clients_by_modality = {0: [_DummyClient()], 1: [_DummyClient()]}
    server = _DummyServer(num_modalities=num_modalities, proj_dim=6, num_classes=num_classes)

    # Should run without touching SemanticBatchBuilder.build
    metrics = evaluate_paired_test(clients_by_modality, server, test_set, torch.device("cpu"))

    assert 0.0 <= metrics["acc"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
