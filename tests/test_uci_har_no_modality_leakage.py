from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from data.uci_har_adapter import load_uci_har_dataset
from server.evaluation import evaluate_paired_test


class _RecorderEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input = None
        self.fc = torch.nn.Linear(384, 16)

    def forward(self, x):
        self.last_input = x.detach().clone()
        return self.fc(x)


class _DummyClient:
    def __init__(self, encoder):
        self.encoder = encoder


class _DummyProjector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(16, 8)

    def forward(self, x):
        return self.fc(x)


class _DummyFusion(torch.nn.Module):
    def forward(self, projected_list):
        return torch.cat(projected_list, dim=1)


class _DummyServer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projectors = torch.nn.ModuleDict({"0": _DummyProjector(), "1": _DummyProjector()})
        self.fusion = _DummyFusion()
        self.classifier = torch.nn.Linear(16, 6)


def test_uci_har_inputs_are_modality_isolated_and_routed():
    cfg = {"dataset": {"root": "./data/uci-har"}}
    split = load_uci_har_dataset(cfg, ROOT)
    train = split["train"]
    test = split["test"]

    acc = train["modalities"][0]
    gyro = train["modalities"][1]
    assert acc.shape[1] == 384
    assert gyro.shape[1] == 384
    assert not torch.allclose(acc, gyro)

    raw_root = ROOT / "data" / "uci-har" / "train" / "Inertial Signals"
    acc_x = np.loadtxt(raw_root / "body_acc_x_train.txt", dtype=np.float32)
    acc_y = np.loadtxt(raw_root / "body_acc_y_train.txt", dtype=np.float32)
    acc_z = np.loadtxt(raw_root / "body_acc_z_train.txt", dtype=np.float32)
    gyro_x = np.loadtxt(raw_root / "body_gyro_x_train.txt", dtype=np.float32)
    gyro_y = np.loadtxt(raw_root / "body_gyro_y_train.txt", dtype=np.float32)
    gyro_z = np.loadtxt(raw_root / "body_gyro_z_train.txt", dtype=np.float32)

    acc_ref = np.stack([acc_x, acc_y, acc_z], axis=1).reshape(acc_x.shape[0], -1)
    gyro_ref = np.stack([gyro_x, gyro_y, gyro_z], axis=1).reshape(gyro_x.shape[0], -1)

    assert np.allclose(acc[:16].cpu().numpy(), acc_ref[:16], atol=1e-5)
    assert np.allclose(gyro[:16].cpu().numpy(), gyro_ref[:16], atol=1e-5)

    acc_encoder = _RecorderEncoder()
    gyro_encoder = _RecorderEncoder()
    clients_by_modality = {0: [_DummyClient(acc_encoder)], 1: [_DummyClient(gyro_encoder)]}
    server = _DummyServer()
    _ = evaluate_paired_test(clients_by_modality, server, test, torch.device("cpu"))

    assert acc_encoder.last_input is not None and gyro_encoder.last_input is not None
    assert torch.allclose(acc_encoder.last_input, test["modalities"][0], atol=1e-6)
    assert torch.allclose(gyro_encoder.last_input, test["modalities"][1], atol=1e-6)
