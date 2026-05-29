import copy
import torch

from clients.client_node import SplitClient
from server.server_core import SplitServer


def _make_client(client_id, modality_id, x, y, cfg):
    return SplitClient(
        {
            "client_id": client_id,
            "modality_id": modality_id,
            "x": x,
            "y": y,
            "gt_cluster": modality_id,
            "input_dim": int(x.shape[1]),
        },
        cfg,
        torch.device("cpu"),
    )


def test_no_second_detach_and_client_update():
    cfg = {
        "seed": 42,
        "num_modalities": 2,
        "num_classes": 3,
        "encoder_hidden_dim": 8,
        "projected_dim": 4,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "temperature": 0.1,
        "lambda_cls": 1.0,
        "lambda_align": 0.1,
        "lambda_proto": 0.0,
    }

    # Ensure common labels exist between two clients.
    x1 = torch.randn(10, 6)
    y1 = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 1], dtype=torch.long)
    x2 = torch.randn(10, 6)
    y2 = torch.tensor([1, 2, 2, 1, 0, 0, 2, 1, 0, 2], dtype=torch.long)

    c1 = _make_client("c1", 0, x1, y1, cfg)
    c2 = _make_client("c2", 1, x2, y2, cfg)
    server = SplitServer(cfg, device=torch.device("cpu"))

    # deterministic mini-batches to guarantee common labels
    b1x, b1y = c1.x[:4], c1.y[:4]  # labels include 0/1/2
    b2x, b2y = c2.x[:4], c2.y[:4]  # labels include 1/2
    z1_client, z1_server = c1.forward_to_server(b1x)
    z2_client, z2_server = c2.forward_to_server(b2x)

    payloads = [
        {"client_id": "c1", "modality_id": 0, "y": b1y, "z_server": z1_server},
        {"client_id": "c2", "modality_id": 1, "y": b2y, "z_server": z2_server},
    ]
    out = server.train_step(payloads)
    assert out is not None

    assert z1_server.grad is not None
    assert z2_server.grad is not None

    g1 = out["grad_to_clients"]["c1"]
    g2 = out["grad_to_clients"]["c2"]
    assert g1.shape == z1_client.shape
    assert g2.shape == z2_client.shape

    before_1 = [p.detach().clone() for p in c1.encoder.parameters()]
    before_2 = [p.detach().clone() for p in c2.encoder.parameters()]

    c1.backward_update(z1_client, g1)
    c2.backward_update(z2_client, g2)

    after_1 = [p.detach().clone() for p in c1.encoder.parameters()]
    after_2 = [p.detach().clone() for p in c2.encoder.parameters()]

    changed_1 = any(not torch.allclose(a, b) for a, b in zip(before_1, after_1))
    changed_2 = any(not torch.allclose(a, b) for a, b in zip(before_2, after_2))
    assert changed_1
    assert changed_2
