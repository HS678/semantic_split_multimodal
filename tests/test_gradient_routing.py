import torch

from server.server_core import SplitServer


def test_gradient_scatter_and_routing_partial_batch():
    cfg = {
        "num_modalities": 2,
        "encoder_hidden_dim": 8,
        "projected_dim": 4,
        "num_classes": 3,
        "temperature": 0.1,
        "learning_rate": 1e-3,
        "lambda_cls": 1.0,
        "lambda_align": 0.1,
        "lambda_proto": 0.0,
    }
    server = SplitServer(cfg, device=torch.device("cpu"))

    # c1 labels: [0,1,2,2], c2 labels: [1,2,2,2] => common [1,2], min counts => 1 + 2 = 3
    z1 = torch.randn(4, 8, requires_grad=True)
    z2 = torch.randn(4, 8, requires_grad=True)
    payloads = [
        {"client_id": "c1", "modality_id": 0, "y": torch.tensor([0, 1, 2, 2]), "z_server": z1},
        {"client_id": "c2", "modality_id": 1, "y": torch.tensor([1, 2, 2, 2]), "z_server": z2},
    ]

    out = server.train_step(payloads)
    assert out is not None

    g1 = out["grad_to_clients"]["c1"]
    g2 = out["grad_to_clients"]["c2"]
    assert g1.shape == z1.shape
    assert g2.shape == z2.shape

    # At least one row should be exactly zero due to partial semantic subset backfill.
    row_norm_1 = torch.norm(g1, dim=1)
    row_norm_2 = torch.norm(g2, dim=1)
    assert (row_norm_1 == 0).any() or (row_norm_2 == 0).any()

    # Participating rows should carry non-zero gradients.
    assert (row_norm_1 > 0).any()
    assert (row_norm_2 > 0).any()
