import torch
from torch import nn

from semantic_split_multimodal.learning.fusion_sl import _train_round


class TinyClient:
    def __init__(self, client_id, pred_cluster, labels):
        self.client_id = client_id
        self.pred_cluster = pred_cluster
        self.label_batches = list(labels) if isinstance(labels, (list, tuple)) else [labels]
        self.labels = self.label_batches[0]
        self.batch_size = int(self.labels.numel())
        self.encoder = nn.Linear(3, 3, bias=False)
        self.optimizer = torch.optim.SGD(self.encoder.parameters(), lr=0.01)
        self.samples = torch.eye(3)[: self.batch_size]
        self.backward_called = False
        self.sample_calls = 0
        self.backward_calls = 0

    def sample_batch(self):
        labels = self.label_batches[min(self.sample_calls, len(self.label_batches) - 1)]
        self.sample_calls += 1
        return self.samples, labels

    def forward_detached(self, x):
        z_client = self.encoder(x)
        z_server = z_client.detach().requires_grad_(True)
        return z_client, z_server

    def backward_from_server(self, z_client, grad):
        self.backward_called = True
        self.backward_calls += 1
        self.optimizer.zero_grad()
        z_client.backward(grad)
        self.optimizer.step()


class TinyServer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6, 2)

    def forward(self, slot_activations):
        fused = torch.cat([slot_activations[0], slot_activations[1]], dim=1)
        return self.fc(fused), fused


def test_train_round_builds_label_random_pseudo_batch_and_splits_backward():
    clients = [
        TinyClient("c0", 0, torch.tensor([0, 1, 1])),
        TinyClient("c1", 1, torch.tensor([1, 0, 1])),
    ]
    server = TinyServer()
    opt = torch.optim.SGD(server.parameters(), lr=0.01)

    metrics = _train_round(
        server,
        opt,
        clients,
        required_clusters=[0, 1],
        cfg={"binding": {"batch_size": 3}, "training": {"batch_size": 3}},
    )

    assert metrics["binding_success"] == 1.0
    assert metrics["pseudo_batch_size"] == 3
    assert metrics["configured_local_steps"] == 1
    assert metrics["attempted_local_steps"] == 1
    assert metrics["effective_local_steps"] == 1
    assert metrics["empty_binding_local_steps"] == 0
    assert metrics["round_status"] == "effective"
    assert all(client.backward_called for client in clients)


def test_train_round_drops_binding_without_complete_pred_cluster_coverage():
    clients = [TinyClient("c0", 0, torch.tensor([0, 1, 1]))]
    server = TinyServer()
    opt = torch.optim.SGD(server.parameters(), lr=0.01)

    try:
        _train_round(
            server,
            opt,
            clients,
            required_clusters=[0, 1],
            cfg={"binding": {"batch_size": 3}, "training": {"batch_size": 3}},
        )
    except RuntimeError as exc:
        assert "Scheduler failed to cover all predicted clusters" in str(exc)
    else:
        raise AssertionError("Expected missing cluster coverage to raise RuntimeError.")

    assert not clients[0].backward_called


def test_train_round_skips_empty_binding_when_clusters_share_no_label():
    clients = [
        TinyClient("c0", 0, torch.tensor([0, 0, 0])),
        TinyClient("c1", 1, torch.tensor([1, 1, 1])),
    ]
    server = TinyServer()
    opt = torch.optim.SGD(server.parameters(), lr=0.01)

    metrics = _train_round(
        server,
        opt,
        clients,
        required_clusters=[0, 1],
        cfg={"binding": {"batch_size": 3}, "training": {"batch_size": 3}},
    )

    assert metrics["binding_success"] == 0.0
    assert metrics["empty_binding_round"] == 1
    assert metrics["empty_binding_local_steps"] == 1
    assert metrics["pseudo_batch_size"] == 0
    assert metrics["common_labels"] == []
    assert metrics["local_step_common_labels"] == [[]]
    assert metrics["round_status"] == "empty_binding_round"
    assert not any(client.backward_called for client in clients)


def test_train_round_reuses_selected_clients_across_local_steps_and_skips_only_empty_steps():
    clients = [
        TinyClient("c0", 0, [torch.tensor([0, 0, 0]), torch.tensor([0, 1, 1]), torch.tensor([1, 1, 0])]),
        TinyClient("c1", 1, [torch.tensor([1, 1, 1]), torch.tensor([1, 0, 1]), torch.tensor([0, 1, 0])]),
    ]
    server = TinyServer()
    opt = torch.optim.SGD(server.parameters(), lr=0.01)

    metrics = _train_round(
        server,
        opt,
        clients,
        required_clusters=[0, 1],
        cfg={"binding": {"batch_size": 3}, "training": {"batch_size": 3, "local_steps": 3}},
    )

    assert metrics["configured_local_steps"] == 3
    assert metrics["attempted_local_steps"] == 3
    assert metrics["effective_local_steps"] == 2
    assert metrics["empty_binding_local_steps"] == 1
    assert metrics["round_binding_success_rate"] == 2 / 3
    assert metrics["total_pseudo_samples"] == 6
    assert metrics["mean_pseudo_batch_size"] == 3.0
    assert metrics["round_status"] == "effective"
    assert all(client.sample_calls == 3 for client in clients)
    assert all(client.backward_calls == 2 for client in clients)
