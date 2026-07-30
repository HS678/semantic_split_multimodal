import json

import torch
from torch import nn

from semantic_split_multimodal.learning import fusion_sl
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


def test_validation_selects_best_early_stops_restores_best_and_tests_once(
    monkeypatch,
    tmp_path,
):
    clients = [
        TinyClient("c0", 0, torch.tensor([0, 1, 1])),
        TinyClient("c1", 1, torch.tensor([1, 0, 1])),
    ]
    for client in clients:
        client.device = torch.device("cpu")

    class Scheduler:
        def sample_round(self):
            return clients

        def metrics(self, _selected):
            return {
                "coverage": 1.0,
                "participation_fairness": 1.0,
                "clients_per_cluster_per_round": 1,
                "per_cluster_selected": {"0": 1, "1": 1},
            }

    train_metrics = {
        "loss": 1.0,
        "mean_loss": 1.0,
        "accuracy": 0.5,
        "K_t": 2,
        "pseudo_batch_size": 2,
        "total_pseudo_samples": 2,
        "mean_pseudo_batch_size": 2.0,
        "common_labels": [0, 1],
        "local_step_common_labels": [[0, 1]],
        "binding_success": 1.0,
        "empty_binding_round": 0,
        "round_status": "effective",
        "configured_local_steps": 1,
        "attempted_local_steps": 1,
        "effective_local_steps": 1,
        "empty_binding_local_steps": 0,
        "round_binding_success_rate": 1.0,
        "cluster_slot_coverage": 1.0,
        "server_update_l1": 0.0,
        "client_update_l1": 0.0,
    }
    validation_scores = iter([0.5, 0.4, 0.3])
    eval_paths = []
    test_server_weight = []

    monkeypatch.setattr(fusion_sl, "_load_clients", lambda *_args: clients)
    monkeypatch.setattr(fusion_sl, "build_scheduler", lambda *_args, **_kwargs: Scheduler())

    def fake_train_round(server, *_args, **_kwargs):
        with torch.no_grad():
            for parameter in server.parameters():
                parameter.add_(1.0)
        return dict(train_metrics)

    def fake_evaluate(server, _clients, path, *_args, **_kwargs):
        eval_paths.append(path.name)
        if path.name == "validation_multimodal.pt":
            score = next(validation_scores)
        else:
            score = 0.45
            test_server_weight.append(next(server.parameters()).detach().clone())
        return {
            "eval_status": "success",
            "eval_failure_reason": None,
            "loss": 1.0,
            "accuracy": score,
            "macro_f1": score,
            "weighted_f1": score,
        }

    monkeypatch.setattr(fusion_sl, "_train_round", fake_train_round)
    monkeypatch.setattr(fusion_sl, "build_oracle_eval_mapping", lambda *_args: {"status": "success"})
    monkeypatch.setattr(fusion_sl, "evaluate_naturally_paired_fusion", fake_evaluate)

    cfg = {
        "seed": 101,
        "encoder_hidden_dim": 3,
        "num_classes": 2,
        "partition": {"output_dir": str(tmp_path / "stage1")},
        "cluster": {"output_dir": str(tmp_path / "stage2")},
        "result": {"output_dir": str(tmp_path / "run")},
        "result_model": {"output_dir": str(tmp_path / "run")},
        "training": {
            "scheduler": "balanced_cluster_round_robin",
            "global_rounds": 5,
            "validation_every": 1,
            "early_stopping": {"patience": 2, "min_rounds": 1, "min_delta": 0.001},
            "local_steps": 1,
            "clients_per_cluster_per_round": 1,
            "server_lr": 0.001,
        },
        "binding": {"type": "label_random", "batch_size": 2},
        "fusion": {
            "type": "concat_mlp",
            "adapter_dim": 3,
            "hidden_dim": 4,
            "num_layers": 1,
            "dropout": 0.0,
        },
    }

    result = fusion_sl.run_mmbind_fusion_stage3_split_training(
        cfg,
        tmp_path,
        torch.device("cpu"),
    )

    assert result["stop_reason"] == "early_stopping"
    assert result["stop_round"] == 3
    assert result["best_round"] == 1
    assert result["test_evaluation_count"] == 1
    assert result["official_result"] == {
        "selection": "best_validation_macro_f1",
        "metrics_file": "final_metrics.json",
        "model_file": "best_model.pt",
    }
    assert eval_paths == [
        "validation_multimodal.pt",
        "validation_multimodal.pt",
        "validation_multimodal.pt",
        "test_multimodal.pt",
    ]
    saved_metrics = json.loads((tmp_path / "run" / "final_metrics.json").read_text(encoding="utf-8"))
    best_metrics = json.loads((tmp_path / "run" / "best_metrics.json").read_text(encoding="utf-8"))
    assert saved_metrics["official_result"] == result["official_result"]
    assert saved_metrics["checkpoint"] == "best_model.pt"
    assert saved_metrics["selected_by"] == "validation_macro_f1"
    assert saved_metrics["test_macro_f1"] == 0.45
    assert saved_metrics["test_weighted_f1"] == 0.45
    assert "final_eval" not in saved_metrics
    assert best_metrics["best_round"] == 1
    best = torch.load(tmp_path / "run" / "best_model.pt", map_location="cpu")
    last = torch.load(tmp_path / "run" / "last_model.pt", map_location="cpu")
    assert not (tmp_path / "run" / "final_model.pt").exists()
    assert any(
        not torch.equal(value, last["server_state_dict"][key])
        for key, value in best["server_state_dict"].items()
    )
    assert torch.equal(test_server_weight[0], next(iter(best["server_state_dict"].values())))
