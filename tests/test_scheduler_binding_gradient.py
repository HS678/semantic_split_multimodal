# 测试调度、binding 与梯度更新的关键行为。
import copy
from dataclasses import dataclass

import pytest
import torch
from torch import nn

from MSL.binding import ClientActivationBatch, build_label_random_pseudo_batch
from MSL.training import _round_modality_metrics, _train_local_step
from MSL.models import ConcatMLPFusionServer
from MSL.scheduling import BalancedClusterRoundRobinScheduler, RandomScheduler


@dataclass
class DummyClient:
    client_id: str
    pred_cluster: int


# 构建指定簇分布的 dummy clients。
def make_clients(counts):
    clients = []
    index = 0
    for cluster_id, count in counts.items():
        for _ in range(count):
            clients.append(DummyClient(f"client_{index:03d}", int(cluster_id)))
            index += 1
    return clients


# 验证 cluster-balanced scheduler 固定每轮总预算并均衡分配到 cluster。
def test_balanced_scheduler_fixed_total_budget_distinct():
    scheduler = BalancedClusterRoundRobinScheduler(make_clients({0: 5, 1: 5}), clients_per_round=4, seed=7)
    selected = scheduler.sample_round()
    assert len(selected) == 4
    assert len({client.client_id for client in selected}) == 4
    assert scheduler.metrics(selected)["per_cluster_selected"] == {"0": 2, "1": 2}
    assert scheduler.metrics(selected)["per_cluster_budget"] == {"0": 2, "1": 2}


# 验证固定总预算小于 cluster 数时允许部分 cluster 本轮为 0，并轮转余数。
def test_balanced_scheduler_allows_zero_budget_clusters_and_rotates():
    scheduler = BalancedClusterRoundRobinScheduler(make_clients({0: 2, 1: 2, 2: 2}), clients_per_round=2, seed=7)
    first = scheduler.sample_round()
    first_metrics = scheduler.metrics(first)
    second = scheduler.sample_round()
    second_metrics = scheduler.metrics(second)

    assert len(first) == 2
    assert len(second) == 2
    assert first_metrics["per_cluster_budget"] == {"0": 1, "1": 1, "2": 0}
    assert second_metrics["per_cluster_budget"] == {"0": 0, "1": 1, "2": 1}


# 验证 cluster size 不足时按容量裁剪并把剩余预算分配给有容量的 cluster。
def test_balanced_scheduler_capacity_aware_small_cluster():
    clients = make_clients({0: 1, 1: 5})
    original_clusters = {client.client_id: client.pred_cluster for client in clients}
    scheduler = BalancedClusterRoundRobinScheduler(clients, clients_per_round=4, seed=7)
    selected = scheduler.sample_round()
    metrics = scheduler.metrics(selected)

    assert len(selected) == 4
    assert len({client.client_id for client in selected}) == 4
    assert metrics["per_cluster_selected"] == {"0": 1, "1": 3}
    assert {client.client_id: client.pred_cluster for client in clients} == original_clusters


def test_balanced_scheduler_capacity_aware_multiple_undersized_clusters():
    scheduler = BalancedClusterRoundRobinScheduler(make_clients({0: 1, 1: 1, 2: 8}), clients_per_round=6, seed=7)
    selected = scheduler.sample_round()
    metrics = scheduler.metrics(selected)

    assert len(selected) == 6
    assert len({client.client_id for client in selected}) == 6
    assert metrics["per_cluster_selected"] == {"0": 1, "1": 1, "2": 4}
    assert metrics["per_cluster_budget"] == {"0": 1, "1": 1, "2": 4}


def test_balanced_scheduler_can_select_all_clients_when_k_equals_total():
    clients = make_clients({0: 1, 1: 1, 2: 3})
    scheduler = BalancedClusterRoundRobinScheduler(clients, clients_per_round=len(clients), seed=7)
    selected = scheduler.sample_round()
    metrics = scheduler.metrics(selected)

    assert len(selected) == len(clients)
    assert len({client.client_id for client in selected}) == len(clients)
    assert metrics["per_cluster_selected"] == {"0": 1, "1": 1, "2": 3}


def test_balanced_scheduler_leftover_rotation_not_fixed_to_one_cluster():
    scheduler = BalancedClusterRoundRobinScheduler(make_clients({0: 1, 1: 9, 2: 14, 3: 16}), clients_per_round=8, seed=7)
    budgets = []
    for _ in range(4):
        selected = scheduler.sample_round()
        metrics = scheduler.metrics(selected)
        assert len(selected) == 8
        assert len({client.client_id for client in selected}) == 8
        assert metrics["per_cluster_selected"]["0"] == 1
        budgets.append(metrics["per_cluster_budget"])

    assert budgets[0] == {"0": 1, "1": 3, "2": 2, "3": 2}
    assert budgets[2] == {"0": 1, "1": 2, "2": 3, "3": 2}
    assert budgets[3] == {"0": 1, "1": 2, "2": 2, "3": 3}


# 验证 RandomScheduler 只按总预算随机选择且不强制覆盖所有 cluster。
def test_random_scheduler_does_not_force_coverage():
    clients = make_clients({0: 9, 1: 1})
    original_clusters = {client.client_id: client.pred_cluster for client in clients}
    scheduler = RandomScheduler(clients, clients_per_round=2, seed=1)
    seen_missing = False
    for _ in range(20):
        selected = scheduler.sample_round()
        assert len(selected) == 2
        assert len({client.client_id for client in selected}) == 2
        if scheduler.metrics(selected)["coverage"] < 1.0:
            seen_missing = True
    assert seen_missing
    assert {client.client_id: client.pred_cluster for client in clients} == original_clusters


# 验证 same-label binding 默认仍要求覆盖全部 required slots。
def test_same_label_binding_requires_complete_slots():
    batches = [
        ClientActivationBatch("a", 0, torch.tensor([[1.0, 0.0], [2.0, 0.0]]), torch.tensor([0, 1])),
        ClientActivationBatch("b", 1, torch.tensor([[0.0, 1.0], [0.0, 2.0]]), torch.tensor([0, 1])),
    ]
    pseudo = build_label_random_pseudo_batch(batches, [0, 1], batch_size=4, generator=torch.Generator().manual_seed(0))
    assert pseudo is not None
    assert pseudo.labels.shape[0] == 4
    assert sorted(pseudo.slot_activations) == [0, 1]
    assert pseudo.slot_activations[0].shape[0] == pseudo.labels.shape[0]
    assert build_label_random_pseudo_batch(batches[:1], [0, 1], batch_size=4) is None


# 验证缺失 cluster 可被 zero slot 填充，真实 source client 只来自被调度 cluster。
def test_same_label_binding_zero_fills_missing_slots_when_requested():
    batches = [
        ClientActivationBatch("a", 0, torch.tensor([[1.0, 0.0], [2.0, 0.0]]), torch.tensor([0, 1])),
    ]
    pseudo = build_label_random_pseudo_batch(
        batches,
        [0, 1],
        batch_size=4,
        generator=torch.Generator().manual_seed(0),
        allow_missing_clusters_with_zero=True,
    )
    assert pseudo is not None
    assert sorted(pseudo.slot_activations) == [0, 1]
    assert torch.equal(pseudo.slot_activations[1], torch.zeros_like(pseudo.slot_activations[1]))
    assert pseudo.source_client_ids[1] == []


# 提供最小可训练 split-learning client fixture。
class TinySplitClient:
    # 初始化最小 encoder 和本地 batch。
    def __init__(self, client_id, pred_cluster, x, y):
        self.client_id = client_id
        self.pred_cluster = pred_cluster
        self.hidden_modality_id = pred_cluster
        self.x = x
        self.y = y
        self.encoder = nn.Linear(3, 4)
        self.optimizer = torch.optim.SGD(self.encoder.parameters(), lr=0.1)

    # 返回固定 batch，保证测试 deterministic。
    def sample_batch(self):
        return self.x, self.y, None

    # 执行 client forward 并返回 detached server activation。
    def forward_detached(self, x, lengths=None):
        z_client = self.encoder(x)
        z_server = z_client.detach().requires_grad_(True)
        return z_client, z_server

    # 接收 server activation gradient 并更新本地 encoder。
    def backward_from_server(self, z_client, grad):
        self.optimizer.zero_grad()
        z_client.backward(grad)
        self.optimizer.step()


# 验证 server 和 client 参数都通过真实 split-learning 梯度发生更新。
def test_split_learning_gradient_updates_server_and_clients():
    torch.manual_seed(0)
    clients = [
        TinySplitClient("c0", 0, torch.randn(5, 3), torch.tensor([0, 1, 0, 1, 0])),
        TinySplitClient("c1", 1, torch.randn(5, 3), torch.tensor([0, 1, 0, 1, 0])),
    ]
    cfg = {
        "num_classes": 2,
        "encoder_hidden_dim": 4,
        "training": {"batch_size": 5, "client_lr": 0.1, "server_lr": 0.1},
        "binding": {"batch_size": 4},
        "fusion": {"training_objective": "label_random_ce", "adapter_dim": 4, "hidden_dim": 8, "num_layers": 1},
    }
    server = ConcatMLPFusionServer([0, 1], 4, 2, cfg)
    optimizer = torch.optim.SGD(server.parameters(), lr=0.1)
    server_before = copy.deepcopy(server.state_dict())
    client_before = [copy.deepcopy(client.encoder.state_dict()) for client in clients]
    metrics = _train_local_step(server, optimizer, clients, [0, 1], cfg)
    assert metrics["pseudo_batch_size"] > 0
    assert any(not torch.equal(server_before[key], value) for key, value in server.state_dict().items())
    for before, client in zip(client_before, clients):
        assert any(not torch.equal(before[key], value) for key, value in client.encoder.state_dict().items())


# 验证缺失 cluster 时训练使用 zero-fill 而不是跳过整轮。
def test_split_learning_zero_fills_missing_cluster_slot():
    torch.manual_seed(0)
    clients = [
        TinySplitClient("c0", 0, torch.randn(5, 3), torch.tensor([0, 1, 0, 1, 0])),
    ]
    cfg = {
        "num_classes": 2,
        "encoder_hidden_dim": 4,
        "training": {"batch_size": 5, "client_lr": 0.1, "server_lr": 0.1},
        "binding": {"batch_size": 4},
        "fusion": {"training_objective": "label_random_ce", "adapter_dim": 4, "hidden_dim": 8, "num_layers": 1},
    }
    server = ConcatMLPFusionServer([0, 1], 4, 2, cfg)
    optimizer = torch.optim.SGD(server.parameters(), lr=0.1)
    metrics = _train_local_step(server, optimizer, clients, [0, 1], cfg)
    assert metrics["pseudo_batch_size"] > 0
    assert metrics["empty_binding_local_step"] == 0
    assert metrics["cluster_slot_coverage"] == 0.5


# 验证真实模态覆盖率按 hidden_modality_id 计算，而不是按 pred_cluster 计算。
def test_round_modality_metrics_use_hidden_modality_id():
    clients = [
        TinySplitClient("c0", 0, torch.randn(2, 3), torch.tensor([0, 1])),
        TinySplitClient("c1", 0, torch.randn(2, 3), torch.tensor([0, 1])),
    ]
    clients[0].hidden_modality_id = 0
    clients[1].hidden_modality_id = 2
    metrics = _round_modality_metrics(clients, [0, 1, 2])
    assert metrics["covered_modality_count"] == 2
    assert metrics["modality_coverage"] == pytest.approx(2 / 3)
    assert metrics["full_modality_coverage"] == 0


# 验证训练日志中的 weighted loss components 能重构 total objective。
def test_loss_decomposition_reconstructs_total_objective():
    torch.manual_seed(1)
    clients = [
        TinySplitClient("c0", 0, torch.randn(6, 3), torch.tensor([0, 1, 0, 1, 0, 1])),
        TinySplitClient("c1", 1, torch.randn(6, 3), torch.tensor([0, 1, 0, 1, 0, 1])),
    ]
    cfg = {
        "num_classes": 2,
        "encoder_hidden_dim": 4,
        "training": {"batch_size": 6, "client_lr": 0.1, "server_lr": 0.1},
        "binding": {"batch_size": 4},
        "fusion": {
            "training_objective": "mmbind_weighted_contrastive",
            "adapter_dim": 4,
            "hidden_dim": 8,
            "num_layers": 1,
            "mmbind": {
                "temperature": 0.2,
                "contrastive_weight": 0.3,
                "heterogeneous_ce_weight": 0.4,
            },
        },
    }
    server = ConcatMLPFusionServer([0, 1], 4, 2, cfg)
    optimizer = torch.optim.SGD(server.parameters(), lr=0.1)

    metrics = _train_local_step(server, optimizer, clients, [0, 1], cfg)

    total = float(metrics["loss"])
    weighted_sum = (
        float(metrics["weighted_classification_loss"])
        + float(metrics["weighted_contrastive_loss"])
        + float(metrics["weighted_heterogeneous_loss"])
    )
    assert metrics["pseudo_batch_size"] > 0
    assert abs(total - weighted_sum) < 1.0e-6
