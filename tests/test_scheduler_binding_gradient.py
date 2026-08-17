import copy
from dataclasses import dataclass

import pytest
import torch
from torch import nn

from MSL.learning.binding import ClientActivationBatch, build_label_random_pseudo_batch
from MSL.learning.fusion_sl import _train_local_step
from MSL.learning.models import ConcatMLPFusionServer
from MSL.learning.scheduling import BalancedClusterRoundRobinScheduler, RandomScheduler


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


# 验证 cluster-balanced scheduler 每簇正好选择 r 个不同客户端。
def test_balanced_scheduler_exact_r_distinct():
    scheduler = BalancedClusterRoundRobinScheduler(make_clients({0: 3, 1: 3}), 2, seed=7)
    selected = scheduler.sample_round()
    assert len(selected) == 4
    assert len({client.client_id for client in selected}) == 4
    assert scheduler.metrics(selected)["per_cluster_selected"] == {"0": 2, "1": 2}


# 验证 cluster size 不足时不能复制客户端填充。
def test_balanced_scheduler_rejects_small_cluster():
    with pytest.raises(ValueError):
        BalancedClusterRoundRobinScheduler(make_clients({0: 1, 1: 3}), 2, seed=7)


# 验证 RandomScheduler 只按总预算随机选择且不强制覆盖所有 cluster。
def test_random_scheduler_does_not_force_coverage():
    clients = make_clients({0: 9, 1: 1})
    scheduler = RandomScheduler(clients, clients_per_round=2, seed=1)
    seen_missing = False
    for _ in range(20):
        selected = scheduler.sample_round()
        assert len(selected) == 2
        assert len({client.client_id for client in selected}) == 2
        if scheduler.metrics(selected)["coverage"] < 1.0:
            seen_missing = True
    assert seen_missing


# 验证 same-label binding 只构造覆盖全部 required slots 的伪样本。
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


# 提供最小可训练 split-learning client fixture。
class TinySplitClient:
    # 初始化最小 encoder 和本地 batch。
    def __init__(self, client_id, pred_cluster, x, y):
        self.client_id = client_id
        self.pred_cluster = pred_cluster
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
