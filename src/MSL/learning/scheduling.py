import random
from collections import Counter, defaultdict


# 从客户端对象读取预测簇编号。
def _client_cluster(client):
    return int(client.pred_cluster)


# 从客户端对象读取客户端编号。
def _client_id(client):
    return str(client.client_id)


# 记录调度器共享的参与次数和覆盖率统计。
class _SchedulerMetricsMixin:
    @property
    def clients_per_round(self):
        raise NotImplementedError

    # 统计本轮选择的 cluster coverage 和参与公平性。
    def metrics(self, selected):
        selected_clusters = {_client_cluster(client) for client in selected}
        counts = [self.participation[_client_id(client)] for client in self.clients]
        fairness = 1.0
        if counts and sum(value * value for value in counts) > 0:
            fairness = (sum(counts) ** 2) / (len(counts) * sum(value * value for value in counts))
        per_cluster_selected = Counter(_client_cluster(client) for client in selected)
        return {
            "coverage": float(len(selected_clusters) / max(1, len(self.cluster_ids))),
            "participation_fairness": float(fairness),
            "round": int(self.round_index),
            "clients_per_round": int(self.clients_per_round),
            "clients_per_cluster_per_round": int(getattr(self, "clients_per_cluster_per_round", 0)),
            "per_cluster_selected": {
                str(cluster_id): int(per_cluster_selected.get(cluster_id, 0))
                for cluster_id in self.cluster_ids
            },
        }


# 根据预测簇从每个簇中无放回地调度 r 个客户端。
class BalancedClusterRoundRobinScheduler:
    """Per-cluster random round-robin sampling without replacement."""

    def __init__(self, clients, clients_per_cluster_per_round, seed=0):
        self.clients = list(clients)
        self.clients_per_cluster_per_round = int(clients_per_cluster_per_round)
        if self.clients_per_cluster_per_round <= 0:
            raise ValueError("clients_per_cluster_per_round must be positive.")
        self.rng = random.Random(int(seed))
        self.round_index = 0
        self.participation = Counter()
        self.by_cluster = defaultdict(list)
        for client in self.clients:
            self.by_cluster[_client_cluster(client)].append(client)
        if not self.by_cluster:
            raise ValueError("Balanced scheduler requires at least one predicted cluster.")
        for cluster_id, group in self.by_cluster.items():
            if len(group) < self.clients_per_cluster_per_round:
                raise ValueError(
                    "clients_per_cluster_per_round cannot exceed the number of clients "
                    f"in pred_cluster {cluster_id}: r={self.clients_per_cluster_per_round}, "
                    f"num_clients={len(group)}"
                )
        self.cluster_ids = sorted(self.by_cluster)
        self.pools = {
            cluster_id: self._new_pool(cluster_id, exclude=set())
            for cluster_id in self.cluster_ids
        }

    @property
    # 返回每轮总客户端预算。
    def clients_per_round(self):
        return self.clients_per_cluster_per_round * len(self.cluster_ids)

    # 为指定簇创建一次洗牌后的候选池。
    def _new_pool(self, cluster_id, exclude):
        pool = [client for client in self.by_cluster[cluster_id] if _client_id(client) not in exclude]
        self.rng.shuffle(pool)
        return pool

    # 从单个预测簇中无放回抽取本轮客户端。
    def _sample_cluster(self, cluster_id):
        selected = []
        selected_ids = set()
        while len(selected) < self.clients_per_cluster_per_round:
            pool = self.pools[cluster_id]
            if not pool:
                self.pools[cluster_id] = self._new_pool(cluster_id, selected_ids)
                pool = self.pools[cluster_id]
            client = pool.pop()
            if _client_id(client) in selected_ids:
                continue
            selected.append(client)
            selected_ids.add(_client_id(client))
        return selected

    # 生成一个 cluster-balanced 训练轮次的客户端列表。
    def sample_round(self):
        selected = []
        for cluster_id in self.cluster_ids:
            selected.extend(self._sample_cluster(cluster_id))
        self.rng.shuffle(selected)
        self.round_index += 1
        for client in selected:
            self.participation[_client_id(client)] += 1
        return selected

    # 统计本轮选择的 cluster coverage 和参与公平性。
    def metrics(self, selected):
        return _SchedulerMetricsMixin.metrics(self, selected)


# 从全体客户端中完全随机无放回选择 r * Q_hat 个客户端。
class RandomScheduler:
    """Uniform random sampling without replacement; no cluster coverage guarantee."""

    def __init__(self, clients, clients_per_round, seed=0):
        self.clients = list(clients)
        self._clients_per_round = int(clients_per_round)
        if self._clients_per_round <= 0:
            raise ValueError("clients_per_round must be positive.")
        if self._clients_per_round > len(self.clients):
            raise ValueError(
                "RandomScheduler cannot sample more distinct clients than available: "
                f"clients_per_round={self._clients_per_round}, num_clients={len(self.clients)}"
            )
        self.rng = random.Random(int(seed))
        self.round_index = 0
        self.participation = Counter()
        self.by_cluster = defaultdict(list)
        for client in self.clients:
            self.by_cluster[_client_cluster(client)].append(client)
        if not self.clients:
            raise ValueError("RandomScheduler requires at least one client.")
        self.cluster_ids = sorted(self.by_cluster)
        self.clients_per_cluster_per_round = 0

    @property
    # 返回每轮随机选择的总客户端数。
    def clients_per_round(self):
        return self._clients_per_round

    # 生成一个完全随机训练轮次的客户端列表。
    def sample_round(self):
        selected = self.rng.sample(self.clients, self._clients_per_round)
        self.round_index += 1
        for client in selected:
            self.participation[_client_id(client)] += 1
        return selected

    # 统计随机选择后的 cluster 覆盖率但不用于补齐选择。
    def metrics(self, selected):
        return _SchedulerMetricsMixin.metrics(self, selected)


# 根据名称构建正式训练使用的调度策略。
def build_scheduler(name, clients, clients_per_cluster_per_round, seed=0):
    key = str(name).lower()
    if key in {"balanced_cluster_round_robin", "balanced"}:
        return BalancedClusterRoundRobinScheduler(clients, clients_per_cluster_per_round, seed)
    if key in {"random", "randomsl"}:
        cluster_ids = sorted({_client_cluster(client) for client in clients})
        clients_per_round = int(clients_per_cluster_per_round) * len(cluster_ids)
        return RandomScheduler(clients, clients_per_round=clients_per_round, seed=seed)
    raise ValueError("Unsupported scheduler. Use 'balanced_cluster_round_robin' or 'random'.")
