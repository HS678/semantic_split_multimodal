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


from dataclasses import dataclass

import numpy as np


# 表示 cluster scheduling 理论不可行或无法修复。
class InfeasibleClusterSchedulingError(ValueError):
    pass


# 保存 cluster feasibility 检查结果。
@dataclass(frozen=True)
class ClusterFeasibilityReport:
    is_feasible: bool
    cluster_sizes: dict[int, int]
    violating_clusters: list[int]
    num_clients: int
    num_clusters: int
    r: int
    required_clients: int


# 保存一次客户端迁移记录。
@dataclass(frozen=True)
class ClusterMigration:
    client_id: str
    from_cluster: int
    to_cluster: int
    delta: float


# 保存 feasibility repair 后的 assignment 和元数据。
@dataclass(frozen=True)
class ClusterRepairResult:
    raw_assignment: np.ndarray
    training_assignment: np.ndarray
    feasibility_checked: bool
    feasibility_repair_applied: bool
    num_reassigned_clients: int
    cluster_sizes_before: dict[int, int]
    cluster_sizes_after: dict[int, int]
    violating_clusters_before: list[int]
    migrations: list[ClusterMigration]

    # 返回可写入 JSON 的 repair metadata。
    def to_metadata(self) -> dict:
        return {
            "feasibility_checked": bool(self.feasibility_checked),
            "feasibility_repair_applied": bool(self.feasibility_repair_applied),
            "num_reassigned_clients": int(self.num_reassigned_clients),
            "cluster_sizes_before": {str(k): int(v) for k, v in self.cluster_sizes_before.items()},
            "cluster_sizes_after": {str(k): int(v) for k, v in self.cluster_sizes_after.items()},
            "violating_clusters_before": [int(v) for v in self.violating_clusters_before],
            "migrations": [
                {
                    "client_id": migration.client_id,
                    "from_cluster": int(migration.from_cluster),
                    "to_cluster": int(migration.to_cluster),
                    "delta": float(migration.delta),
                }
                for migration in self.migrations
            ],
        }


# 计算每个 cluster 的 client 数量。
def cluster_sizes(assignments) -> dict[int, int]:
    labels = np.asarray(assignments, dtype=int).reshape(-1)
    return {int(label): int(np.sum(labels == label)) for label in sorted(np.unique(labels).tolist())}


# 检查所有预测簇是否满足每轮至少调度 r 个不同客户端的要求。
def validate_cluster_feasibility(fingerprints, cluster_assignments, r: int) -> ClusterFeasibilityReport:
    features = np.asarray(fingerprints, dtype=np.float64)
    labels = np.asarray(cluster_assignments, dtype=int).reshape(-1)
    if features.ndim != 2:
        raise ValueError(f"fingerprints must be a 2D matrix, got shape {features.shape}")
    if labels.ndim != 1:
        raise ValueError("cluster_assignments must be 1D.")
    if int(features.shape[0]) != int(labels.shape[0]):
        raise ValueError(
            "fingerprints and cluster_assignments must have the same number of clients: "
            f"{features.shape[0]} != {labels.shape[0]}"
        )
    if int(r) <= 0:
        raise ValueError("r must be positive.")
    sizes = cluster_sizes(labels)
    num_clients = int(labels.shape[0])
    num_clusters = int(len(sizes))
    required_clients = int(num_clusters * int(r))
    if required_clients > num_clients:
        raise InfeasibleClusterSchedulingError(
            "Cluster scheduling is theoretically infeasible: "
            f"N={num_clients}, K={num_clusters}, r={int(r)}, required_clients={required_clients}"
        )
    violating = [cluster_id for cluster_id, size in sizes.items() if int(size) < int(r)]
    return ClusterFeasibilityReport(
        is_feasible=not violating,
        cluster_sizes=sizes,
        violating_clusters=violating,
        num_clients=num_clients,
        num_clusters=num_clusters,
        r=int(r),
        required_clients=required_clients,
    )


# 计算当前 assignment 下每个 cluster 的 centroid。
def _centroids(features: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    return {
        int(cluster_id): features[labels == int(cluster_id)].mean(axis=0)
        for cluster_id in sorted(np.unique(labels).tolist())
    }


# 选择迁移到目标 undersized cluster 时增量失真最小的 donor client。
def _best_migration(features, labels, client_ids, target_cluster: int, r: int):
    centers = _centroids(features, labels)
    sizes = cluster_sizes(labels)
    candidates = []
    target_center = centers[int(target_cluster)]
    for donor_cluster, size in sizes.items():
        if int(donor_cluster) == int(target_cluster) or int(size) <= int(r):
            continue
        donor_center = centers[int(donor_cluster)]
        for index in np.where(labels == int(donor_cluster))[0].tolist():
            to_target = float(np.sum((features[index] - target_center) ** 2))
            from_donor = float(np.sum((features[index] - donor_center) ** 2))
            delta = float(to_target - from_donor)
            candidates.append((delta, str(client_ids[index]), int(index), int(donor_cluster)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (float(item[0]), item[1], int(item[2]), int(item[3])))
    return candidates[0]


# 在不使用真实模态信息的情况下最小化指纹空间失真并修复过小簇。
def repair_cluster_feasibility(
    fingerprints,
    cluster_assignments,
    r: int,
    client_ids=None,
) -> ClusterRepairResult:
    report = validate_cluster_feasibility(fingerprints, cluster_assignments, r)
    features = np.asarray(fingerprints, dtype=np.float64)
    raw = np.asarray(cluster_assignments, dtype=int).reshape(-1)
    labels = raw.copy()
    if client_ids is None:
        client_ids = [str(index) for index in range(int(labels.shape[0]))]
    client_ids = [str(value) for value in client_ids]
    if len(client_ids) != int(labels.shape[0]):
        raise ValueError("client_ids must have the same length as cluster_assignments.")

    migrations: list[ClusterMigration] = []
    if not report.is_feasible:
        while True:
            current_report = validate_cluster_feasibility(features, labels, r)
            if current_report.is_feasible:
                break
            target_cluster = sorted(
                current_report.violating_clusters,
                key=lambda cluster_id: (current_report.cluster_sizes[int(cluster_id)], int(cluster_id)),
            )[0]
            needed = int(r) - int(current_report.cluster_sizes[int(target_cluster)])
            for _ in range(needed):
                best = _best_migration(features, labels, client_ids, target_cluster, int(r))
                if best is None:
                    raise InfeasibleClusterSchedulingError(
                        "Cluster feasibility repair failed because no donor cluster has size > r."
                    )
                delta, client_id, index, donor_cluster = best
                labels[int(index)] = int(target_cluster)
                migrations.append(
                    ClusterMigration(
                        client_id=str(client_id),
                        from_cluster=int(donor_cluster),
                        to_cluster=int(target_cluster),
                        delta=float(delta),
                    )
                )

    final_report = validate_cluster_feasibility(features, labels, r)
    return ClusterRepairResult(
        raw_assignment=raw,
        training_assignment=labels,
        feasibility_checked=True,
        feasibility_repair_applied=bool(migrations),
        num_reassigned_clients=int(len(migrations)),
        cluster_sizes_before=report.cluster_sizes,
        cluster_sizes_after=final_report.cluster_sizes,
        violating_clusters_before=report.violating_clusters,
        migrations=migrations,
    )
