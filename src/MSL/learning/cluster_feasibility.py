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
