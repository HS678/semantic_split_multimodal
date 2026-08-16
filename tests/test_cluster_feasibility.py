import inspect

import numpy as np
import pytest

from MSL.learning.cluster_feasibility import (
    InfeasibleClusterSchedulingError,
    repair_cluster_feasibility,
    validate_cluster_feasibility,
)


# 根据 cluster sizes 构造 deterministic fingerprints 和 assignments。
def make_fixture(sizes):
    fingerprints = []
    assignments = []
    client_ids = []
    index = 0
    for cluster_id, size in enumerate(sizes):
        for offset in range(size):
            fingerprints.append([float(cluster_id) * 10.0 + float(offset) * 0.01, float(offset)])
            assignments.append(cluster_id)
            client_ids.append(f"client_{index:03d}")
            index += 1
    return np.asarray(fingerprints, dtype=np.float32), np.asarray(assignments, dtype=int), client_ids


# 验证无需 repair 时 assignment 完全不变。
def test_no_repair_keeps_assignment():
    fingerprints, assignments, client_ids = make_fixture([4, 4, 4])
    result = repair_cluster_feasibility(fingerprints, assignments, r=2, client_ids=client_ids)
    assert not result.feasibility_repair_applied
    assert result.num_reassigned_clients == 0
    assert np.array_equal(result.training_assignment, assignments)


# 验证单个 undersized cluster 会迁移一个 client。
def test_single_undersized_cluster_repaired():
    fingerprints, assignments, client_ids = make_fixture([5, 4, 1])
    result = repair_cluster_feasibility(fingerprints, assignments, r=2, client_ids=client_ids)
    assert result.feasibility_repair_applied
    assert result.num_reassigned_clients == 1
    assert min(result.cluster_sizes_after.values()) >= 2


# 验证多个 undersized clusters 都能被修复。
def test_multiple_undersized_clusters_repaired():
    fingerprints, assignments, client_ids = make_fixture([7, 1, 1, 6])
    result = repair_cluster_feasibility(fingerprints, assignments, r=2, client_ids=client_ids)
    assert result.num_reassigned_clients == 2
    assert min(result.cluster_sizes_after.values()) >= 2


# 验证刚好可行时最终每个 cluster 都正好达到 r。
def test_exactly_feasible_repair():
    fingerprints, assignments, client_ids = make_fixture([6, 1, 1, 1, 1])
    result = repair_cluster_feasibility(fingerprints, assignments, r=2, client_ids=client_ids)
    assert sorted(result.cluster_sizes_after.values()) == [2, 2, 2, 2, 2]


# 验证理论不可行时明确失败。
def test_theoretically_infeasible_raises():
    fingerprints, assignments, _ = make_fixture([5, 1, 1, 1, 1])
    with pytest.raises(InfeasibleClusterSchedulingError):
        validate_cluster_feasibility(fingerprints, assignments, r=2)


# 验证相同输入多次 repair 得到相同 training assignment。
def test_repair_is_deterministic():
    fingerprints, assignments, client_ids = make_fixture([5, 4, 1])
    first = repair_cluster_feasibility(fingerprints, assignments, r=2, client_ids=client_ids)
    second = repair_cluster_feasibility(fingerprints, assignments, r=2, client_ids=client_ids)
    assert np.array_equal(first.training_assignment, second.training_assignment)
    assert first.to_metadata() == second.to_metadata()


# 验证 feasibility API 不接受 true modality 参数。
def test_repair_api_has_no_ground_truth_parameter():
    parameters = inspect.signature(repair_cluster_feasibility).parameters
    forbidden = {"true", "true_modality", "hidden_modality_id", "modality"}
    assert forbidden.isdisjoint(parameters)
