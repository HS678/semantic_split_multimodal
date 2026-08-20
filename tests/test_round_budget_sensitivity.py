# 测试 RQ2 round-budget sensitivity 实验的 manifest、路径和聚合逻辑。
from pathlib import Path

import pytest

from experiments.sensitivity.round_budget import (
    DATASET_ROUND_BUDGETS,
    DEFAULT_METHODS,
    DEFAULT_ROUND_BUDGETS,
    SENSITIVITY_PROTOCOL_VERSION,
    aggregate,
    budget_auc,
    build_run_grid,
    build_sensitivity_manifest,
    load_formal_protocol_manifest,
    resolve_dataset_round_budgets,
    resolve_datasets,
    rounds_to_target,
    sensitivity_protocol_hash,
    sensitivity_run_dir,
)


# 验证 sensitivity 结果目录与正式 RQ2 msl/baselines 目录隔离。
def test_sensitivity_run_dir_is_isolated_from_formal_rq2_layout():
    path = sensitivity_run_dir(Path("results/sensitivity/round_budget"), "mhealth", "ours", 50, 2, 42)
    assert path == Path("results/sensitivity/round_budget/mhealth/ours/rounds_50/fold_02/seed_42")
    assert "msl" not in path.parts
    assert "baselines" not in path.parts


# 验证第一阶段默认 run grid 复用 MHEALTH 正式 5 folds / seed 42。
def test_default_mhealth_run_grid_uses_formal_fold_protocol():
    grid = build_run_grid("mhealth", seeds=None, folds=None)
    assert grid == [(1, 42), (2, 42), (3, 42), (4, 42), (5, 42)]


# 验证多数据集默认 budget 包含各自正式训练终点。
def test_dataset_round_budget_defaults_follow_formal_endpoints():
    assert DATASET_ROUND_BUDGETS["uci_har"] == (25, 50, 100, 150, 200)
    assert DATASET_ROUND_BUDGETS["mhealth"] == (25, 50, 100, 150, 200)
    assert DATASET_ROUND_BUDGETS["pamap2"] == (25, 50, 100, 150, 200, 250, 300)
    assert DATASET_ROUND_BUDGETS["iemocap"] == (25, 50, 100, 150, 200, 250, 300)


# 验证 CLI 数据集解析保留单数据集兼容并支持多数据集去重。
def test_resolve_datasets_supports_single_and_multi_dataset_modes():
    assert resolve_datasets("mhealth", None) == ["mhealth"]
    assert resolve_datasets("mhealth", ["uci_har", "pamap2", "uci_har"]) == ["uci_har", "pamap2"]


# 验证 round budget override 会应用到所有选中数据集。
def test_round_budget_override_applies_to_all_selected_datasets():
    budgets = resolve_dataset_round_budgets(["uci_har", "iemocap"], [25, 50])
    assert budgets == {"uci_har": [25, 50], "iemocap": [25, 50]}


# 验证 sensitivity manifest hash 不受 timestamp 和输出路径影响。
def test_sensitivity_manifest_hash_is_deterministic():
    run_grids = {"mhealth": [(1, 42), (2, 42)]}
    dataset_round_budgets = {"mhealth": list(DEFAULT_ROUND_BUDGETS)}
    first = build_sensitivity_manifest(
        results_root=Path("results/sensitivity/round_budget"),
        datasets=["mhealth"],
        methods=DEFAULT_METHODS,
        dataset_round_budgets=dataset_round_budgets,
        run_grids=run_grids,
    )
    second = build_sensitivity_manifest(
        results_root=Path("other"),
        datasets=["mhealth"],
        methods=DEFAULT_METHODS,
        dataset_round_budgets=dataset_round_budgets,
        run_grids=run_grids,
    )
    assert first["sensitivity_protocol_version"] == SENSITIVITY_PROTOCOL_VERSION
    assert first["sensitivity_protocol_hash"] == second["sensitivity_protocol_hash"]
    first["timestamp"] = "changed"
    first["results_root"] = "changed"
    assert sensitivity_protocol_hash(first) == second["sensitivity_protocol_hash"]


# 验证 sensitivity 引用已完成正式 RQ2 的 protocol manifest，避免当前工作树 git hash 漂移影响正式 hash。
def test_sensitivity_uses_recorded_formal_protocol_manifest():
    formal = load_formal_protocol_manifest()
    manifest = build_sensitivity_manifest(
        results_root=Path("results/sensitivity/round_budget"),
        datasets=["mhealth"],
        methods=DEFAULT_METHODS,
        dataset_round_budgets={"mhealth": list(DEFAULT_ROUND_BUDGETS)},
        run_grids={"mhealth": [(1, 42)]},
    )
    assert manifest["formal_protocol_version"] == formal["protocol_version"]
    assert manifest["formal_protocol_hash"] == formal["protocol_hash"]


# 验证 multi-dataset manifest 为每个 dataset 记录自己的 budget、clients_per_round 和正式总轮数。
def test_sensitivity_manifest_records_per_dataset_budget_protocol():
    manifest = build_sensitivity_manifest(
        results_root=Path("results/sensitivity/round_budget"),
        datasets=["uci_har", "pamap2", "iemocap"],
        methods=DEFAULT_METHODS,
        dataset_round_budgets={
            "uci_har": list(DATASET_ROUND_BUDGETS["uci_har"]),
            "pamap2": list(DATASET_ROUND_BUDGETS["pamap2"]),
            "iemocap": list(DATASET_ROUND_BUDGETS["iemocap"]),
        },
        run_grids={
            "uci_har": [(None, 42), (None, 123)],
            "pamap2": [(1, 42)],
            "iemocap": [(1, 42)],
        },
    )
    assert manifest["datasets"]["uci_har"]["clients_per_round"] == 4
    assert manifest["datasets"]["uci_har"]["formal_global_rounds"] == 200
    assert manifest["datasets"]["pamap2"]["round_budgets"][-1] == 300
    assert manifest["datasets"]["pamap2"]["clients_per_round"] == 6
    assert manifest["datasets"]["pamap2"]["formal_global_rounds"] == 300
    assert manifest["datasets"]["iemocap"]["round_budgets"][-1] == 300
    assert manifest["datasets"]["iemocap"]["clients_per_round"] == 6
    assert manifest["datasets"]["iemocap"]["formal_global_rounds"] == 300


# 验证 aggregate 按 method 和 round budget 计算 mean/std/count。
def test_aggregate_groups_by_method_and_round_budget():
    records = [
        {
            "status": "success",
            "dataset": "mhealth",
            "method": "ours",
            "round_budget": 25,
            "accuracy": 0.8,
            "macro_f1": 0.7,
            "weighted_f1": 0.75,
            "final_loss": 1.0,
            "modality_full_coverage_rate": 1.0,
            "modality_coverage_mean": 1.0,
        },
        {
            "status": "success",
            "dataset": "mhealth",
            "method": "ours",
            "round_budget": 25,
            "accuracy": 0.9,
            "macro_f1": 0.8,
            "weighted_f1": 0.85,
            "final_loss": 0.8,
            "modality_full_coverage_rate": 1.0,
            "modality_coverage_mean": 1.0,
        },
        {
            "status": "failed",
            "dataset": "mhealth",
            "method": "randomsl",
            "round_budget": 25,
        },
    ]
    rows = aggregate(records)
    ours = [row for row in rows if row["method"] == "ours"][0]
    randomsl = [row for row in rows if row["method"] == "randomsl"][0]

    assert ours["count"] == 2
    assert ours["failed"] == 0
    assert ours["accuracy_mean"] == pytest.approx(0.85)
    assert ours["accuracy_std"] == pytest.approx(0.05)
    assert ours["modality_full_coverage_mean"] == pytest.approx(1.0)
    assert randomsl["count"] == 0
    assert randomsl["failed"] == 1


# 验证 Budget AUC 使用 performance-vs-round-budget 离散曲线。
def test_budget_auc_uses_trapezoidal_integration_on_budget_axis():
    summary = [
        {"dataset": "mhealth", "method": "ours", "round_budget": 25, "accuracy_mean": 0.5, "macro_f1_mean": 0.4},
        {"dataset": "mhealth", "method": "ours", "round_budget": 50, "accuracy_mean": 0.7, "macro_f1_mean": 0.6},
        {"dataset": "mhealth", "method": "ours", "round_budget": 100, "accuracy_mean": 0.9, "macro_f1_mean": 0.8},
    ]
    rows = budget_auc(summary)
    item = rows[0]
    expected_accuracy_auc = ((50 - 25) * (0.5 + 0.7) / 2 + (100 - 50) * (0.7 + 0.9) / 2) / (100 - 25)
    assert item["accuracy_budget_auc"] == pytest.approx(expected_accuracy_auc)


# 验证 rounds-to-target 返回达到 target 的最小离散 round budget。
def test_rounds_to_target_returns_first_budget_or_not_reached():
    summary = [
        {"dataset": "mhealth", "method": "ours", "round_budget": 25, "accuracy_mean": 0.85},
        {"dataset": "mhealth", "method": "ours", "round_budget": 50, "accuracy_mean": 0.91},
        {"dataset": "mhealth", "method": "randomsl", "round_budget": 25, "accuracy_mean": 0.7},
        {"dataset": "mhealth", "method": "randomsl", "round_budget": 50, "accuracy_mean": 0.8},
    ]
    rows = rounds_to_target(summary, 0.9)
    by_method = {row["method"]: row["rounds_to_target"] for row in rows}
    assert by_method["ours"] == 50
    assert by_method["randomsl"] == "not_reached"
