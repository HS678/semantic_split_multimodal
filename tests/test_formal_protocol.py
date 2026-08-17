from pathlib import Path

from experiments.common import (
    DATASET_DEFAULTS,
    RQ1_METHODS,
    RQ2_METHODS,
    build_protocol_manifest,
    formal_result_dir,
    formal_run_grid,
    protocol_hash,
    resolved_cfg,
    run_type_metadata,
    write_protocol_manifest,
)
from experiments.run_rq2_training import rq2_run_dir


# 验证正式协议：无 CV 数据集用 5 seeds，CV 数据集每折固定 seed 42。
def test_formal_run_grid_uses_cv_folds_once_and_repeats_non_cv_seed():
    assert formal_run_grid("uci_har") == [(None, 42), (None, 123), (None, 2025), (None, 3407), (None, 7777)]
    assert formal_run_grid("mhealth") == [(1, 42), (2, 42), (3, 42), (4, 42), (5, 42)]
    assert formal_run_grid("pamap2") == [(fold, 42) for fold in range(1, 9)]
    assert formal_run_grid("iemocap") == [(fold, 42) for fold in range(1, 6)]


# 验证正式 run 总数不再对 CV fold 叠加 5 个随机种子。
def test_formal_run_counts():
    total_run_points = sum(len(formal_run_grid(dataset)) for dataset in DATASET_DEFAULTS)
    assert total_run_points == 23
    assert total_run_points * len(RQ1_METHODS) == 92
    assert total_run_points * len(RQ2_METHODS) == 138


# 验证 UCI-HAR 的重复 seed 写入 split signature，避免 Stage1/Stage2 产物互相覆盖。
def test_uci_har_repeated_seed_split_signature():
    cfg = resolved_cfg("uci_har", None, 123)
    assert cfg["dataset"]["split_protocol"] == "subject_disjoint_70_30_seed123"


# 显式传入 --seeds 时保留为调试覆盖能力。
def test_explicit_seed_override_keeps_cross_product():
    assert formal_run_grid("pamap2", [1, 2]) == [
        (fold, seed)
        for fold in range(1, 9)
        for seed in [1, 2]
    ]


# 验证正式结果目录使用 dataset/method/fold/seed 层级。
def test_formal_result_directory_layout():
    root = Path("results")
    assert formal_result_dir(root, "rq1", "pamap2", "kmeans3", 2, 42) == (
        root / "rq1" / "pamap2" / "kmeans3" / "fold_02" / "seed_42"
    )
    assert rq2_run_dir(root, "uci_har", None, 123, "ours", None) == (
        root / "rq2" / "uci_har" / "ours" / "fold_00" / "seed_123"
    )
    assert rq2_run_dir(root, "mhealth", 3, 42, "randomsl", 1) == (
        root / "rq2" / "mhealth" / "randomsl" / "fold_03" / "seed_42" / "rounds_1"
    )


# 验证 protocol manifest 包含正式协议 hash，且 hash 不受 timestamp/output path 影响。
def test_protocol_manifest_hash_is_deterministic(tmp_path):
    first = build_protocol_manifest(tmp_path / "results")
    second = build_protocol_manifest(tmp_path / "other_results")
    assert first["protocol_hash"] == second["protocol_hash"]
    first["timestamp"] = "changed"
    first["results_root"] = "changed"
    assert protocol_hash(first) == second["protocol_hash"]


# 验证 protocol manifest 会写入结果根目录。
def test_write_protocol_manifest(tmp_path):
    manifest = write_protocol_manifest(tmp_path)
    path = tmp_path / "protocol_manifest.json"
    assert path.exists()
    assert manifest["protocol_hash"] == protocol_hash(manifest)


# 验证 smoke 结果和 formal 结果在 metadata 中明确区分。
def test_run_type_metadata_marks_smoke_roots():
    assert run_type_metadata(Path("results_smoke")) == {"run_type": "smoke", "formal": False}
    assert run_type_metadata(Path("results")) == {"run_type": "formal", "formal": True}
