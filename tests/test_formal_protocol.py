# 测试正式协议 grid、路径布局、manifest hash 和 pipeline locator。
from pathlib import Path

from experiments.common import (
    DISCOVERY_METHODS,
    TRAINING_METHODS,
    build_protocol_manifest,
    find_clients_dir,
    find_discovery_dir,
    formal_result_dir,
    formal_run_grid,
    protocol_hash,
    resolved_cfg,
    run_type_metadata,
    write_protocol_manifest,
)
from experiments.training import training_run_dir
from MSL.protocol import DATASET_PROTOCOLS
from MSL.utils import partition_signature

# 验证正式协议：无 CV 数据集用 5 seeds，CV 数据集每折固定 seed 42。
def test_formal_run_grid_uses_cv_folds_once_and_repeats_non_cv_seed():
    assert formal_run_grid("uci_har") == [(None, 42), (None, 123), (None, 2025), (None, 3407), (None, 7777)]
    assert formal_run_grid("mhealth") == [(1, 42), (2, 42), (3, 42), (4, 42), (5, 42)]
    assert formal_run_grid("pamap2") == [(fold, 42) for fold in range(1, 9)]
    assert formal_run_grid("iemocap") == [(fold, 42) for fold in range(1, 6)]


# 验证正式 run 总数不再对 CV fold 叠加 5 个随机种子。
def test_formal_run_counts():
    total_run_points = sum(len(formal_run_grid(dataset)) for dataset in DATASET_PROTOCOLS)
    assert total_run_points == 23
    assert total_run_points * len(DISCOVERY_METHODS) == 115
    assert total_run_points * len(TRAINING_METHODS) == 161


# 验证正式 global rounds 只将 IEMOCAP 调整到 300，其他既有默认不被误改。
def test_dataset_protocol_global_rounds():
    assert resolved_cfg("uci_har", None, 42)["training"]["global_rounds"] == 200
    assert resolved_cfg("mhealth", 1, 42)["training"]["global_rounds"] == 200
    assert resolved_cfg("iemocap", 1, 42)["training"]["global_rounds"] == 300
    assert resolved_cfg("pamap2", 1, 42)["training"]["global_rounds"] == 300


# 验证正式 RQ2 使用固定每轮总客户端预算，不再使用每簇客户端预算。
def test_dataset_protocol_clients_per_round_budget():
    assert resolved_cfg("uci_har", None, 42)["training"]["clients_per_round"] == 4
    assert resolved_cfg("mhealth", 1, 42)["training"]["clients_per_round"] == 8
    assert resolved_cfg("pamap2", 1, 42)["training"]["clients_per_round"] == 6
    assert resolved_cfg("iemocap", 1, 42)["training"]["clients_per_round"] == 6
    assert "clients_per_cluster_per_round" not in resolved_cfg("mhealth", 1, 42)["training"]


# 验证 UCI-HAR 的重复 seed 写入 split signature，避免 pipeline 产物互相覆盖。
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
    assert formal_result_dir(root, "discovery", "pamap2", "kmeans3", 2, 42) == (
        root / "discovery" / "pamap2" / "kmeans3" / "fold_02" / "seed_42"
    )
    assert training_run_dir(root, "uci_har", None, 123, "ours", None) == (
        root / "msl" / "uci_har" / "ours" / "fold_00" / "seed_123"
    )
    assert training_run_dir(root, "mhealth", 3, 42, "randomsl", 1) == (
        root / "baselines" / "mhealth" / "randomsl" / "fold_03" / "seed_42" / "rounds_1"
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


def _touch_client_artifacts(path: Path) -> None:
    (path / "train_clients").mkdir(parents=True)
    (path / "test_multimodal.pt").write_bytes(b"")


def _touch_discovery_artifacts(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "pred_cluster.csv").write_text("client_id,cluster_id\n", encoding="utf-8")
    (path / "pretrained_encoders").mkdir()
    (path / "visualization").mkdir()
    (path / "visualization" / "fingerprints.npz").write_bytes(b"")

def test_find_clients_dir_supports_pipeline_layout(tmp_path):
    cfg = resolved_cfg("mhealth", 1, 42)

    partition_name = partition_signature(
        ("acc", "gyro", "mag", "ecg"),
        10,
        cfg["dataset"]["split_protocol"],
    )

    expected = (
        tmp_path
        / "results"
        / "pipeline"
        / "clients"
        / "mhealth"
        / partition_name
    )

    _touch_client_artifacts(expected)

    assert find_clients_dir(tmp_path, cfg) == expected.resolve()

def test_find_discovery_dir_supports_pipeline_layout(tmp_path):
    cfg = resolved_cfg("mhealth", 1, 42)

    partition_name = partition_signature(
        ("acc", "gyro", "mag", "ecg"),
        10,
        cfg["dataset"]["split_protocol"],
    )

    clients_dir = (
        tmp_path
        / "results"
        / "pipeline"
        / "clients"
        / "mhealth"
        / partition_name
    )
    _touch_client_artifacts(clients_dir)

    expected = (
        tmp_path
        / "results"
        / "pipeline"
        / "discovery"
        / "mhealth"
        / partition_name
        / "adaptive_isodata"
    )
    _touch_discovery_artifacts(expected)

    resolved_clients = find_clients_dir(tmp_path, cfg)

    assert resolved_clients == clients_dir.resolve()
    assert (
        find_discovery_dir(
            tmp_path,
            resolved_clients,
            "adaptive_isodata",
        )
        == expected.resolve()
    )
