from pathlib import Path

import pytest

from MSL.utils.config import (
    load_config,
    save_config_artifacts,
    write_config,
)
from MSL.utils.results import (
    cluster_assignment_scope,
    experiment_config_signature,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_all_dataset_configs_are_ini_style_and_use_true_cluster_for_current_development():
    paths = {
        "uci_har": "configs/uci_har.config",
        "mhealth": "configs/mhealth/fold1.config",
        "pamap2": "configs/pamap2/fold1.config",
        "iemocap": "configs/iemocap/fold1.config",
    }
    for dataset, relative_path in paths.items():
        path = PROJECT_ROOT / relative_path
        cfg = load_config(path)
        assert cfg["dataset"]["type"] == dataset
        assert cfg["training"]["cluster_assignment_source"] == "true_cluster"
        assert cfg["stage2"]["stage1_dir"]
        assert cfg["stage3"]["stage1_dir"]
        assert cfg["stage3"]["stage2_dir"]
        assert cfg["stage3"]["attempt"] == 1


def test_config_round_trip_preserves_nested_types(tmp_path):
    cfg = {
        "seed": 101,
        "device": "cpu",
        "dataset": {"type": "demo", "subjects": [1, 2], "normalize": True},
        "cluster": {"known_k": None, "adaptive": {"epsilon": 1.0e-8}},
    }
    path = write_config(cfg, tmp_path / "demo.config")
    assert load_config(path) == cfg


def test_none_is_preserved_as_enum_while_null_is_empty(tmp_path):
    path = tmp_path / "enums.config"
    path.write_text(
        "[config]\nclass_weighting=none\nknown_k=null\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg["class_weighting"] == "none"
    assert cfg["known_k"] is None


def test_config_extends_deep_merges_relative_parent(tmp_path):
    parent = tmp_path / "parent.config"
    child = tmp_path / "nested" / "child.config"
    child.parent.mkdir()
    parent.write_text(
        "[config]\nseed=42\n[training]\nclient_lr=0.001\nserver_lr=0.001\n",
        encoding="utf-8",
    )
    child.write_text(
        "[config]\nextends=../parent.config\nseed=101\n[training]\nclient_lr=0.0002\n",
        encoding="utf-8",
    )

    cfg = load_config(child)
    assert cfg["seed"] == 101
    assert cfg["training"] == {"client_lr": 0.0002, "server_lr": 0.001}
    assert "extends" not in cfg


def test_MSL_configs_resolve_expected_protocols_and_objective():
    config_dir = PROJECT_ROOT / "configs"
    for dataset in ["uci_har", "mhealth", "pamap2"]:
        relative = (
            f"{dataset}.config"
            if dataset == "uci_har"
            else f"{dataset}/fold1.config"
        )
        cfg = load_config(config_dir / relative)
        assert cfg["dataset"]["type"] == dataset
        assert cfg["fusion"]["training_objective"] == "mmbind_weighted_contrastive"
        assert cfg["training"]["cluster_assignment_source"] == "true_cluster"
        assert cfg["training"].get("validation_enabled") is False
        assert cfg["evaluation"]["run_test"] is True

    for fold in range(1, 6):
        cfg = load_config(config_dir / "iemocap" / f"fold{fold}.config")
        assert cfg["dataset"]["test_sessions"] == [fold]
        assert set(cfg["dataset"]["train_sessions"]) == set(range(1, 6)) - {fold}
        assert cfg["dataset"]["validation_sessions"] == []
        assert cfg["dataset"]["split_strategy"] == "fixed_session_split_v1"
        assert cfg["dataset"]["split_protocol"] == f"session_5fold_loso_fold{fold}_v1"


def test_yaml_extension_is_rejected(tmp_path):
    path = tmp_path / "legacy.yaml"
    path.write_text("seed: 42\n", encoding="utf-8")
    with pytest.raises(ValueError, match=".config"):
        load_config(path)


def test_config_artifacts_preserve_source_bytes_and_resolved_snapshot(tmp_path):
    source = tmp_path / "input.config"
    source.write_text("# original comment\n[config]\nseed=42\n", encoding="utf-8")
    resolved = {"seed": 101, "stage3": {"attempt": 2}}
    artifacts = save_config_artifacts(source, resolved, tmp_path / "run")

    assert Path(artifacts["source_config"]).read_bytes() == source.read_bytes()
    assert load_config(artifacts["resolved_config"]) == resolved


def test_signature_excludes_seed_attempt_and_paths_but_tracks_training_changes():
    cfg = load_config(PROJECT_ROOT / "configs" / "uci_har.config")
    signature = experiment_config_signature(cfg)
    changed_runtime = {
        **cfg,
        "seed": 505,
        "stage3": {**cfg["stage3"], "attempt": 9, "output_root": "/tmp/elsewhere"},
    }
    assert experiment_config_signature(changed_runtime) == signature

    changed_objective = {
        **cfg,
        "fusion": {**cfg["fusion"], "training_objective": "label_random_ce"},
    }
    assert experiment_config_signature(changed_objective) != signature
    assert cluster_assignment_scope(cfg) == "oracle_true_cluster"
