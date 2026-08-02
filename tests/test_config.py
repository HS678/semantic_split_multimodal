from pathlib import Path

import pytest

from semantic_split_multimodal.utils.config import (
    load_config,
    save_config_artifacts,
    write_config,
)
from semantic_split_multimodal.utils.results import (
    cluster_assignment_scope,
    experiment_config_signature,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_all_dataset_configs_are_ini_style_and_use_true_cluster_for_current_development():
    for dataset in ["uci_har", "mhealth", "pamap2", "cmu_mosei", "iemocap"]:
        path = PROJECT_ROOT / "configs" / f"{dataset}.config"
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
        "fusion": {**cfg["fusion"], "training_objective": "mmbind_weighted_contrastive"},
    }
    assert experiment_config_signature(changed_objective) != signature
    assert cluster_assignment_scope(cfg) == "oracle_true_cluster"
