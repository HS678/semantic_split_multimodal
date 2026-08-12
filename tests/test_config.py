from pathlib import Path

import pytest

from MSL.utils.config import (
    apply_experiment_overrides,
    load_config,
    normalize_experiment_config,
    save_config_artifacts,
    split_protocol_for_fold,
    write_config,
)
from MSL.utils.experiment_args import load_experiment_config_from_args
from MSL.utils.results import (
    cluster_assignment_scope,
    experiment_config_signature,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Args:
    def __init__(self, **kwargs):
        defaults = {
            "config": None,
            "dataset": "mhealth",
            "fold": None,
            "split_protocol": None,
            "clients": None,
            "global_rounds": None,
            "client_lr": None,
            "server_lr": None,
            "batch_size": None,
            "eval_batch_size": None,
            "clients_per_cluster_per_round": None,
            "pretrain_epochs": None,
            "pretrain_lr": None,
            "fingerprint_type": None,
            "fusion_training_objective": None,
            "cluster_assignment_source": None,
            "scheduler": None,
            "seed": None,
            "print_config": False,
        }
        defaults.update(kwargs)
        self.__dict__.update(defaults)


def test_all_dataset_configs_are_ini_style_and_use_pred_cluster_for_mainline():
    paths = {
        "uci_har": "configs/MSL/uci_har.config",
        "mhealth": "configs/MSL/mhealth.config",
        "pamap2": "configs/MSL/pamap2.config",
        "iemocap": "configs/MSL/iemocap.config",
    }
    for dataset, relative_path in paths.items():
        path = PROJECT_ROOT / relative_path
        cfg = normalize_experiment_config(load_config(path))
        assert cfg["dataset"]["type"] == dataset
        assert cfg["base_dir"] == "./results/MSL"
        assert cfg["training"]["cluster_assignment_source"] == "pred_cluster"
        assert "global_rounds" in cfg["training"]
        assert cfg["partition"]["clients_per_modality"] == 10
        assert cfg["cluster"]["method"] == "adaptive_isodata"
        assert cfg["evaluation"]["run_test"] is True


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
    config_dir = PROJECT_ROOT / "configs" / "MSL"
    for dataset in ["uci_har", "mhealth", "pamap2"]:
        cfg = normalize_experiment_config(load_config(config_dir / f"{dataset}.config"))
        assert cfg["dataset"]["type"] == dataset
        assert cfg["fusion"]["training_objective"] in {
            "label_random_ce",
            "mmbind_weighted_contrastive",
        }
        assert cfg["training"]["cluster_assignment_source"] == "pred_cluster"
        assert "validation_enabled" not in cfg["training"]
        assert cfg["evaluation"]["run_test"] is True

    for fold in range(1, 6):
        cfg = normalize_experiment_config(load_config(config_dir / "iemocap.config"))
        cfg = apply_experiment_overrides(cfg, fold=fold)
        assert cfg["dataset"]["split_protocol"] == f"session_5fold_loso_fold{fold}"
        assert "train_sessions" not in cfg["dataset"]
        assert "test_sessions" not in cfg["dataset"]
        assert "validation_sessions" not in cfg["dataset"]


def test_fold_override_generates_dataset_protocols_without_fold_configs():
    cases = {
        "mhealth": (5, "subject_5fold_fold5"),
        "pamap2": (9, "subject_9fold_loso_fold9"),
        "iemocap": (5, "session_5fold_loso_fold5"),
    }
    for dataset, (fold, expected_protocol) in cases.items():
        cfg = normalize_experiment_config(load_config(PROJECT_ROOT / "configs" / "MSL" / f"{dataset}.config"))
        overridden = apply_experiment_overrides(cfg, fold=fold)
        assert split_protocol_for_fold(dataset, fold) == expected_protocol
        assert overridden["dataset"]["split_protocol"] == expected_protocol
        assert overridden["runtime_overrides"]["fold"] == fold


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
    cfg = normalize_experiment_config(load_config(PROJECT_ROOT / "configs" / "MSL" / "uci_har.config"))
    signature = experiment_config_signature(cfg)
    changed_runtime = {**cfg, "seed": 505, "base_dir": "/tmp/elsewhere"}
    assert experiment_config_signature(changed_runtime) == signature

    changed_objective = {
        **cfg,
        "fusion": {**cfg["fusion"], "training_objective": "label_random_ce"},
    }
    assert experiment_config_signature(changed_objective) != signature
    assert cluster_assignment_scope(cfg) == "predicted_cluster"


def test_baseline_configs_write_under_results_and_reuse_msl_artifacts_by_default():
    config_dir = PROJECT_ROOT / "configs" / "baseline" / "randomSL"
    for dataset in ["uci_har", "mhealth", "pamap2", "iemocap"]:
        cfg = normalize_experiment_config(load_config(config_dir / f"{dataset}.config"))
        assert cfg["base_dir"] == "./results/baseline/randomSL"
        assert cfg["training"]["cluster_assignment_source"] == "pred_cluster"
        assert "global_rounds" in cfg["training"]
        assert "stage3" not in cfg


def test_dataset_arg_loads_full_defaults_and_cli_overrides():
    args = Args(
        dataset="mhealth",
        fold=3,
        clients=20,
        global_rounds=50,
        client_lr=0.0007,
        fusion_training_objective="label_random_ce",
    )
    cfg, source_path = load_experiment_config_from_args(args)
    assert source_path is None
    assert cfg["dataset"]["type"] == "mhealth"
    assert cfg["dataset"]["split_protocol"] == "subject_5fold_fold3"
    assert cfg["partition"]["clients_per_modality"] == 20
    assert cfg["training"]["global_rounds"] == 50
    assert cfg["training"]["client_lr"] == 0.0007
    assert cfg["fusion"]["training_objective"] == "label_random_ce"
    assert cfg["model"]["encoder"]["type"] == "temporal_conv_gru"


def test_baseline_dataset_arg_defaults_to_random_scheduler():
    cfg, source_path = load_experiment_config_from_args(Args(dataset="pamap2", fold=2), baseline=True)
    assert source_path is None
    assert cfg["base_dir"] == "./results/baseline/randomSL"
    assert cfg["dataset"]["split_protocol"] == "subject_9fold_loso_fold2"
    assert cfg["training"]["scheduler"] == "random"
