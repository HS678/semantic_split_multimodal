import argparse
import json
from pathlib import Path

import pytest

from MSL.utils.experiment_args import (
    add_experiment_args,
    apply_experiment_overrides,
    build_experiment_config,
    load_experiment_config_from_args,
    save_resolved_config_artifact,
    stage1_config_snapshot,
    stage2_config_snapshot,
    split_protocol_for_fold,
)
from MSL.utils.results import cluster_assignment_scope, experiment_config_signature


class Args:
    def __init__(self, **kwargs):
        defaults = {
            "dataset": "mhealth",
            "fold": None,
            "split_protocol": None,
            "clients": 10,
            "global_rounds": None,
            "local_steps": 1,
            "client_lr": None,
            "server_lr": None,
            "batch_size": None,
            "eval_batch_size": None,
            "clients_per_cluster_per_round": 2,
            "pretrain_epochs": None,
            "pretrain_lr": None,
            "fingerprint_type": None,
            "fusion_training_objective": "mmbind_weighted_contrastive",
            "cluster_assignment_source": "pred_cluster",
            "scheduler": "balanced_cluster_round_robin",
            "binding_type": "label_random",
            "seed": 42,
            "device": "auto",
            "print_config": False,
        }
        defaults.update(kwargs)
        self.__dict__.update(defaults)


def test_parser_requires_dataset_and_has_no_config_argument():
    parser = argparse.ArgumentParser()
    add_experiment_args(parser)

    args = parser.parse_args(["--dataset", "uci_har"])
    assert args.dataset == "uci_har"

    with pytest.raises(SystemExit):
        parser.parse_args(["--config", "old.config"])


def test_dataset_defaults_are_full_mainline_configs():
    for dataset in ["uci_har", "mhealth", "pamap2", "iemocap"]:
        cfg = build_experiment_config(dataset_type=dataset)
        assert cfg["dataset"]["type"] == dataset
        assert cfg["base_dir"] == "./results/MSL"
        assert cfg["training"]["cluster_assignment_source"] == "pred_cluster"
        assert cfg["training"]["scheduler"] == "balanced_cluster_round_robin"
        assert cfg["training"]["global_rounds"] > 0
        assert cfg["partition"]["clients_per_modality"] == 10
        assert cfg["cluster"]["method"] == "adaptive_isodata"
        assert cfg["cluster"]["known_k"] is None
        assert cfg["evaluation"]["run_test"] is True
        assert "validation_enabled" not in cfg["training"]


def test_fold_override_generates_dataset_protocols_without_fold_configs():
    cases = {
        "mhealth": (5, "subject_5fold_fold5"),
        "pamap2": (8, "subject_8fold_loso_fold8"),
        "iemocap": (5, "session_5fold_loso_fold5"),
    }
    for dataset, (fold, expected_protocol) in cases.items():
        cfg = build_experiment_config(dataset_type=dataset)
        overridden = apply_experiment_overrides(cfg, fold=fold)
        assert split_protocol_for_fold(dataset, fold) == expected_protocol
        assert overridden["dataset"]["split_protocol"] == expected_protocol
        assert overridden["runtime_overrides"]["fold"] == fold


def test_fold_and_split_protocol_are_mutually_exclusive():
    cfg = build_experiment_config(dataset_type="mhealth")
    with pytest.raises(ValueError, match="--fold and --split-protocol"):
        apply_experiment_overrides(cfg, fold=1, split_protocol="subject_5fold_fold1")


def test_signature_excludes_seed_attempt_and_paths_but_tracks_training_changes():
    cfg = build_experiment_config(dataset_type="uci_har")
    signature = experiment_config_signature(cfg)
    changed_runtime = {**cfg, "seed": 505, "base_dir": "/tmp/elsewhere"}
    assert experiment_config_signature(changed_runtime) == signature

    changed_objective = {
        **cfg,
        "fusion": {**cfg["fusion"], "training_objective": "label_random_ce"},
    }
    assert experiment_config_signature(changed_objective) != signature
    assert cluster_assignment_scope(cfg) == "predicted_cluster"


def test_baseline_dataset_defaults_to_random_scheduler():
    cfg = build_experiment_config(dataset_type="pamap2", baseline=True)
    assert cfg["base_dir"] == "./results/baseline/randomSL"
    assert cfg["training"]["cluster_assignment_source"] == "pred_cluster"
    assert cfg["training"]["scheduler"] == "random"
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
    cfg = load_experiment_config_from_args(args)
    assert cfg["dataset"]["type"] == "mhealth"
    assert cfg["dataset"]["split_protocol"] == "subject_5fold_fold3"
    assert cfg["partition"]["clients_per_modality"] == 20
    assert cfg["training"]["global_rounds"] == 50
    assert cfg["training"]["client_lr"] == 0.0007
    assert cfg["fusion"]["training_objective"] == "label_random_ce"
    assert cfg["model"]["encoder"]["type"] == "temporal_conv_gru"


def test_resolved_config_artifact_is_json(tmp_path):
    resolved = {"seed": 101, "stage3": {"attempt": 2}}
    artifacts = save_resolved_config_artifact(resolved, tmp_path / "run")
    path = Path(artifacts["resolved_config"])
    assert path.name == "resolved_config.json"
    assert json.loads(path.read_text(encoding="utf-8")) == resolved


def test_stage1_config_snapshot_keeps_only_partition_inputs():
    cfg = build_experiment_config(dataset_type="uci_har")
    snapshot = stage1_config_snapshot(cfg)

    assert snapshot["config_scope"] == "stage1_partition"
    assert snapshot["dataset"]["type"] == "uci_har"
    assert snapshot["partition"]["clients_per_modality"] == 10
    assert "training" not in snapshot
    assert "pretrain" not in snapshot
    assert "cluster" not in snapshot
    assert "fusion" not in snapshot


def test_stage2_config_snapshot_excludes_stage3_training_inputs():
    cfg = build_experiment_config(dataset_type="pamap2")
    snapshot = stage2_config_snapshot(cfg)

    assert snapshot["config_scope"] == "stage2_discovery"
    assert snapshot["pretrain"]["epochs"] == 5
    assert snapshot["cluster"]["method"] == "adaptive_isodata"
    assert "training" not in snapshot
    assert "binding" not in snapshot
    assert "fusion" not in snapshot
    assert "model" not in snapshot
    assert "evaluation" not in snapshot
