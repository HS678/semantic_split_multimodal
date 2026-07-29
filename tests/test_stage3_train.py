import csv
import importlib.util
import json
from pathlib import Path

import pytest
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "stage3_train.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("stage3_train", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cfg():
    return {
        "seed": 7,
        "device": "cpu",
        "dataset": {"type": "synthetic_stage3", "split_protocol": "subject_disjoint_tvt_v1"},
        "training": {
            "scheduler": "balanced_cluster_round_robin",
            "global_rounds": 2,
            "validation_every": 1,
            "early_stopping": {"patience": 2, "min_rounds": 1, "min_delta": 0.001},
            "local_steps": 1,
            "batch_size": 4,
            "clients_per_cluster_per_round": 1,
            "server_lr": 0.1,
            "client_lr": 0.1,
        },
        "binding": {"type": "label_random", "batch_size": 4},
        "fusion": {"type": "concat_mlp", "adapter_dim": 8, "hidden_dim": 8},
        "cluster": {"method": "adaptive_isodata", "known_k": None},
        "num_classes": 3,
        "encoder_hidden_dim": 8,
    }


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _stage1_dir(tmp_path, client_ids=("client_000", "client_001"), dataset="synthetic_stage3"):
    stage1 = tmp_path / "partition" / dataset / "m0_1clients_m1_1clients"
    train_dir = stage1 / "train_clients"
    train_dir.mkdir(parents=True)
    _write_csv(
        stage1 / "client_meta.csv",
        ["client_id", "hidden_modality_id", "hidden_modality_name", "num_samples", "encoder_type"],
        [
            {
                "client_id": client_id,
                "hidden_modality_id": idx,
                "hidden_modality_name": f"m{idx}",
                "num_samples": 4,
                "encoder_type": "time_series",
            }
            for idx, client_id in enumerate(client_ids)
        ],
    )
    (stage1 / "partition_config.json").write_text(
        json.dumps({"dataset_type": dataset, "num_clients": len(client_ids)}),
        encoding="utf-8",
    )
    for split_name in ("validation", "test"):
        torch.save(
            {
                "modalities": {"m0": torch.zeros(2, 1), "m1": torch.ones(2, 1)},
                "modality_names": ["m0", "m1"],
                "modality_input_shapes": [[1], [1]],
                "label": torch.tensor([0, 1]),
                "split": split_name,
            },
            stage1 / f"{split_name}_multimodal.pt",
        )
    for idx, client_id in enumerate(client_ids):
        torch.save(
            {
                "client_id": client_id,
                "samples": torch.randn(4, 1),
                "labels": torch.tensor([0, 1, 0, 1]),
                "input_shape": [1],
                "encoder_type": "time_series",
                "hidden_modality_id": idx,
            },
            train_dir / f"{client_id}.pt",
        )
    return stage1


def _stage2_dir(tmp_path, client_ids=("client_000", "client_001"), clusters=(0, 1), dataset="synthetic_stage3"):
    stage2 = tmp_path / "cluster" / dataset / "m0_1clients_m1_1clients" / "adaptive_isodata"
    encoder_dir = stage2 / "pretrained_encoders"
    encoder_dir.mkdir(parents=True)
    _write_csv(
        stage2 / "pred_cluster.csv",
        ["client_id", "pred_cluster"],
        [{"client_id": client_id, "pred_cluster": clusters[idx]} for idx, client_id in enumerate(client_ids)],
    )
    _write_csv(
        stage2 / "true_cluster.csv",
        ["client_id", "true_cluster"],
        [{"client_id": client_id, "true_cluster": idx} for idx, client_id in enumerate(client_ids)],
    )
    metadata = {
        "stage": "stage2_discovery",
        "git_commit": "freeze-sha",
        "run_type": "user_formal",
        "dataset": dataset,
        "partition_signature": "m0_1clients_m1_1clients",
        "cluster_method": "adaptive_isodata",
        "metrics": {
            "method": "adaptive_isodata",
            "discovery_status": "discovery_success",
            "estimated_Q": len(set(clusters)),
            "estimated_num_clusters": len(set(clusters)),
        },
    }
    (stage2 / "stage2_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    for client_id in client_ids:
        torch.save({"client_id": client_id, "state_dict": {}}, encoder_dir / f"{client_id}_encoder.pt")
    return stage2


def _config_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_cfg(), sort_keys=False), encoding="utf-8")
    return path


def _write_success_outputs(result_dir: Path, metrics: dict):
    metrics.setdefault("configured_global_rounds", 2)
    metrics.setdefault("executed_global_rounds", 2)
    metrics.setdefault("test_evaluation_count", 1)
    metrics.setdefault("best_round", 1)
    metrics.setdefault("checkpoint", "best_model.pt")
    metrics.setdefault("selected_by", "validation_macro_f1")
    (result_dir / "train_log.csv").write_text("round,loss\n1,1.0\n", encoding="utf-8")
    (result_dir / "validation_log.csv").write_text(
        "round,eval_status,eval_failure_reason,loss,accuracy,macro_f1,is_best,checks_without_improvement\n"
        "1,success,,1.0,0.5,0.4,1,0\n",
        encoding="utf-8",
    )
    (result_dir / "final_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (result_dir / "best_metrics.json").write_text(
        json.dumps({"best_round": metrics["best_round"], "macro_f1": 0.4}),
        encoding="utf-8",
    )
    (result_dir / "best_model.pt").write_text("checkpoint", encoding="utf-8")
    (result_dir / "last_model.pt").write_text("checkpoint", encoding="utf-8")


def test_build_stage3_run_injects_separate_stage1_stage2_and_outputs(tmp_path):
    script = _load_script()
    stage1 = _stage1_dir(tmp_path)
    stage2 = _stage2_dir(tmp_path)

    cfg, paths = script.build_stage3_run(
        _cfg(),
        stage1_dir=stage1,
        stage2_dir=stage2,
        output_root=tmp_path / "experiments",
        run_id="run_1",
        run_type="user_formal",
    )

    expected_run = (tmp_path / "experiments" / "synthetic_stage3" / "run_1").resolve()
    assert Path(cfg["partition"]["output_dir"]) == stage1.resolve()
    assert Path(cfg["cluster"]["output_dir"]) == stage2.resolve()
    assert Path(cfg["result"]["output_dir"]) == expected_run
    assert Path(cfg["result_model"]["output_dir"]) == expected_run
    assert paths["run_dir"] == expected_run


def test_stage3_refuses_existing_non_empty_run_dir(tmp_path):
    script = _load_script()
    stage1 = _stage1_dir(tmp_path)
    stage2 = _stage2_dir(tmp_path)
    existing = tmp_path / "experiments" / "synthetic_stage3" / "run_1"
    existing.mkdir(parents=True)
    (existing / "old.txt").write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        script.build_stage3_run(_cfg(), stage1, stage2, tmp_path / "experiments", "run_1", "user_formal")


def test_stage3_rejects_run_id_dataset_escape_and_output_overlap(tmp_path):
    script = _load_script()
    stage1 = _stage1_dir(tmp_path)
    stage2 = _stage2_dir(tmp_path)

    for run_id in ["../escape", "a/b", r"a\\b", ""]:
        with pytest.raises(ValueError, match="run_id"):
            script.build_stage3_run(_cfg(), stage1, stage2, tmp_path / "experiments", run_id, "user_formal")

    cfg = _cfg()
    cfg["dataset"]["type"] = "../escape"
    with pytest.raises(ValueError, match="dataset"):
        script.build_stage3_run(cfg, stage1, stage2, tmp_path / "experiments", "run_1", "user_formal")
    with pytest.raises(ValueError, match="overlap Stage1"):
        script.build_stage3_run(_cfg(), stage1, stage2, stage1, "run_1", "user_formal")
    with pytest.raises(ValueError, match="overlap Stage2"):
        script.build_stage3_run(_cfg(), stage1, stage2, stage2, "run_1", "user_formal")


def test_codex_test_output_must_stay_under_codex_results(tmp_path):
    script = _load_script()

    with pytest.raises(ValueError, match="codex_test output_root"):
        script.build_stage3_run(
            _cfg(),
            stage1_dir=tmp_path / "stage1",
            stage2_dir=tmp_path / "stage2",
            output_root=tmp_path / "outside",
            run_id="run_1",
            run_type="codex_test",
        )


def test_audit_accepts_valid_stage1_and_stage2_inputs(tmp_path):
    script = _load_script()
    stage1 = _stage1_dir(tmp_path)
    stage2 = _stage2_dir(tmp_path)

    audit = script.audit_stage3_inputs(_cfg(), stage1, stage2)

    assert audit["stage1"]["num_clients"] == 2
    assert audit["stage2"]["estimated_Q"] == 2
    assert audit["stage2"]["cluster_ids"] == [0, 1]


def test_missing_stage1_file_blocks_training_before_output_creation(monkeypatch, tmp_path):
    script = _load_script()
    config = _config_file(tmp_path)
    stage1 = _stage1_dir(tmp_path)
    stage2 = _stage2_dir(tmp_path)
    (stage1 / "test_multimodal.pt").unlink()
    called = {"train": False}

    monkeypatch.setattr(script, "run_mmbind_fusion_stage3_split_training", lambda *_args: called.update(train=True))
    with pytest.raises(FileNotFoundError, match="test_multimodal"):
        script.main(
            [
                "--config",
                str(config),
                "--stage1-dir",
                str(stage1),
                "--stage2-dir",
                str(stage2),
                "--output-root",
                str(tmp_path / "experiments"),
                "--run-id",
                "run_1",
                "--run-type",
                "user_formal",
            ]
        )

    assert not called["train"]
    assert not (tmp_path / "experiments" / "synthetic_stage3" / "run_1").exists()


def test_missing_validation_file_blocks_training_before_output_creation(monkeypatch, tmp_path):
    script = _load_script()
    config = _config_file(tmp_path)
    stage1 = _stage1_dir(tmp_path)
    stage2 = _stage2_dir(tmp_path)
    (stage1 / "validation_multimodal.pt").unlink()
    called = {"train": False}

    monkeypatch.setattr(script, "run_mmbind_fusion_stage3_split_training", lambda *_args: called.update(train=True))
    with pytest.raises(FileNotFoundError, match="validation_multimodal"):
        script.main(
            [
                "--config",
                str(config),
                "--stage1-dir",
                str(stage1),
                "--stage2-dir",
                str(stage2),
                "--output-root",
                str(tmp_path / "experiments"),
                "--run-id",
                "run_1",
                "--run-type",
                "user_formal",
            ]
        )

    assert not called["train"]
    assert not (tmp_path / "experiments" / "synthetic_stage3" / "run_1").exists()


def test_missing_stage2_file_blocks_training_before_output_creation(monkeypatch, tmp_path):
    script = _load_script()
    config = _config_file(tmp_path)
    stage1 = _stage1_dir(tmp_path)
    stage2 = _stage2_dir(tmp_path)
    (stage2 / "pred_cluster.csv").unlink()
    called = {"train": False}

    monkeypatch.setattr(script, "run_mmbind_fusion_stage3_split_training", lambda *_args: called.update(train=True))
    with pytest.raises(FileNotFoundError, match="pred_cluster"):
        script.main(
            [
                "--config",
                str(config),
                "--stage1-dir",
                str(stage1),
                "--stage2-dir",
                str(stage2),
                "--output-root",
                str(tmp_path / "experiments"),
                "--run-id",
                "run_1",
                "--run-type",
                "user_formal",
            ]
        )

    assert not called["train"]
    assert not (tmp_path / "experiments" / "synthetic_stage3" / "run_1").exists()


def test_true_cluster_is_optional_and_never_gates_training_audit(tmp_path):
    script = _load_script()
    stage1 = _stage1_dir(tmp_path)
    stage2 = _stage2_dir(tmp_path)

    (stage2 / "true_cluster.csv").unlink()
    audit = script.audit_stage3_inputs(_cfg(), stage1, stage2)

    assert audit["stage2"]["true_cluster_path"] is None
    assert audit["stage2"]["true_cluster_audit"]["available"] is False
    assert audit["stage2"]["estimated_Q"] == 2


def test_stage2_metadata_is_optional_and_never_gates_training_audit(tmp_path):
    script = _load_script()
    stage1 = _stage1_dir(tmp_path)
    stage2 = _stage2_dir(tmp_path)

    (stage2 / "stage2_metadata.json").unlink()
    audit = script.audit_stage3_inputs(_cfg(), stage1, stage2)

    assert audit["stage2"]["metadata_path"] is None
    assert audit["stage2"]["stage2_metadata"] is None
    assert audit["stage2"]["discovery_status"] is None
    assert audit["stage2"]["estimated_Q"] == 2


def test_inconsistent_true_cluster_is_recorded_but_does_not_gate_training_audit(tmp_path):
    script = _load_script()
    stage1 = _stage1_dir(tmp_path)
    stage2 = _stage2_dir(tmp_path)
    _write_csv(
        stage2 / "true_cluster.csv",
        ["client_id", "true_cluster"],
        [{"client_id": "audit_only_unknown", "true_cluster": 999}],
    )

    audit = script.audit_stage3_inputs(_cfg(), stage1, stage2)

    assert audit["stage2"]["true_cluster_audit"]["available"] is True
    assert audit["stage2"]["true_cluster_audit"]["client_ids_match_stage1"] is False


def test_malformed_optional_stage2_audit_files_never_gate_training(tmp_path):
    script = _load_script()
    stage1 = _stage1_dir(tmp_path)
    stage2 = _stage2_dir(tmp_path)
    (stage2 / "stage2_metadata.json").write_text("{not-json", encoding="utf-8")
    (stage2 / "true_cluster.csv").write_bytes(b"\xff\xfe\x00")

    audit = script.audit_stage3_inputs(_cfg(), stage1, stage2)

    assert audit["stage2"]["stage2_metadata"] is None
    assert "JSONDecodeError" in audit["stage2"]["metadata_read_error"]
    assert "UnicodeDecodeError" in audit["stage2"]["true_cluster_audit"]["read_error"]
    assert audit["stage2"]["estimated_Q"] == 2


def test_discovery_scores_and_reported_true_q_are_audit_only(tmp_path):
    script = _load_script()
    stage1 = _stage1_dir(tmp_path)
    stage2 = _stage2_dir(tmp_path)
    metadata_path = stage2 / "stage2_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["metrics"].update(
        {
            "discovery_status": "discovery_failure",
            "estimated_Q": 999,
            "true_Q": 2,
            "ACC": 0.0,
            "NMI": 0.0,
            "ARI": -1.0,
        }
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    audit = script.audit_stage3_inputs(_cfg(), stage1, stage2)

    assert audit["stage2"]["estimated_Q"] == 2
    assert audit["stage2"]["reported_estimated_Q"] == 999
    assert audit["stage2"]["reported_estimated_Q_matches_pred_cluster"] is False
    assert audit["stage2"]["discovery_status"] == "discovery_failure"


def test_stage2_client_mismatch_missing_pred_cluster_and_duplicates_are_rejected(tmp_path):
    script = _load_script()
    stage1 = _stage1_dir(tmp_path)
    stage2 = _stage2_dir(tmp_path)

    _write_csv(stage2 / "pred_cluster.csv", ["client_id", "pred_cluster"], [{"client_id": "unknown", "pred_cluster": 0}])
    with pytest.raises(ValueError, match="client IDs mismatch"):
        script.audit_stage3_inputs(_cfg(), stage1, stage2)

    _write_csv(stage2 / "pred_cluster.csv", ["client_id"], [{"client_id": "client_000"}])
    with pytest.raises(ValueError, match="pred_cluster"):
        script.audit_stage3_inputs(_cfg(), stage1, stage2)

    _write_csv(
        stage2 / "pred_cluster.csv",
        ["client_id", "pred_cluster"],
        [{"client_id": "client_000", "pred_cluster": 0}, {"client_id": "client_000", "pred_cluster": 1}],
    )
    with pytest.raises(ValueError, match="duplicate client_id"):
        script.audit_stage3_inputs(_cfg(), stage1, stage2)


def test_mocked_success_records_metadata_and_required_outputs(monkeypatch, tmp_path):
    script = _load_script()
    config = _config_file(tmp_path)
    before = config.read_text(encoding="utf-8")
    stage1 = _stage1_dir(tmp_path)
    stage2 = _stage2_dir(tmp_path)
    seen = {}

    def fake_train(cfg, root, device):
        seen["cfg"] = cfg
        seen["root"] = root
        seen["device"] = device
        result_dir = Path(cfg["result"]["output_dir"])
        metrics = {
            "test_eval_status": "success",
            "test_eval_failure_reason": None,
            "test_accuracy": 0.5,
            "test_macro_f1": 0.4,
            "test_loss": 1.0,
            "effective_global_rounds": 2,
            "configured_global_rounds": 2,
            "executed_global_rounds": 2,
        }
        _write_success_outputs(result_dir, metrics)
        return metrics

    monkeypatch.setattr(script, "select_device", lambda _value: torch.device("cpu"))
    monkeypatch.setattr(script, "run_mmbind_fusion_stage3_split_training", fake_train)
    script.main(
        [
            "--config",
            str(config),
            "--seed",
            "101",
            "--stage1-dir",
            str(stage1),
            "--stage2-dir",
            str(stage2),
            "--output-root",
            str(tmp_path / "experiments"),
            "--run-id",
            "run_1",
            "--run-type",
            "user_formal",
        ]
    )

    assert config.read_text(encoding="utf-8") == before
    run_dir = tmp_path / "experiments" / "synthetic_stage3" / "run_1"
    metadata = json.loads((run_dir / "stage3_metadata.json").read_text(encoding="utf-8"))
    assert seen["root"] == script.ROOT
    assert Path(seen["cfg"]["partition"]["output_dir"]) == stage1.resolve()
    assert Path(seen["cfg"]["cluster"]["output_dir"]) == stage2.resolve()
    assert Path(seen["cfg"]["result"]["output_dir"]) == run_dir.resolve()
    assert seen["cfg"]["seed"] == 101
    assert metadata["status"] == "success"
    assert metadata["git_commit"]
    assert metadata["runtime_seconds"] >= 0
    assert metadata["stage1_dir"] == str(stage1.resolve())
    assert metadata["stage2_dir"] == str(stage2.resolve())
    assert metadata["run_id"] == "run_1"
    assert metadata["seed"] == 101
    assert metadata["config_snapshot"]["seed"] == 101
    assert metadata["cli_arguments"]["seed"] == 101
    assert metadata["estimated_Q"] == 2
    assert metadata["stage2_git_commit"] == "freeze-sha"
    assert not (run_dir / "train_clients").exists()
    assert not (run_dir / "pred_cluster.csv").exists()
    for name in [
        "resolved_config.yaml",
        "train_log.csv",
        "validation_log.csv",
        "final_metrics.json",
        "best_metrics.json",
        "best_model.pt",
        "last_model.pt",
        "training_curves.png",
        "stage3_metadata.json",
    ]:
        assert (run_dir / name).exists()


@pytest.mark.parametrize("discovery_status", ["discovery_failure", None])
def test_discovery_status_never_gates_mocked_trainer(
    monkeypatch,
    tmp_path,
    discovery_status,
):
    script = _load_script()
    config = _config_file(tmp_path)
    stage1 = _stage1_dir(tmp_path)
    stage2 = _stage2_dir(tmp_path)
    metadata_path = stage2 / "stage2_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if discovery_status is None:
        metadata["metrics"].pop("discovery_status", None)
    else:
        metadata["metrics"]["discovery_status"] = discovery_status
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    called = {"train": False}

    def fake_train(cfg, *_args):
        called["train"] = True
        metrics = {
            "test_eval_status": "success",
            "test_eval_failure_reason": None,
            "test_accuracy": 0.5,
            "test_macro_f1": 0.4,
            "test_loss": 1.0,
            "effective_global_rounds": 2,
            "configured_global_rounds": 2,
            "executed_global_rounds": 2,
        }
        _write_success_outputs(Path(cfg["result"]["output_dir"]), metrics)
        return metrics

    monkeypatch.setattr(script, "select_device", lambda _value: torch.device("cpu"))
    monkeypatch.setattr(script, "run_mmbind_fusion_stage3_split_training", fake_train)
    script.main(
        [
            "--config",
            str(config),
            "--stage1-dir",
            str(stage1),
            "--stage2-dir",
            str(stage2),
            "--output-root",
            str(tmp_path / "experiments"),
            "--run-id",
            "run_1",
            "--run-type",
            "user_formal",
        ]
    )

    assert called["train"]
    run_dir = tmp_path / "experiments" / "synthetic_stage3" / "run_1"
    recorded = json.loads((run_dir / "stage3_metadata.json").read_text(encoding="utf-8"))
    assert recorded["status"] == "success"
    assert recorded["stage2_discovery_status"] == discovery_status


def test_mocked_failure_records_failed_metadata(monkeypatch, tmp_path):
    script = _load_script()
    config = _config_file(tmp_path)
    stage1 = _stage1_dir(tmp_path)
    stage2 = _stage2_dir(tmp_path)

    monkeypatch.setattr(script, "select_device", lambda _value: torch.device("cpu"))
    monkeypatch.setattr(script, "run_mmbind_fusion_stage3_split_training", lambda *_args: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        script.main(
            [
                "--config",
                str(config),
                "--stage1-dir",
                str(stage1),
                "--stage2-dir",
                str(stage2),
                "--output-root",
                str(tmp_path / "experiments"),
                "--run-id",
                "run_1",
                "--run-type",
                "user_formal",
            ]
        )

    metadata = json.loads((tmp_path / "experiments" / "synthetic_stage3" / "run_1" / "stage3_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert metadata["failure_reason"] == "boom"


def test_test_evaluation_failure_or_missing_outputs_do_not_record_success(monkeypatch, tmp_path):
    script = _load_script()
    config = _config_file(tmp_path)
    stage1 = _stage1_dir(tmp_path)
    stage2 = _stage2_dir(tmp_path)

    def fake_train(cfg, *_args):
        result_dir = Path(cfg["result"]["output_dir"])
        metrics = {
            "test_eval_status": "failed",
            "test_eval_failure_reason": "mapping_failed",
            "test_accuracy": None,
            "test_macro_f1": None,
            "test_loss": None,
            "effective_global_rounds": 2,
            "configured_global_rounds": 2,
            "executed_global_rounds": 2,
        }
        _write_success_outputs(result_dir, metrics)
        return metrics

    monkeypatch.setattr(script, "select_device", lambda _value: torch.device("cpu"))
    monkeypatch.setattr(script, "run_mmbind_fusion_stage3_split_training", fake_train)
    with pytest.raises(RuntimeError, match="mapping_failed"):
        script.main(
            [
                "--config",
                str(config),
                "--stage1-dir",
                str(stage1),
                "--stage2-dir",
                str(stage2),
                "--output-root",
                str(tmp_path / "experiments"),
                "--run-id",
                "run_1",
                "--run-type",
                "user_formal",
            ]
        )

    metadata = json.loads((tmp_path / "experiments" / "synthetic_stage3" / "run_1" / "stage3_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert metadata["failure_reason"] == "mapping_failed"


def test_latest_run_marker_is_ignored_by_stage3(monkeypatch, tmp_path):
    script = _load_script()
    config = _config_file(tmp_path)
    stage1 = _stage1_dir(tmp_path)
    stage2 = _stage2_dir(tmp_path)
    output_root = tmp_path / "experiments"
    (output_root / "synthetic_stage3").mkdir(parents=True)
    (output_root / "synthetic_stage3" / "latest_run.txt").write_text("old_run", encoding="utf-8")

    def fake_train(cfg, *_args):
        metrics = {
            "test_eval_status": "success",
            "test_eval_failure_reason": None,
            "test_accuracy": 0.5,
            "test_macro_f1": 0.4,
            "test_loss": 1.0,
            "effective_global_rounds": 2,
            "configured_global_rounds": 2,
            "executed_global_rounds": 2,
        }
        _write_success_outputs(Path(cfg["result"]["output_dir"]), metrics)
        return metrics

    monkeypatch.setattr(script, "select_device", lambda _value: torch.device("cpu"))
    monkeypatch.setattr(script, "run_mmbind_fusion_stage3_split_training", fake_train)
    script.main(
        [
            "--config",
            str(config),
            "--stage1-dir",
            str(stage1),
            "--stage2-dir",
            str(stage2),
            "--output-root",
            str(output_root),
            "--run-id",
            "run_1",
            "--run-type",
            "user_formal",
        ]
    )

    assert (output_root / "synthetic_stage3" / "latest_run.txt").read_text(encoding="utf-8") == "old_run"
    assert (output_root / "synthetic_stage3" / "run_1").exists()
    assert not (output_root / "synthetic_stage3" / "old_run").exists()


def test_stage3_source_does_not_use_pipeline_or_training_leakage_tokens():
    text = SCRIPT_PATH.read_text(encoding="utf-8-sig")

    assert "latest_run" not in text
    assert "stage1_partition" not in text
    assert "run_stage2" not in text
    assert "hidden_modality_id" not in text
    assert "hidden_modality_name" not in text
    assert "true_Q" not in text
    assert "num_modalities" not in text
