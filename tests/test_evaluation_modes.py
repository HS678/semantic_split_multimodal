import csv
import json
from pathlib import Path

import pytest
import torch

from MSL.models import create_client_encoder
from MSL.training import (
    _curve_evaluation_rounds,
    _evaluate_test_checkpoint,
    _evaluation_mode_spec,
    train_msl_split_learning,
)
import MSL.training as training_mod


def _write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_tiny_run(tmp_path: Path, mode: str, rounds: int = 12):
    partition_dir = tmp_path / "partition"
    cluster_dir = tmp_path / "cluster"
    result_dir = tmp_path / "result"
    model_dir = tmp_path / "checkpoints"
    train_dir = partition_dir / "train_clients"
    encoder_dir = cluster_dir / "pretrained_encoders"
    train_dir.mkdir(parents=True)
    encoder_dir.mkdir(parents=True)

    cfg = {
        "seed": 7,
        "num_classes": 2,
        "encoder_hidden_dim": 4,
        "partition": {"output_dir": str(partition_dir)},
        "cluster": {"output_dir": str(cluster_dir)},
        "result": {"output_dir": str(result_dir)},
        "result_model": {"output_dir": str(model_dir)},
        "model": {"encoder": {"type": "mlp"}},
        "training": {
            "cluster_assignment_source": "pred_cluster",
            "scheduler": "balanced_cluster_round_robin",
            "global_rounds": int(rounds),
            "local_steps": 1,
            "clients_per_round": 2,
            "batch_size": 4,
            "eval_batch_size": 4,
            "client_lr": 0.01,
            "server_lr": 0.01,
        },
        "binding": {"batch_size": 4},
        "fusion": {
            "training_objective": "label_random_ce",
            "adapter_dim": 4,
            "hidden_dim": 8,
            "num_layers": 1,
        },
        "evaluation": {"run_test": True, "evaluation_mode": mode, "eval_every_rounds": 10},
    }

    torch.manual_seed(0)
    rows = []
    for client_index, cluster_id in enumerate([0, 1]):
        client_id = f"client_{client_index:03d}"
        payload = {
            "client_id": client_id,
            "hidden_modality_id": int(cluster_id),
            "samples": torch.randn(8, 3),
            "labels": torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long),
            "encoder_type": "mlp",
            "input_shape": [3],
        }
        torch.save(payload, train_dir / f"{client_id}.pt")
        encoder = create_client_encoder(cfg, input_shape=[3], encoder_type="mlp")
        torch.save(
            {
                "client_id": client_id,
                "encoder_type": "mlp",
                "input_shape": [3],
                "hidden_dim": 4,
                "state_dict": encoder.state_dict(),
            },
            encoder_dir / f"{client_id}_encoder.pt",
        )
        rows.append(
            {
                "client_id": client_id,
                "hidden_modality_id": int(cluster_id),
                "hidden_modality_name": f"m{cluster_id}",
                "num_samples": 8,
                "encoder_type": "mlp",
            }
        )

    _write_csv(
        partition_dir / "client_meta.csv",
        ["client_id", "hidden_modality_id", "hidden_modality_name", "num_samples", "encoder_type"],
        rows,
    )
    _write_csv(
        cluster_dir / "pred_cluster.csv",
        ["client_id", "pred_cluster"],
        [{"client_id": row["client_id"], "pred_cluster": row["hidden_modality_id"]} for row in rows],
    )
    _write_csv(
        cluster_dir / "true_cluster.csv",
        ["client_id", "true_cluster"],
        [{"client_id": row["client_id"], "true_cluster": row["hidden_modality_id"]} for row in rows],
    )
    torch.save(
        {
            "label": torch.tensor([0, 1], dtype=torch.long),
            "modalities": {"m0": torch.randn(2, 3), "m1": torch.randn(2, 3)},
            "modality_names": ["m0", "m1"],
        },
        partition_dir / "test_multimodal.pt",
    )
    return cfg, result_dir, model_dir


def test_invalid_evaluation_mode_rejected():
    with pytest.raises(ValueError, match="evaluation_mode"):
        _evaluation_mode_spec({"evaluation": {"evaluation_mode": "maybe"}})


def test_curve_eval_rounds_include_multiples_and_final():
    assert _curve_evaluation_rounds(30, 10) == [10, 20, 30]
    assert _curve_evaluation_rounds(25, 10) == [10, 20, 25]


def test_formal_periodic_test_path_fails_loudly():
    with pytest.raises(RuntimeError, match="periodic test evaluation"):
        _evaluate_test_checkpoint(
            evaluation_mode="formal",
            checkpoint_role="periodic",
            server=None,
            clients_by_id={},
            multimodal_path=Path("unused.pt"),
            oracle_mapping={},
            cfg={},
            device=torch.device("cpu"),
        )


def test_curve_mode_writes_isolated_schema_and_does_not_early_stop(tmp_path, monkeypatch):
    cfg, result_dir, model_dir = _make_tiny_run(tmp_path, "curve", rounds=12)
    calls = []

    def fake_evaluator(*args, **kwargs):
        calls.append(len(calls) + 1)
        return {
            "eval_status": "success",
            "eval_failure_reason": None,
            "loss": 0.0,
            "classification_loss": 0.0,
            "accuracy": 0.5 + 0.1 * len(calls),
            "balanced_accuracy": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.4 + 0.1 * len(calls),
            "weighted_f1": 0.3 + 0.1 * len(calls),
            "binary_f1": None,
        }

    monkeypatch.setattr(training_mod, "evaluate_naturally_paired_fusion", fake_evaluator)
    metrics = train_msl_split_learning(cfg, tmp_path, torch.device("cpu"))

    assert calls == [1, 2]
    assert metrics["evaluation_mode"] == "curve"
    assert metrics["periodic_test_evaluation_count"] == 2
    assert metrics["final_test_evaluation_count"] == 0
    assert metrics["executed_global_rounds"] == 12
    assert metrics["stop_reason"] == "max_global_rounds"
    assert metrics["checkpoint"] == "last_model.pt"
    assert metrics["checkpoint_policy"] == "final"
    assert (model_dir / "last_model.pt").exists()

    with (result_dir / "test_curve.csv").open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == ["round", "test_accuracy", "test_macro_f1", "test_weighted_f1"]
    assert [int(row["round"]) for row in rows] == [10, 12]
    assert [float(row["test_accuracy"]) for row in rows] == pytest.approx([0.6, 0.7])
    assert metrics["test_accuracy"] is None
    assert not (result_dir / "formal_test_access.json").exists()

    with (result_dir / "train_log.csv").open("r", newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 12


def test_formal_mode_has_no_intermediate_test_and_final_call_once(tmp_path, monkeypatch):
    cfg, result_dir, _ = _make_tiny_run(tmp_path, "formal", rounds=12)
    calls = []

    def fake_evaluator(*args, **kwargs):
        assert (result_dir / "formal_test_access.json").exists()
        calls.append("final")
        return {
            "eval_status": "success",
            "eval_failure_reason": None,
            "loss": 0.0,
            "classification_loss": 0.0,
            "accuracy": 0.9,
            "balanced_accuracy": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.8,
            "weighted_f1": 0.7,
            "binary_f1": None,
        }

    monkeypatch.setattr(training_mod, "evaluate_naturally_paired_fusion", fake_evaluator)
    metrics = train_msl_split_learning(cfg, tmp_path, torch.device("cpu"))

    assert calls == ["final"]
    assert metrics["evaluation_mode"] == "formal"
    assert metrics["periodic_test_evaluation_count"] == 0
    assert metrics["final_test_evaluation_count"] == 1
    assert metrics["test_evaluations"] == 1
    assert metrics["checkpoint_policy"] == "final"
    assert metrics["checkpoint"] == "last_model.pt"
    assert metrics["formal_test_access_marker"] == str(result_dir / "formal_test_access.json")
    with (result_dir / "formal_test_access.json").open("r", encoding="utf-8") as handle:
        marker = json.load(handle)
    assert marker["status"] == "accessed"
    assert marker["evaluation_mode"] == "formal"
    assert marker["checkpoint_policy"] == "final"
    assert marker["checkpoint_sha256"]
    assert not (result_dir / "test_curve.csv").exists()
    assert metrics["test_accuracy"] == pytest.approx(0.9)


def test_formal_guard_rejects_second_access(tmp_path, monkeypatch):
    cfg, result_dir, _ = _make_tiny_run(tmp_path, "formal", rounds=2)
    calls = []

    def fake_evaluator(*args, **kwargs):
        calls.append("final")
        return {
            "eval_status": "success",
            "eval_failure_reason": None,
            "loss": 0.0,
            "classification_loss": 0.0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "binary_f1": None,
        }

    monkeypatch.setattr(training_mod, "evaluate_naturally_paired_fusion", fake_evaluator)
    train_msl_split_learning(cfg, tmp_path, torch.device("cpu"))

    with pytest.raises(RuntimeError, match="Formal test access marker already exists"):
        train_msl_split_learning(cfg, tmp_path, torch.device("cpu"))

    assert calls == ["final"]
    assert (result_dir / "formal_test_access.json").exists()


def test_existing_scheduler_and_binding_regression_still_updates(tmp_path, monkeypatch):
    cfg, _, _ = _make_tiny_run(tmp_path, "formal", rounds=2)
    monkeypatch.setattr(
        training_mod,
        "evaluate_naturally_paired_fusion",
        lambda *args, **kwargs: {
            "eval_status": "success",
            "eval_failure_reason": None,
            "loss": 0.0,
            "classification_loss": 0.0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "binary_f1": None,
        },
    )

    metrics = train_msl_split_learning(cfg, tmp_path, torch.device("cpu"))

    assert metrics["clients_per_round"] == 2
    assert metrics["configured_global_rounds"] == 2
    assert metrics["executed_global_rounds"] == 2
    assert metrics["total_effective_local_steps"] == 2
