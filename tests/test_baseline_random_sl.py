"""Tests for the randomSL baseline (random-scheduling Split Learning)."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from baseline.randomSL.scheduling import RandomScheduler
from baseline.randomSL.training import _random_sl_round, run_random_sl_stage3_split_training
from MSL.learning.fusion_sl import _load_clients
from MSL.learning.models import ConcatMLPFusionServer, create_client_encoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "baseline" / "randomSL" / "stage3_train.py"
CLIENT_IDS = ("client_000", "client_001", "client_002", "client_003")


class SchedulerClient:
    def __init__(self, client_id, pred_cluster):
        self.client_id = client_id
        self.pred_cluster = pred_cluster

    @property
    def hidden_modality_id(self):
        raise AssertionError("randomSL scheduler must not read hidden_modality_id.")


def _client_cfg():
    return {
        "encoder_hidden_dim": 8,
        "model": {"encoder": {"type": "time_series"}},
        "num_classes": 3,
    }


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _encoder_state():
    encoder = create_client_encoder(_client_cfg(), input_shape=[2], encoder_type="time_series")
    return encoder.state_dict()


def _stage1_dir(tmp_path):
    stage1 = tmp_path / "partition" / "synthetic_stage3" / "m0_2clients_m1_2clients"
    train_dir = stage1 / "train_clients"
    train_dir.mkdir(parents=True)
    _write_csv(
        stage1 / "client_meta.csv",
        ["client_id", "hidden_modality_id", "hidden_modality_name", "num_samples", "encoder_type"],
        [
            {
                "client_id": client_id,
                "hidden_modality_id": idx % 2,
                "hidden_modality_name": f"m{idx % 2}",
                "num_samples": 8,
                "encoder_type": "time_series",
            }
            for idx, client_id in enumerate(CLIENT_IDS)
        ],
    )
    (stage1 / "partition_config.json").write_text(
        json.dumps({"dataset_type": "synthetic_stage3", "num_clients": len(CLIENT_IDS)}),
        encoding="utf-8",
    )
    torch.save(
        {
            "modalities": {"m0": torch.zeros(3, 2), "m1": torch.ones(3, 2)},
            "modality_names": ["m0", "m1"],
            "modality_input_shapes": [[2], [2]],
            "label": torch.tensor([0, 1, 2]),
            "split": "test",
        },
        stage1 / "test_multimodal.pt",
    )
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    for idx, client_id in enumerate(CLIENT_IDS):
        torch.save(
            {
                "client_id": client_id,
                "samples": torch.randn(8, 2),
                "labels": labels.clone(),
                "input_shape": [2],
                "encoder_type": "time_series",
                "hidden_modality_id": idx % 2,
            },
            train_dir / f"{client_id}.pt",
        )
    return stage1


def _stage2_dir(tmp_path):
    stage2 = tmp_path / "cluster" / "synthetic_stage3" / "m0_2clients_m1_2clients" / "adaptive_isodata"
    encoder_dir = stage2 / "pretrained_encoders"
    encoder_dir.mkdir(parents=True)
    _write_csv(
        stage2 / "pred_cluster.csv",
        ["client_id", "pred_cluster"],
        [{"client_id": client_id, "pred_cluster": idx % 2} for idx, client_id in enumerate(CLIENT_IDS)],
    )
    _write_csv(
        stage2 / "true_cluster.csv",
        ["client_id", "true_cluster"],
        [{"client_id": client_id, "true_cluster": idx % 2} for idx, client_id in enumerate(CLIENT_IDS)],
    )
    (stage2 / "stage2_metadata.json").write_text(
        json.dumps(
            {
                "stage": "stage2_discovery",
                "dataset": "synthetic_stage3",
                "partition_signature": "m0_2clients_m1_2clients",
                "cluster_method": "adaptive_isodata",
                "metrics": {"discovery_status": "discovery_success", "estimated_Q": 2},
            }
        ),
        encoding="utf-8",
    )
    state = _encoder_state()
    for client_id in CLIENT_IDS:
        torch.save(
            {"client_id": client_id, "state_dict": state},
            encoder_dir / f"{client_id}_encoder.pt",
        )
    return stage2


def _train_cfg(stage1, stage2, result_dir, global_rounds=2):
    return {
        "seed": 7,
        "device": "cpu",
        "dataset": {"type": "synthetic_stage3", "split_protocol": "subject_disjoint_tvt_v1"},
        "partition": {"output_dir": str(stage1), "clients_per_modality": 2},
        "cluster": {"output_dir": str(stage2)},
        "result": {"output_dir": str(result_dir)},
        "result_model": {"output_dir": str(result_dir)},
        "training": {
            "scheduler": "random",
            "cluster_assignment_source": "pred_cluster",
            "global_rounds": global_rounds,
            "local_steps": 1,
            "batch_size": 4,
            "clients_per_cluster_per_round": 1,
            "server_lr": 0.1,
            "client_lr": 0.1,
            "eval_batch_size": 4,
        },
        "binding": {"type": "label_random", "batch_size": 4},
        "fusion": {
            "type": "concat_mlp",
            "training_objective": "label_random_ce",
            "adapter_dim": 8,
            "hidden_dim": 8,
            "dropout": 0.0,
        },
        "evaluation": {"run_test": True},
        "num_classes": 3,
        "encoder_hidden_dim": 8,
    }


def test_random_scheduler_never_reads_hidden_modality_id():
    clients = [SchedulerClient(f"c{idx}", idx % 2) for idx in range(8)]
    scheduler = RandomScheduler(clients, clients_per_round=4, seed=3)

    rounds = [scheduler.sample_round() for _ in range(3)]

    for selected in rounds:
        assert len(selected) == 4
        assert len({client.client_id for client in selected}) == 4


def test_random_scheduler_can_miss_a_cluster():
    clients = [SchedulerClient(f"c{idx}", idx % 2) for idx in range(4)]
    scheduler = RandomScheduler(clients, clients_per_round=1, seed=5)

    for _ in range(10):
        selected = scheduler.sample_round()
        metrics = scheduler.metrics(selected)
        assert metrics["coverage"] == 0.5
        assert metrics["participation_fairness"] > 0.0


def test_random_sl_round_tolerates_missing_cluster(tmp_path):
    stage1 = _stage1_dir(tmp_path)
    stage2 = _stage2_dir(tmp_path)
    cfg = _train_cfg(stage1, stage2, tmp_path / "results")
    device = torch.device("cpu")
    clients = _load_clients(cfg, stage1, stage2, device)
    cluster_ids = [0, 1]
    server = ConcatMLPFusionServer(cluster_ids, 8, 3, cfg, cluster_to_slot={0: 0, 1: 1})
    optimizer = torch.optim.Adam(server.parameters(), lr=0.1)

    selected = [client for client in clients if int(client.pred_cluster) == 0]
    metrics = _random_sl_round(server, optimizer, selected, cluster_ids, cfg)

    assert metrics["empty_binding_round"] == 1
    assert metrics["missing_cluster_ids"] == [1]
    assert metrics["empty_binding_reason"] == "missing_cluster"
    assert metrics["loss"] == 0.0


def test_random_sl_training_smoke_run(tmp_path):
    stage1 = _stage1_dir(tmp_path)
    stage2 = _stage2_dir(tmp_path)
    result_dir = tmp_path / "results"
    cfg = _train_cfg(stage1, stage2, result_dir, global_rounds=2)

    metrics = run_random_sl_stage3_split_training(cfg, PROJECT_ROOT, torch.device("cpu"))

    assert metrics["method"] == "random_sl"
    assert metrics["baseline"] == "randomSL"
    assert metrics["scheduler"] == "random"
    assert metrics["training_mode"] == "random_sl_split_learning"
    assert metrics["executed_global_rounds"] == 2
    assert metrics["test_eval_status"] == "success"
    assert (result_dir / "train_log.csv").exists()
    assert (result_dir / "final_metrics.json").exists()
    assert (result_dir / "last_model.pt").exists()
    with (result_dir / "train_log.csv").open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert "missing_cluster_ids_json" in reader.fieldnames
        assert "empty_binding_reason" in reader.fieldnames
        assert len(rows) == 2


def test_baseline_script_reuses_stage3_run_layout(tmp_path):
    spec = importlib.util.spec_from_file_location("baseline_random_sl", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stage3 = module._load_stage3_script()
    stage1 = _stage1_dir(tmp_path)
    stage2 = _stage2_dir(tmp_path)

    cfg, paths = stage3.build_stage3_run(
        _train_cfg(stage1, stage2, tmp_path / "results"),
        stage1_dir=stage1,
        stage2_dir=stage2,
        output_root=tmp_path / "experiments",
        attempt=1,
    )

    assert paths["run_dir"].name == "seed-7"
    assert Path(cfg["partition"]["output_dir"]) == stage1.resolve()
    assert Path(cfg["cluster"]["output_dir"]) == stage2.resolve()
