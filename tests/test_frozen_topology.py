import csv
import json
from pathlib import Path

import numpy as np

import experiments.training as training


def _read_assignment(path: Path, column: str) -> dict[str, int]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return {
            str(row["client_id"]): int(row[column])
            for row in csv.DictReader(handle)
        }


def _payload():
    return {
        "client_ids": np.asarray([f"client_{idx:03d}" for idx in range(10)]),
        "fingerprints": np.asarray([[float(idx), float(idx % 3)] for idx in range(10)], dtype=np.float32),
        "pred_cluster": np.asarray([0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int),
        "true_cluster": np.asarray([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=int),
    }


def _prepare_dirs(tmp_path: Path):
    clients_dir = tmp_path / "clients"
    adaptive_dir = tmp_path / "adaptive"
    topology_dir = tmp_path / "topology"
    (adaptive_dir / "pretrained_encoders").mkdir(parents=True)
    return clients_dir, adaptive_dir, topology_dir


def test_ours_topology_uses_raw_adaptive_assignment_without_repair(tmp_path, monkeypatch):
    payload = _payload()
    clients_dir, adaptive_dir, topology_dir = _prepare_dirs(tmp_path)
    monkeypatch.setattr(training, "load_fingerprint_npz", lambda _path: payload)

    cluster_dir, assignment_source, metadata = training.prepare_method_topology(
        "ours",
        clients_dir,
        adaptive_dir,
        topology_dir,
        seed=7,
        clients_per_round=4,
    )

    raw = _read_assignment(Path(cluster_dir) / "raw_cluster_assignment.csv", "raw_cluster")
    pred = _read_assignment(Path(cluster_dir) / "pred_cluster.csv", "pred_cluster")
    expected = {str(client_id): int(label) for client_id, label in zip(payload["client_ids"], payload["pred_cluster"])}
    assert assignment_source == "pred_cluster"
    assert raw == expected
    assert pred == expected
    assert metadata["raw_cluster_assignment"] == metadata["training_cluster_assignment"]
    assert metadata["feasibility_repair_applied"] is False
    assert metadata["num_reassigned_clients"] == 0
    assert metadata["assignment_immutable"] is True

    with (Path(cluster_dir) / "feasibility_metadata.json").open("r", encoding="utf-8") as handle:
        saved = json.load(handle)
    assert saved["raw_cluster_assignment"] == saved["training_cluster_assignment"]


def test_kmeans_topology_uses_raw_kmeans_assignment_without_repair(tmp_path, monkeypatch):
    payload = _payload()
    kmeans_assignment = np.asarray([0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int)
    clients_dir, adaptive_dir, topology_dir = _prepare_dirs(tmp_path)
    monkeypatch.setattr(training, "load_fingerprint_npz", lambda _path: payload)
    monkeypatch.setattr(training, "run_kmeans", lambda fingerprints, k, seed=0: kmeans_assignment)

    cluster_dir, assignment_source, metadata = training.prepare_method_topology(
        "kmeans2",
        clients_dir,
        adaptive_dir,
        topology_dir,
        seed=7,
        clients_per_round=4,
    )

    raw = _read_assignment(Path(cluster_dir) / "raw_cluster_assignment.csv", "raw_cluster")
    pred = _read_assignment(Path(cluster_dir) / "pred_cluster.csv", "pred_cluster")
    expected = {str(client_id): int(label) for client_id, label in zip(payload["client_ids"], kmeans_assignment)}
    assert assignment_source == "pred_cluster"
    assert raw == expected
    assert pred == expected
    assert metadata["raw_cluster_assignment"] == metadata["training_cluster_assignment"]
    assert metadata["feasibility_repair_applied"] is False
    assert metadata["num_reassigned_clients"] == 0
    assert metadata["assignment_immutable"] is True


def test_cluster_aware_method_policies_disable_formal_repair():
    assert training.resolve_method_policy("ours").allow_repair is False
    assert training.resolve_method_policy("kmeans2").allow_repair is False
    assert training.resolve_method_policy("kmeans5").allow_repair is False
