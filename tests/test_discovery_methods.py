import inspect
from pathlib import Path

import numpy as np

import experiments.discovery_comparison as discovery_runner
from experiments.run_all_discovery import aggregate
from MSL.discovery import run_auto_kmeans, run_gmm_bic
from MSL.protocol import DISCOVERY_METHODS


def _four_cluster_fingerprints():
    centers = np.asarray(
        [
            [-8.0, -8.0],
            [-8.0, 8.0],
            [8.0, -8.0],
            [8.0, 8.0],
        ],
        dtype=np.float32,
    )
    rows = []
    for center in centers:
        for offset in [(-0.15, 0.0), (0.0, 0.15), (0.15, -0.15), (0.08, 0.08), (-0.08, -0.08)]:
            rows.append(center + np.asarray(offset, dtype=np.float32))
    return np.asarray(rows, dtype=np.float32)


def test_auto_kmeans_candidate_search_and_metadata():
    labels, metadata = run_auto_kmeans(_four_cluster_fingerprints(), seed=11)

    assert metadata["candidate_ks"] == [2, 3, 4, 5]
    assert metadata["selection_metric"] == "silhouette"
    assert metadata["tie_break"] == "smaller_k"
    assert metadata["selected_Q_hat"] == len(set(labels.tolist()))
    assert metadata["selected_k"] == 4
    assert [row["k"] for row in metadata["candidates"]] == [2, 3, 4, 5]
    assert all(row["status"] == "success" for row in metadata["candidates"])


def test_auto_kmeans_has_no_true_q_dependency_and_is_deterministic():
    signature = inspect.signature(run_auto_kmeans)
    assert "true_q" not in signature.parameters
    assert "true_modality" not in signature.parameters

    x = _four_cluster_fingerprints()
    first, first_meta = run_auto_kmeans(x, seed=19)
    second, second_meta = run_auto_kmeans(x, seed=19)

    assert first.tolist() == second.tolist()
    assert first_meta["selected_k"] == second_meta["selected_k"]


def test_gmm_bic_candidate_search_and_metadata():
    labels, metadata = run_gmm_bic(_four_cluster_fingerprints(), seed=11)

    assert metadata["candidate_ks"] == [2, 3, 4, 5]
    assert metadata["selection_metric"] == "bic"
    assert metadata["tie_break"] == "smaller_k"
    assert metadata["selected_Q_hat"] == len(set(labels.tolist()))
    assert metadata["selected_k"] == 4
    assert [row["k"] for row in metadata["candidates"]] == [2, 3, 4, 5]
    assert all(row["status"] == "success" for row in metadata["candidates"])
    assert all(isinstance(row["bic"], float) for row in metadata["candidates"])


def test_gmm_bic_has_no_true_q_dependency_and_is_deterministic():
    signature = inspect.signature(run_gmm_bic)
    assert "true_q" not in signature.parameters
    assert "true_modality" not in signature.parameters

    x = _four_cluster_fingerprints()
    first, first_meta = run_gmm_bic(x, seed=23)
    second, second_meta = run_gmm_bic(x, seed=23)

    assert first.tolist() == second.tolist()
    assert first_meta["selected_k"] == second_meta["selected_k"]


def test_discovery_registry_and_runner_support_new_methods(tmp_path, monkeypatch):
    assert DISCOVERY_METHODS == (
        "kmeans2",
        "kmeans3",
        "kmeans4",
        "kmeans5",
        "auto_kmeans",
        "gmm_bic",
        "adaptive_isodata",
    )
    payload = {
        "fingerprints": _four_cluster_fingerprints(),
        "client_ids": np.asarray([f"client_{idx:03d}" for idx in range(20)]),
        "true_cluster": np.repeat(np.arange(4), 5).astype(int),
        "pred_cluster": np.repeat(np.arange(4), 5).astype(int),
    }
    monkeypatch.setattr(discovery_runner, "project_root", lambda: tmp_path)
    monkeypatch.setattr(discovery_runner, "find_clients_dir", lambda root, cfg: tmp_path / "clients")
    monkeypatch.setattr(discovery_runner, "find_discovery_dir", lambda root, clients_dir, method: tmp_path / "adaptive")
    monkeypatch.setattr(discovery_runner, "load_fingerprint_npz", lambda discovery_dir: payload)

    for method in ("auto_kmeans", "gmm_bic"):
        result = discovery_runner.run_one("mhealth", 1, 42, method, tmp_path / "results")
        assert result["status"] == "success"
        assert result["estimated_Q"] == 4
        assert result["abs_Q_error"] == 0
        assert result["hungarian_ACC"] == 1.0
        assert result["selection_metadata"]["selected_Q_hat"] == 4
        assert Path(result["artifact_dir"], "pred_cluster.csv").exists()


def test_discovery_aggregation_table_ii_fields_keep_per_run_q():
    rows = [
        {
            "dataset": "mhealth",
            "fold": 1,
            "seed": 42,
            "method": "auto_kmeans",
            "status": "success",
            "estimated_Q": 4,
            "abs_Q_error": 0,
            "ARI": 1.0,
            "NMI": 1.0,
            "hungarian_ACC": 1.0,
        },
        {
            "dataset": "mhealth",
            "fold": 2,
            "seed": 42,
            "method": "auto_kmeans",
            "status": "success",
            "estimated_Q": 3,
            "abs_Q_error": 1,
            "ARI": 0.5,
            "NMI": 0.6,
            "hungarian_ACC": 0.7,
        },
    ]

    summary = aggregate(rows)["mhealth_auto_kmeans"]

    assert summary["estimated_Q_values"] == [4, 3]
    assert summary["estimated_Q_mean"] == 3.5
    assert summary["abs_Q_error_mean"] == 0.5
    assert summary["ARI_mean"] == 0.75
    assert summary["NMI_mean"] == 0.8
    assert summary["hungarian_ACC_mean"] == 0.85
