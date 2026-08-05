import inspect
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from MSL.discovery.clustering import (
    AdaptiveISODATAEstimator,
    _best_merge_proposal,
    _best_split_proposal,
    _partition_bic,
    adaptive_isodata,
)
from MSL.evaluation.metrics import discovery_metrics
from MSL.utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _blobs(num_clusters, samples_per_cluster=12, dim=6, separation=8.0, noise=0.35, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    labels = []
    for cluster_id in range(num_clusters):
        center = np.zeros(dim, dtype=np.float32)
        center[cluster_id % dim] = float(separation)
        rows.append(center + rng.normal(0.0, noise, size=(samples_per_cluster, dim)))
        labels.extend([cluster_id] * samples_per_cluster)
    return np.vstack(rows).astype(np.float32), np.asarray(labels, dtype=int)


def _sized_blobs(sizes, dim=6, separation=8.0, noise=0.35, seed=0, noise_dims=0):
    rng = np.random.default_rng(seed)
    rows = []
    labels = []
    for cluster_id, size in enumerate(sizes):
        center = np.zeros(dim, dtype=np.float32)
        center[cluster_id % dim] = float(separation)
        rows.append(center + rng.normal(0.0, noise, size=(int(size), dim)))
        labels.extend([cluster_id] * int(size))
    x = np.vstack(rows).astype(np.float32)
    if noise_dims:
        noise_block = rng.normal(0.0, 1.0, size=(x.shape[0], int(noise_dims))).astype(np.float32)
        x = np.concatenate([x, noise_block], axis=1)
    return x, np.asarray(labels, dtype=int)


def _line_blobs(centers, samples_per_cluster=10, noise=0.08, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    labels = []
    for cluster_id, center in enumerate(centers):
        rows.append(rng.normal(float(center), noise, size=(samples_per_cluster, 1)))
        labels.extend([cluster_id] * samples_per_cluster)
    return np.vstack(rows).astype(np.float32), np.asarray(labels, dtype=int)


def _estimate(x, **kwargs):
    params = {
        "seeds": [11, 23, 37],
        "max_iter": 20,
        "q_max": 8,
        "min_cluster_size": 2,
        "min_cluster_size_fraction": None,
    }
    params.update(kwargs)
    pred, diag = adaptive_isodata(
        x,
        **params,
    )
    return pred, diag


def _best_permutation_accuracy(y_true, y_pred):
    values = sorted(set(y_true))
    clusters = sorted(set(y_pred))
    if len(values) != len(clusters):
        return 0.0
    best = 0.0
    for perm in permutations(values):
        mapping = {cluster: value for cluster, value in zip(clusters, perm)}
        mapped = np.asarray([mapping[v] for v in y_pred])
        best = max(best, float((mapped == y_true).mean()))
    return best


def _assert_each_true_modality_is_represented_without_predicted_mixing(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    for value in np.unique(y_true):
        assert np.any(y_pred[y_true == value] >= 0)
    for cluster_id in np.unique(y_pred):
        assert len(set(y_true[y_pred == cluster_id])) == 1


def test_adaptive_isodata_recovers_two_modalities():
    x, y = _blobs(2)
    pred, diag = _estimate(x)

    assert len(set(pred)) == 2
    assert _best_permutation_accuracy(y, pred) == 1.0
    assert diag["estimated_Q"] == 2
    assert diag["selection_confidence"] == "high"


def test_adaptive_isodata_recovers_three_modalities():
    x, y = _blobs(3)
    pred, diag = _estimate(x)

    assert len(set(pred)) >= 3
    _assert_each_true_modality_is_represented_without_predicted_mixing(y, pred)
    assert all(q >= 3 for q in diag["per_seed_estimated_Q"])


def test_adaptive_isodata_recovers_four_modalities():
    x, y = _blobs(4)
    pred, diag = _estimate(x)

    assert len(set(pred)) >= 4
    _assert_each_true_modality_is_represented_without_predicted_mixing(y, pred)
    assert not diag["boundary_saturation"]


def test_adaptive_isodata_is_translation_and_scale_invariant():
    x, _ = _blobs(3)
    shifted = x + 123.0
    scaled = x * 7.5

    pred, _ = _estimate(x)
    shifted_pred, _ = _estimate(shifted)
    scaled_pred, _ = _estimate(scaled)

    assert np.array_equal(pred, shifted_pred)
    assert np.array_equal(pred, scaled_pred)


def test_adaptive_isodata_handles_uninformative_dimensions():
    x, y = _blobs(3)
    padded = np.concatenate([x, np.ones((x.shape[0], 20), dtype=np.float32)], axis=1)

    pred, diag = _estimate(padded)

    assert len(set(pred)) >= 3
    _assert_each_true_modality_is_represented_without_predicted_mixing(y, pred)
    assert diag["preprocessing"]["removed_near_zero_variance_dims"] == 20


def test_adaptive_isodata_is_deterministic_for_fixed_seed_list():
    x, _ = _blobs(4)

    first, first_diag = _estimate(x)
    second, second_diag = _estimate(x)

    assert np.array_equal(first, second)
    assert first_diag["per_seed_estimated_Q"] == second_diag["per_seed_estimated_Q"]


def test_clear_structure_is_stable_across_seeds():
    x, _ = _blobs(4)
    _, diag = _estimate(x)

    assert diag["q_stability"] == 1.0
    assert diag["assignment_stability"] == 1.0


def test_single_unstructured_cloud_does_not_split_to_q_max():
    rng = np.random.default_rng(4)
    x = rng.normal(0.0, 1.0, size=(30, 6)).astype(np.float32)

    pred, diag = _estimate(x)

    assert len(set(pred)) == 1
    assert diag["estimated_Q"] == 1
    assert not diag["boundary_saturation"]


def test_small_outlier_group_is_not_accepted_as_modality():
    large, _ = _blobs(1, samples_per_cluster=24, seed=1)
    outliers = np.full((2, large.shape[1]), 12.0, dtype=np.float32)
    x = np.vstack([large, outliers])

    pred, diag = _estimate(x, min_cluster_size=4)

    assert len(set(pred)) == 1
    assert not diag["small_cluster_present"]


def test_q_max_boundary_saturation_is_marked():
    x, _ = _blobs(4)

    pred, diag = adaptive_isodata(
        x,
        seeds=[11, 23, 37],
        q_max=2,
        min_cluster_size_fraction=0.08,
        min_split_silhouette=0.0,
    )

    assert len(set(pred)) == 2
    assert diag["boundary_saturation"]
    assert diag["selection_confidence"] == "low"


def test_split_merge_stops_without_infinite_loop():
    x, _ = _blobs(3)
    _, diag = _estimate(x)

    assert diag["convergence_reason"] in {
        "repeated_partition",
        "global_objective_not_improved",
        "no_acceptable_split_or_merge",
        "q_max_reached",
        "unsupervised_selection_patience",
        "max_iter",
    }
    assert len(diag["split_history"]) <= 20


def test_estimator_api_does_not_accept_hidden_modality_or_true_q():
    signature = inspect.signature(AdaptiveISODATAEstimator.fit_predict)
    params = set(signature.parameters)

    assert "hidden_modality_id" not in params
    assert "true_modality" not in params
    assert "true_q" not in params
    assert list(signature.parameters) == ["self", "client_reps"]


def test_unknown_q_configs_use_same_adaptive_parameters_without_true_initial_k():
    configs = [
        PROJECT_ROOT / "configs" / "uci_har.config",
        PROJECT_ROOT / "configs" / "mhealth" / "fold1.config",
        PROJECT_ROOT / "configs" / "pamap2" / "fold1.config",
        PROJECT_ROOT / "configs" / "iemocap" / "fold1.config",
    ]
    adaptive_blocks = []
    for path in configs:
        cfg = load_config(path)
        cluster = cfg["cluster"]
        assert cluster["method"] == "adaptive_isodata"
        assert cluster["known_k"] is None
        assert "isodata" not in cluster
        assert "initial_k" not in cluster.get("adaptive", {})
        assert cluster["adaptive"]["q_max"] == 8
        assert cluster["adaptive"]["min_cluster_size"] == 2
        assert cluster["adaptive"]["min_cluster_size_fraction"] is None
        assert "min_silhouette_improvement" not in cluster["adaptive"]
        adaptive_blocks.append(cluster["adaptive"])

    assert all(block == adaptive_blocks[0] for block in adaptive_blocks[1:])


def test_MSL_configs_freeze_stage_boundaries_and_no_validation_selection():
    expected = {
        "configs/uci_har.config": {
            "lr": 0.0002,
            "split_protocol": "subject_disjoint_70_30_no_val_v1",
        },
        "configs/mhealth/fold1.config": {
            "lr": 0.0002,
            "split_protocol": "subject_5fold_no_val_fold1_v1",
        },
        "configs/pamap2/fold1.config": {
            "lr": 0.0001,
            "split_protocol": "subject_9fold_loso_no_val_fold1_v1",
        },
    }
    adaptive_blocks = []
    forbidden_top_level = {
        "num_modalities",
        "learning_rate",
        "batch_size",
        "result",
        "result_model",
    }
    for relative, values in expected.items():
        path = PROJECT_ROOT / relative
        cfg = load_config(path)
        assert cfg["seed"] == 42
        assert not forbidden_top_level.intersection(cfg)
        assert "server" not in cfg["model"]
        assert "output_dir" not in cfg["partition"]
        assert "output_dir" not in cfg["cluster"]
        assert cfg["partition"]["clients_per_modality"] == 10
        assert cfg["cluster"]["method"] == "adaptive_isodata"
        assert cfg["cluster"]["known_k"] is None
        assert cfg["training"]["scheduler"] == "balanced_cluster_round_robin"
        assert cfg["dataset"]["split_protocol"] == values["split_protocol"]
        assert cfg["training"]["validation_enabled"] is False
        assert cfg["dataset"]["validation_subjects"] == []
        assert cfg["training"]["global_rounds"] == 200
        assert "early_stopping" not in cfg["training"]
        assert cfg["training"]["local_steps"] == 1
        assert cfg["training"]["clients_per_cluster_per_round"] == 4
        assert cfg["training"]["client_lr"] == values["lr"]
        assert cfg["training"]["server_lr"] == values["lr"]
        assert cfg["binding"]["type"] == "label_random"
        assert cfg["fusion"]["type"] == "concat_mlp"
        assert cfg["d2d"]["enabled"] is False
        adaptive_blocks.append(cfg["cluster"]["adaptive"])

    assert all(block == adaptive_blocks[0] for block in adaptive_blocks[1:])


def test_discovery_metrics_distinguish_overclustering_from_success():
    true = np.asarray([0] * 6 + [1] * 6)
    overclustered = np.asarray([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    perfect = np.asarray([1] * 6 + [0] * 6)

    over = discovery_metrics(true, overclustered)
    ok = discovery_metrics(true, perfect)

    assert over["ACC"] == 1.0
    assert over["hungarian_ACC"] < 1.0
    assert over["discovery_status"] == "incorrect_q_over_or_under_clustering"
    assert over["true_modality_pred_cluster_splits"]["0"]["num_pred_clusters"] == 3
    assert ok["discovery_status"] == "discovery_success"
    assert ok["hungarian_ACC"] == 1.0


def test_partition_bic_uses_hard_partition_parameter_count_without_mixture_weights():
    x = np.asarray([[0.0], [0.1], [8.0], [8.1]], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1], dtype=int)
    n, d = x.shape
    sse = ((x[:2] - x[:2].mean()) ** 2).sum() + ((x[2:] - x[2:].mean()) ** 2).sum()
    sigma2 = max(float(sse) / (n * d), 1.0e-9)
    expected = (
        -0.5 * n * d * (np.log(2.0 * np.pi * sigma2) + 1.0)
        - 0.5 * (2 * d + 1) * np.log(n)
    )

    assert np.isclose(_partition_bic(x, labels), expected)


def test_adaptive_objective_rejects_non_finite_inputs():
    x, _ = _blobs(2)
    x[0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        adaptive_isodata(x, seeds=[11], q_max=8, min_cluster_size=2, min_cluster_size_fraction=None)


def test_zero_variance_inputs_are_safe_and_finite():
    x = np.ones((6, 4), dtype=np.float32)

    pred, diag = adaptive_isodata(x, seeds=[11], q_max=8, min_cluster_size=2, min_cluster_size_fraction=None)

    assert len(set(pred)) == 1
    assert np.isfinite(diag["per_seed_objective"][0])
    assert diag["preprocessing"]["removed_near_zero_variance_dims"] == 3


def test_extremely_small_sample_set_exits_safely():
    x = np.asarray([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)

    pred, diag = adaptive_isodata(x, seeds=[11], q_max=8, min_cluster_size=2, min_cluster_size_fraction=None)

    assert len(pred) == 2
    assert len(set(pred)) == 1
    assert diag["algorithm_config"]["q_max"] == 1


def test_split_restart_seeds_are_derived_from_run_seed():
    x, _ = _blobs(2, samples_per_cluster=10)
    labels = np.zeros(x.shape[0], dtype=int)

    first = _best_split_proposal(
        x,
        labels,
        seed=11,
        q_max=4,
        min_cluster_size=4,
        bic_improvement_min=0.0,
        min_split_silhouette=0.0,
        split_kmeans_restarts=5,
    )
    second = _best_split_proposal(
        x,
        labels,
        seed=23,
        q_max=4,
        min_cluster_size=4,
        bic_improvement_min=0.0,
        min_split_silhouette=0.0,
        split_kmeans_restarts=5,
    )

    assert first is not None
    assert second is not None
    assert first["restart_seed"] != second["restart_seed"]
    assert adjusted_rand_score(first["labels"], second["labels"]) == 1.0


def test_clear_imbalanced_18_to_2_modalities_are_recovered():
    x, y = _sized_blobs([18, 2], separation=10.0, noise=0.15, seed=1)

    pred, diag = _estimate(x, min_split_silhouette=0.0)

    assert diag["estimated_Q"] >= 2
    _assert_each_true_modality_is_represented_without_predicted_mixing(y, pred)


def test_clear_imbalanced_16_to_2_to_2_modalities_are_recovered():
    x, y = _sized_blobs([16, 2, 2], separation=10.0, noise=0.15, seed=1)

    pred, diag = _estimate(x, min_split_silhouette=0.0)

    assert diag["estimated_Q"] >= 3
    _assert_each_true_modality_is_represented_without_predicted_mixing(y, pred)


def test_imbalanced_partly_overlapping_modalities_are_not_swallowed():
    x, y = _sized_blobs([14, 6], separation=4.0, noise=0.8, seed=4)

    pred, diag = _estimate(x, min_split_silhouette=0.0)

    assert diag["estimated_Q"] >= 2
    _assert_each_true_modality_is_represented_without_predicted_mixing(y, pred)


def test_high_dimensional_noise_does_not_hide_effective_modalities():
    x, y = _sized_blobs([8, 8, 8], dim=4, separation=8.0, noise=0.2, seed=5)
    rng = np.random.default_rng(5)
    nuisance = rng.normal(0.0, 1.0e-10, size=(x.shape[0], 80)).astype(np.float32)
    x = np.concatenate([x, nuisance], axis=1)

    pred, diag = _estimate(x, pca_variance=0.8, min_split_silhouette=0.0)

    assert diag["estimated_Q"] >= 3
    _assert_each_true_modality_is_represented_without_predicted_mixing(y, pred)
    assert diag["preprocessing"]["pca_dim"] < x.shape[1]
    assert diag["preprocessing"]["removed_near_zero_variance_dims"] >= 80


def test_fuzzy_structure_reports_low_confidence_or_instability():
    x, _ = _sized_blobs([10, 10], dim=8, separation=1.5, noise=1.0, seed=4)

    _, diag = adaptive_isodata(
        x,
        seeds=[11, 23, 37, 53, 71],
        q_max=8,
        min_cluster_size=2,
        min_cluster_size_fraction=None,
        min_split_silhouette=0.0,
    )

    assert diag["selection_confidence"] in {"low", "medium"} or diag["assignment_stability"] < 0.9


def test_different_seeds_use_different_restart_paths():
    x, _ = _sized_blobs([10, 10], separation=10.0, noise=0.15, seed=1)

    _, diag = adaptive_isodata(
        x,
        seeds=[11, 23],
        q_max=8,
        min_cluster_size=2,
        min_cluster_size_fraction=None,
        min_split_silhouette=0.0,
    )

    first_seed = diag["runs"][0]["split_history"][0]["restart_seed"]
    second_seed = diag["runs"][1]["split_history"][0]["restart_seed"]
    assert first_seed != second_seed


def test_accepted_split_strictly_improves_global_bic_like_score():
    x, _ = _sized_blobs([10, 10], separation=10.0, noise=0.15, seed=1)

    _, diag = _estimate(x, min_split_silhouette=0.0)
    accepted = [item for item in diag["split_history"] if item["accepted"]]

    assert accepted
    assert all(item["new_global_bic"] > item["old_global_bic"] for item in accepted)


def test_contextual_separation_is_diagnostic_only_for_improving_split():
    left, _ = _line_blobs([0.0], samples_per_cluster=10, noise=0.05, seed=1)
    middle, _ = _line_blobs([10.0], samples_per_cluster=10, noise=0.05, seed=2)
    external, _ = _line_blobs([15.0], samples_per_cluster=10, noise=0.05, seed=3)
    x = np.vstack([left, middle, external]).astype(np.float32)
    labels = np.asarray([0] * 20 + [1] * 10, dtype=int)

    proposal = _best_split_proposal(
        x,
        labels,
        seed=11,
        q_max=4,
        min_cluster_size=2,
        bic_improvement_min=0.0,
        min_split_silhouette=0.0,
        split_kmeans_restarts=10,
    )

    assert proposal is not None
    assert proposal["context_separation"] < 1.1
    assert proposal["context_role"] == "diagnostic_only"
    assert "min_context_separation" not in proposal
    assert proposal["new_global_bic"] > proposal["old_global_bic"]
    assert proposal["eligible"]
    assert proposal["reason"] == "bic_improved"


def test_contextual_separation_cannot_rescue_worse_bic_split():
    rng = np.random.default_rng(1)
    left = rng.normal(0.0, 1.0, size=(10, 20)).astype(np.float32)
    external = rng.normal(0.01, 1.0, size=(10, 20)).astype(np.float32)
    x = np.vstack([left, external]).astype(np.float32)
    labels = np.asarray([0] * 10 + [1] * 10, dtype=int)

    proposal = _best_split_proposal(
        x,
        labels,
        seed=11,
        q_max=4,
        min_cluster_size=2,
        bic_improvement_min=0.0,
        min_split_silhouette=0.0,
        split_kmeans_restarts=20,
    )

    assert proposal is not None
    assert proposal["context_role"] == "diagnostic_only"
    assert proposal["context_separation"] > 1.1
    assert proposal["new_global_bic"] <= proposal["old_global_bic"]
    assert not proposal["eligible"]
    assert proposal["reason"] == "bic_penalty_not_paid"


def test_contextual_separation_is_recorded_on_accepted_splits():
    x, _ = _line_blobs([0.0, 10.0, 15.0], samples_per_cluster=10, noise=0.05, seed=4)

    _, diag = _estimate(x, min_split_silhouette=0.0)
    contextual = [item for item in diag["split_history"] if item.get("context_separation") is not None]

    assert contextual
    assert all(item["context_role"] == "diagnostic_only" for item in contextual)
    assert all("min_context_separation" not in item for item in contextual)


def test_uneven_three_cluster_spacing_continues_split_below_old_context_threshold():
    x, y = _line_blobs([0.0, 10.0, 15.0], samples_per_cluster=10, noise=0.05, seed=5)

    pred, diag = _estimate(x, min_split_silhouette=0.0)

    assert diag["estimated_Q"] >= 3
    _assert_each_true_modality_is_represented_without_predicted_mixing(y, pred)
    assert any(
        item.get("accepted") and item.get("context_separation") is not None and item["context_separation"] < 1.1
        for item in diag["split_history"]
    )


def test_uneven_four_cluster_spacing_continues_split_below_old_context_threshold():
    x, y = _line_blobs([0.0, 10.0, 15.0, 19.0], samples_per_cluster=10, noise=0.05, seed=6)

    pred, diag = _estimate(x, min_split_silhouette=0.0)

    assert diag["estimated_Q"] >= 4
    _assert_each_true_modality_is_represented_without_predicted_mixing(y, pred)
    assert any(
        item.get("accepted") and item.get("context_separation") is not None and item["context_separation"] < 1.1
        for item in diag["split_history"]
    )


def test_accepted_merge_strictly_improves_global_bic_like_score():
    x = np.asarray([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.2, 0.0], [5.0, 5.0], [5.1, 5.0]], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=int)

    proposal = _best_merge_proposal(x, labels, bic_improvement_min=0.0)

    assert proposal is not None
    assert proposal["new_global_bic"] > proposal["old_global_bic"]
    assert proposal["reason"] == "bic_improved"


def test_adaptive_discovery_code_does_not_read_hidden_modality_name():
    checked_files = [
        PROJECT_ROOT / "src" / "MSL" / "discovery" / "clustering.py",
        PROJECT_ROOT / "src" / "MSL" / "discovery" / "fingerprint.py",
        PROJECT_ROOT / "src" / "MSL" / "learning" / "pretrain.py",
    ]

    for path in checked_files:
        text = path.read_text(encoding="utf-8-sig")
        assert "hidden_modality_name" not in text
