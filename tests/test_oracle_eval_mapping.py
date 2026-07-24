import csv
from pathlib import Path

from evaluation.oracle_mapping import (
    MERGED_TRUE_MODALITY_FAILURE,
    SPLIT_TRUE_MODALITY_FAILURE,
    SUCCESS,
    build_oracle_eval_mapping,
)


def _write_csv(path: Path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_inputs(tmp_path: Path, meta_rows, assignment_rows):
    meta = tmp_path / "client_meta.csv"
    assignments = tmp_path / "cluster_assignments.csv"
    _write_csv(meta, ["client_id", "hidden_modality_id"], meta_rows)
    _write_csv(assignments, ["client_id", "pred_cluster"], assignment_rows)
    return meta, assignments


def test_oracle_one_to_one_mapping_success(tmp_path):
    meta, assignments = _write_inputs(
        tmp_path,
        [
            {"client_id": "client_002", "hidden_modality_id": 0},
            {"client_id": "client_001", "hidden_modality_id": 0},
            {"client_id": "client_010", "hidden_modality_id": 1},
        ],
        [
            {"client_id": "client_002", "pred_cluster": 3},
            {"client_id": "client_001", "pred_cluster": 3},
            {"client_id": "client_010", "pred_cluster": 7},
        ],
    )

    result = build_oracle_eval_mapping(meta, assignments, tmp_path / "oracle_eval_modality_to_cluster.json")

    assert result["status"] == SUCCESS
    assert result["modality_to_cluster"] == {"0": 3, "1": 7}
    assert result["cluster_to_modality"] == {"3": 0, "7": 1}
    assert result["representative_clients"] == {"0": "client_001", "1": "client_010"}
    assert (tmp_path / "oracle_eval_modality_to_cluster.json").exists()


def test_oracle_mapping_fails_when_true_modality_is_split(tmp_path):
    meta, assignments = _write_inputs(
        tmp_path,
        [
            {"client_id": "client_001", "hidden_modality_id": 0},
            {"client_id": "client_002", "hidden_modality_id": 0},
            {"client_id": "client_010", "hidden_modality_id": 1},
        ],
        [
            {"client_id": "client_001", "pred_cluster": 3},
            {"client_id": "client_002", "pred_cluster": 4},
            {"client_id": "client_010", "pred_cluster": 7},
        ],
    )

    result = build_oracle_eval_mapping(meta, assignments)

    assert result["status"] == "failed"
    assert result["failure_reason"] == SPLIT_TRUE_MODALITY_FAILURE
    assert result["modality_to_cluster"] is None


def test_oracle_mapping_fails_when_cluster_merges_true_modalities(tmp_path):
    meta, assignments = _write_inputs(
        tmp_path,
        [
            {"client_id": "client_001", "hidden_modality_id": 0},
            {"client_id": "client_010", "hidden_modality_id": 1},
        ],
        [
            {"client_id": "client_001", "pred_cluster": 3},
            {"client_id": "client_010", "pred_cluster": 3},
        ],
    )

    result = build_oracle_eval_mapping(meta, assignments)

    assert result["status"] == "failed"
    assert result["failure_reason"] == MERGED_TRUE_MODALITY_FAILURE
    assert result["cluster_to_modality"] is None
