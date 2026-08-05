import json

import numpy as np
import pytest

from MSL.evaluation.plot_fingerprint_embedding import (
    _prepare_pca,
    write_fingerprint_pca_figure,
)


def _fingerprints():
    return np.asarray(
        [
            [-3.0, -2.8, 0.1],
            [-2.7, -3.1, 0.0],
            [-3.2, -2.9, 0.2],
            [3.0, 2.8, -0.1],
            [2.7, 3.1, 0.0],
            [3.2, 2.9, -0.2],
        ],
        dtype=np.float32,
    )


def test_pca_coordinates_are_deterministic_and_label_independent(tmp_path):
    fingerprints = _fingerprints()
    coordinates, explained = _prepare_pca(fingerprints, standardize=True)
    repeated, repeated_explained = _prepare_pca(fingerprints, standardize=True)

    assert np.allclose(coordinates, repeated)
    assert np.allclose(explained, repeated_explained)

    first = write_fingerprint_pca_figure(
        fingerprints,
        [f"client_{idx:03d}" for idx in range(6)],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        tmp_path / "first",
        "synthetic",
    )
    second = write_fingerprint_pca_figure(
        fingerprints,
        [f"client_{idx:03d}" for idx in range(6)],
        [1, 0, 1, 0, 1, 0],
        [2, 2, 1, 1, 0, 0],
        tmp_path / "second",
        "synthetic",
    )
    first_npz = np.load(first["npz"])
    second_npz = np.load(second["npz"])
    assert np.allclose(first_npz["pca_coordinates"], second_npz["pca_coordinates"])


def test_publication_outputs_and_metadata_are_written(tmp_path):
    outputs = write_fingerprint_pca_figure(
        _fingerprints(),
        [f"client_{idx:03d}" for idx in range(6)],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        tmp_path,
        "synthetic",
        true_cluster_names={0: "acc", 1: "gyro"},
        visualization_cfg={"png_dpi": 600, "show_ellipses": True},
    )

    for path in outputs.values():
        assert path.is_file()
        assert path.stat().st_size > 0
    assert outputs["pdf"].read_bytes().startswith(b"%PDF")
    metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
    assert metadata["coordinate_input"] == "fingerprints_only"
    assert metadata["labels_used_for_pca"] is False
    assert metadata["true_cluster_usage"] == "post_hoc_audit_coloring_only"
    assert metadata["pdf_format"] == "vector"
    assert metadata["png_dpi"] == 600


def test_invalid_method_and_misaligned_rows_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="method"):
        write_fingerprint_pca_figure(
            _fingerprints(),
            [str(index) for index in range(6)],
            [0] * 6,
            [0] * 6,
            tmp_path,
            "synthetic",
            visualization_cfg={"method": "tsne"},
        )
    with pytest.raises(ValueError, match="equal row counts"):
        write_fingerprint_pca_figure(
            _fingerprints(),
            [str(index) for index in range(5)],
            [0] * 6,
            [0] * 6,
            tmp_path,
            "synthetic",
        )
