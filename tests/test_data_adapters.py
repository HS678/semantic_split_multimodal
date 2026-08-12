import csv
from pathlib import Path

import pytest
import torch

import MSL.data.partitioner as partitioner
from MSL.data.partitioner import run_stage1_partition
from MSL.data.registry import load_dataset
from MSL.data.datasets import (
    PAMAP2_ACTIVITY_IDS,
    _normalize_from_train,
    _pamap2_remap_labels,
    _validate_subject_splits,
)
from MSL.utils.experiment_args import apply_experiment_overrides, build_experiment_config
from MSL.utils.results import configure_result_run


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DATASET_CASES = [
    ("uci_har", None, ["acc", "gyro"], [[6, 128], [3, 128]]),
    (
        "mhealth",
        1,
        ["acc", "gyro", "mag", "ecg"],
        [[9, 128], [6, 128], [6, 128], [2, 128]],
    ),
    (
        "pamap2",
        1,
        ["acc", "gyro", "mag"],
        [[9, 200], [9, 200], [9, 200]],
    ),
]


def _dataset_cfg(dataset_name: str, fold: int | None = None) -> dict:
    cfg = build_experiment_config(dataset_type=dataset_name)
    return apply_experiment_overrides(cfg, fold=fold) if fold is not None else cfg


@pytest.mark.parametrize("dataset_name,fold,expected_names,expected_shapes", DATASET_CASES)
def test_dataset_loaders_return_unified_contract(dataset_name, fold, expected_names, expected_shapes):
    cfg = _dataset_cfg(dataset_name, fold)
    dataset = load_dataset(cfg, PROJECT_ROOT)

    assert set(dataset) >= {"modality_input_shapes", "modality_names", "root", "train", "test"}
    assert dataset["modality_names"] == expected_names
    assert dataset["modality_input_shapes"] == expected_shapes

    for split_name in ("train", "test"):
        split = dataset[split_name]
        labels = split["labels"]
        modalities = split["modalities"]
        assert torch.is_tensor(labels)
        assert len(modalities) == len(expected_names)
        n = int(labels.shape[0])
        for modality, expected_shape in zip(modalities, expected_shapes):
            assert torch.is_tensor(modality)
            assert int(modality.shape[0]) == n
            if n > 0:
                assert [int(v) for v in modality.shape[1:]] == expected_shape

    test_n = int(dataset["test"]["labels"].shape[0])
    assert all(int(x.shape[0]) == test_n for x in dataset["test"]["modalities"])


@pytest.mark.parametrize("dataset_name,fold,expected_names,expected_shapes", DATASET_CASES)
def test_stage1_writes_metadata_and_naturally_paired_test_payloads(
    tmp_path,
    dataset_name,
    fold,
    expected_names,
    expected_shapes,
):
    cfg = _dataset_cfg(dataset_name, fold)
    cfg["partition"] = {
        **cfg.get("partition", {}),
        "output_dir": str(tmp_path / dataset_name / "01_dataset_partition"),
        "clients_per_modality": 2,
    }

    info = run_stage1_partition(cfg, PROJECT_ROOT)
    output_dir = Path(info["output_dir"])
    assert info["num_clients"] == 2 * len(expected_names)
    assert [item["hidden_modality_name"] for item in info["modalities"]] == expected_names
    assert [item["input_shape"] for item in info["modalities"]] == expected_shapes
    assert info["split_protocol"] == cfg["dataset"]["split_protocol"]
    assert info["dataset_config"] == cfg["dataset"]
    assert set(info["split_num_samples"]) == {"train", "test"}
    assert info["split_num_samples"]["train"] > 0
    assert info["split_num_samples"]["test"] > 0

    with (output_dir / "client_meta.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == info["num_clients"]
    assert all(int(row["num_samples"]) > 0 for row in rows)
    assert {row["encoder_type"] for row in rows} == {cfg["model"]["encoder"]["type"]}

    for split_name in ("test",):
        path = output_dir / f"{split_name}_multimodal.pt"
        payload = torch.load(path, map_location="cpu")
        assert payload["split"] == split_name
        assert payload["modality_names"] == expected_names
        assert payload["modality_input_shapes"] == dict(zip(expected_names, expected_shapes))

        labels = payload["label"]
        assert torch.is_tensor(labels)
        assert len(payload["modalities"]) == len(expected_names)
        for name, expected_shape in zip(expected_names, expected_shapes):
            tensor = payload["modalities"][name]
            assert torch.equal(payload[name], tensor)
            assert int(tensor.shape[0]) == int(labels.shape[0])
            assert [int(v) for v in tensor.shape[1:]] == expected_shape


def test_stage1_auto_partition_layout_uses_modality_signature_and_refuses_overwrite(monkeypatch, tmp_path):
    def fake_load_dataset(_cfg, _root):
        train_labels = torch.tensor([0, 1, 0, 1])
        test_labels = torch.tensor([0, 1])
        return {
            "root": str(tmp_path / "dataset"),
            "modality_names": ["acc", "gyro"],
            "modality_input_shapes": [[1], [1]],
            "train": {
                "labels": train_labels,
                "modalities": [torch.zeros(4, 1), torch.ones(4, 1)],
            },
            "test": {
                "labels": test_labels,
                "modalities": [torch.zeros(2, 1), torch.ones(2, 1)],
            },
        }

    monkeypatch.setattr(partitioner, "load_dataset", fake_load_dataset)
    cfg = {
        "seed": 3,
        "dataset": {"type": "synthetic_layout", "split_protocol": "subject_disjoint_tvt_v1"},
        "results": {"base_dir": str(tmp_path / "results")},
        "partition": {"clients_per_modality": 2},
        "model": {"encoder": {"type": "time_series"}},
    }
    cfg = configure_result_run(cfg, tmp_path)

    info = run_stage1_partition(cfg, tmp_path)

    expected = (
        tmp_path
        / "results"
        / "partition"
        / "synthetic_layout"
        / "acc_2clients_gyro_2clients__subject_disjoint_tvt_v1"
    )
    assert Path(info["output_dir"]) == expected.resolve()
    assert (expected / "train_clients").exists()
    assert not (tmp_path / "results" / "partition" / "synthetic_layout" / "latest_run.txt").exists()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_stage1_partition(cfg, tmp_path)


def test_subject_splits_must_be_non_empty_and_disjoint():
    splits = _validate_subject_splits([1, 2], [4], "synthetic")
    assert splits == {"train": {1, 2}, "test": {4}}

    with pytest.raises(ValueError, match="must be disjoint"):
        _validate_subject_splits([1, 2], [2], "synthetic")
    with pytest.raises(ValueError, match="must not be empty"):
        _validate_subject_splits([], [2], "synthetic")
    with pytest.raises(ValueError, match="must not be empty"):
        _validate_subject_splits([1], [], "synthetic")


def test_normalization_statistics_are_fitted_on_train_only():
    train = {
        "modalities": [torch.tensor([[[1.0, 3.0]], [[5.0, 7.0]]])],
        "labels": torch.tensor([0, 1]),
    }
    test = {
        "modalities": [torch.tensor([[[9.0, 11.0]]])],
        "labels": torch.tensor([0]),
    }

    normalized_train, normalized_test = _normalize_from_train(train, test)
    train_mean = train["modalities"][0].mean(dim=(0, 2), keepdim=True)
    train_std = train["modalities"][0].std(dim=(0, 2), keepdim=True)

    assert torch.allclose(normalized_train["modalities"][0].mean(dim=(0, 2)), torch.zeros(1), atol=1e-6)
    assert torch.allclose(
        normalized_test["modalities"][0],
        (test["modalities"][0] - train_mean) / train_std,
    )


def test_pamap2_label_mapping_is_fixed_without_test_label_union():
    train = {"modalities": [], "labels": torch.tensor([1, 24])}
    test = {"modalities": [], "labels": torch.tensor([17])}

    remapped_train, remapped_test, mapping = _pamap2_remap_labels(
        train,
        test,
    )

    assert mapping == {activity_id: idx for idx, activity_id in enumerate(PAMAP2_ACTIVITY_IDS)}
    assert remapped_train["labels"].tolist() == [mapping[1], mapping[24]]
    assert remapped_test["labels"].tolist() == [mapping[17]]
