import csv
from pathlib import Path

import pytest
import torch

import semantic_split_multimodal.data.partitioner as partitioner
from semantic_split_multimodal.data.partitioner import run_stage1_partition
from semantic_split_multimodal.data.registry import load_dataset
from semantic_split_multimodal.utils.config import load_config
from semantic_split_multimodal.utils.results import configure_result_run


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DATASET_CASES = [
    ("uci_har", "configs/uci_har.yaml", ["acc", "gyro"], [[6, 128], [3, 128]]),
    (
        "mhealth",
        "configs/mhealth.yaml",
        ["accelerometer", "gyroscope", "magnetometer", "ecg"],
        [[9, 128], [6, 128], [6, 128], [2, 128]],
    ),
    (
        "pamap2",
        "configs/pamap2.yaml",
        ["accelerometer", "gyroscope", "magnetometer"],
        [[9, 128], [9, 128], [9, 128]],
    ),
]


@pytest.mark.parametrize("dataset_name,config_path,expected_names,expected_shapes", DATASET_CASES)
def test_dataset_loaders_return_unified_contract(dataset_name, config_path, expected_names, expected_shapes):
    cfg = load_config(PROJECT_ROOT / config_path)
    dataset = load_dataset(cfg, PROJECT_ROOT)

    assert set(dataset) >= {"modality_input_shapes", "modality_names", "root", "test", "train"}
    assert dataset["modality_names"] == expected_names
    assert dataset["modality_input_shapes"] == expected_shapes

    for split_name in ("train", "test"):
        split = dataset[split_name]
        labels = split["labels"]
        modalities = split["modalities"]
        assert torch.is_tensor(labels)
        assert len(modalities) == len(expected_names)
        for modality, expected_shape in zip(modalities, expected_shapes):
            assert torch.is_tensor(modality)
            assert int(modality.shape[0]) == int(labels.shape[0])
            assert [int(v) for v in modality.shape[1:]] == expected_shape

    test_n = int(dataset["test"]["labels"].shape[0])
    assert all(int(x.shape[0]) == test_n for x in dataset["test"]["modalities"])


@pytest.mark.parametrize("dataset_name,config_path,expected_names,expected_shapes", DATASET_CASES)
def test_stage1_writes_metadata_and_naturally_paired_test_payload(
    tmp_path,
    dataset_name,
    config_path,
    expected_names,
    expected_shapes,
):
    cfg = load_config(PROJECT_ROOT / config_path)
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

    with (output_dir / "client_meta.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == info["num_clients"]
    assert all(int(row["num_samples"]) > 0 for row in rows)
    assert {row["encoder_type"] for row in rows} == {cfg["model"]["encoder"]["type"]}

    test_payload = torch.load(output_dir / "test_multimodal.pt", map_location="cpu")
    assert test_payload["modality_names"] == expected_names
    assert test_payload["modality_input_shapes"] == dict(zip(expected_names, expected_shapes))

    labels = test_payload["label"]
    assert torch.is_tensor(labels)
    assert len(test_payload["modalities"]) == len(expected_names)
    for name, expected_shape in zip(expected_names, expected_shapes):
        tensor = test_payload["modalities"][name]
        assert torch.equal(test_payload[name], tensor)
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
        "dataset": {"type": "synthetic_layout"},
        "results": {"base_dir": str(tmp_path / "results")},
        "partition": {"clients_per_modality": 2},
        "model": {"encoder": {"type": "time_series"}},
    }
    cfg = configure_result_run(cfg, tmp_path, stage="stage1_partition", create_new=True)

    info = run_stage1_partition(cfg, tmp_path)

    expected = tmp_path / "results" / "partition" / "synthetic_layout" / "acc-gyro_2clients"
    assert Path(info["output_dir"]) == expected.resolve()
    assert (expected / "train_clients").exists()
    assert not (tmp_path / "results" / "partition" / "synthetic_layout" / "latest_run.txt").exists()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_stage1_partition(cfg, tmp_path)
