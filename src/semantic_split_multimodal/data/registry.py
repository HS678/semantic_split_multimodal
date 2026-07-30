from pathlib import Path
from typing import Callable

import torch

from semantic_split_multimodal.data.datasets import (
    load_cmu_mosei_dataset,
    load_mhealth_dataset,
    load_pamap2_dataset,
    load_uci_har_dataset,
)


DatasetLoader = Callable[[dict, Path], dict]

_DATASET_LOADERS: dict[str, DatasetLoader] = {}


def register_dataset_loader(name: str, loader: DatasetLoader) -> None:
    key = str(name).strip().lower()
    if not key:
        raise ValueError("Dataset loader name cannot be empty.")
    _DATASET_LOADERS[key] = loader


def available_datasets() -> list[str]:
    return sorted(_DATASET_LOADERS)


def load_dataset(cfg: dict, project_root: Path) -> dict:
    dataset_cfg = cfg.get("dataset", {})
    dataset_type = str(dataset_cfg.get("type", "uci_har")).strip().lower()
    if dataset_type not in _DATASET_LOADERS:
        supported = ", ".join(available_datasets()) or "<none>"
        raise ValueError(f"Unsupported dataset.type: {dataset_type}. Supported datasets: {supported}")

    dataset = _DATASET_LOADERS[dataset_type](cfg, project_root)
    validate_dataset_contract(dataset, dataset_type)
    return dataset


def validate_dataset_contract(dataset: dict, dataset_type: str = "<unknown>") -> None:
    required_top = {"train", "validation", "test", "modality_names", "modality_input_shapes"}
    missing_top = sorted(required_top - set(dataset))
    if missing_top:
        raise ValueError(f"Dataset '{dataset_type}' is missing required keys: {missing_top}")

    modality_names = list(dataset["modality_names"])
    input_shapes = list(dataset["modality_input_shapes"])
    if len(modality_names) == 0:
        raise ValueError(f"Dataset '{dataset_type}' must define at least one modality.")
    if len(input_shapes) != len(modality_names):
        raise ValueError(
            f"Dataset '{dataset_type}' has mismatched modality_names and modality_input_shapes lengths: "
            f"{len(modality_names)} vs {len(input_shapes)}"
        )

    for split_name in ("train", "validation", "test"):
        split = dataset[split_name]
        if "modalities" not in split or "labels" not in split:
            raise ValueError(f"Dataset '{dataset_type}' split '{split_name}' must contain modalities and labels.")
        modalities = split["modalities"]
        labels = split["labels"]
        if len(modalities) != len(modality_names):
            raise ValueError(
                f"Dataset '{dataset_type}' split '{split_name}' has {len(modalities)} modalities, "
                f"expected {len(modality_names)}."
            )
        if not torch.is_tensor(labels):
            raise TypeError(f"Dataset '{dataset_type}' split '{split_name}' labels must be a torch.Tensor.")
        n = int(labels.shape[0])
        for idx, x in enumerate(modalities):
            if not torch.is_tensor(x):
                raise TypeError(
                    f"Dataset '{dataset_type}' split '{split_name}' modality {idx} must be a torch.Tensor."
                )
            if int(x.shape[0]) != n:
                raise ValueError(
                    f"Dataset '{dataset_type}' split '{split_name}' modality {idx} sample count "
                    f"{int(x.shape[0])} does not match labels {n}."
                )
            actual_shape = [int(v) for v in x.shape[1:]]
            expected_shape = [int(v) for v in input_shapes[idx]]
            if actual_shape != expected_shape:
                raise ValueError(
                    f"Dataset '{dataset_type}' split '{split_name}' modality {idx} shape "
                    f"{actual_shape} does not match modality_input_shapes {expected_shape}."
                )


register_dataset_loader("cmu_mosei", load_cmu_mosei_dataset)
register_dataset_loader("uci_har", load_uci_har_dataset)
register_dataset_loader("mhealth", load_mhealth_dataset)
register_dataset_loader("pamap2", load_pamap2_dataset)
