from pathlib import Path
from typing import Callable

import torch

from data.uci_har_adapter import load_uci_har_dataset


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
    required_top = {"train", "test", "modality_names", "modality_input_dims"}
    missing_top = sorted(required_top - set(dataset))
    if missing_top:
        raise ValueError(f"Dataset '{dataset_type}' is missing required keys: {missing_top}")

    modality_names = list(dataset["modality_names"])
    input_dims = list(dataset["modality_input_dims"])
    if len(modality_names) == 0:
        raise ValueError(f"Dataset '{dataset_type}' must define at least one modality.")
    if len(modality_names) != len(input_dims):
        raise ValueError(
            f"Dataset '{dataset_type}' has mismatched modality_names and modality_input_dims lengths: "
            f"{len(modality_names)} vs {len(input_dims)}"
        )

    for split_name in ("train", "test"):
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
            if int(x.reshape(n, -1).shape[1]) != int(input_dims[idx]):
                raise ValueError(
                    f"Dataset '{dataset_type}' split '{split_name}' modality {idx} flattened dim "
                    f"{int(x.reshape(n, -1).shape[1])} does not match modality_input_dims {int(input_dims[idx])}."
                )


register_dataset_loader("uci_har", load_uci_har_dataset)
