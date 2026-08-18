# 数据集统一 dispatcher、client partition 生成和 artifact 读写逻辑。
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Client:
    client_id: str
    hidden_modality_id: int
    samples: torch.Tensor
    labels: torch.Tensor
    encoder_type: str
    pred_cluster: Optional[int] = None
    input_shape: Optional[list[int]] = None
    sequence_lengths: Optional[torch.Tensor] = None

    @classmethod
    def from_payload(cls, payload: dict, pred_cluster: Optional[int] = None):
        return cls(
            client_id=str(payload["client_id"]),
            hidden_modality_id=int(payload["hidden_modality_id"]),
            samples=payload["samples"],
            labels=payload["labels"],
            encoder_type=str(payload.get("encoder_type", "time_series")),
            pred_cluster=None if pred_cluster is None else int(pred_cluster),
            input_shape=[int(v) for v in payload.get("input_shape", list(payload["samples"].shape[1:]))],
            sequence_lengths=payload.get("sequence_lengths"),
        )

    def training_view(self):
        return {
            "client_id": self.client_id,
            "samples": self.samples,
            "labels": self.labels,
            "encoder_type": self.encoder_type,
            "pred_cluster": self.pred_cluster,
            "input_shape": self.input_shape,
            "sequence_lengths": self.sequence_lengths,
        }

    def to_payload(self):
        return {
            "client_id": self.client_id,
            "hidden_modality_id": int(self.hidden_modality_id),
            "samples": self.samples,
            "labels": self.labels,
            "encoder_type": self.encoder_type,
            "pred_cluster": self.pred_cluster,
            "input_shape": [int(v) for v in (self.input_shape or list(self.samples.shape[1:]))],
            "sequence_lengths": self.sequence_lengths,
        }


import csv
import json
from pathlib import Path
from typing import Callable

from MSL.utils import partition_signature


def _validate_subject_splits(train_subjects, test_subjects, dataset_name):
    splits = {
        "train": {int(v) for v in train_subjects},
        "test": {int(v) for v in test_subjects},
    }
    for split_name, subjects in splits.items():
        if not subjects:
            raise ValueError(f"{dataset_name} {split_name}_subjects must not be empty.")
    overlap = sorted(splits["train"] & splits["test"])
    if overlap:
        raise ValueError(
            f"{dataset_name} subject splits must be disjoint, train/test overlap={overlap}."
        )
    return splits

def _fold_number(split_protocol: str) -> int:
    """从 split_protocol（如 subject_5fold_fold1 / session_5fold_loso_fold3）提取折号。"""
    import re
    match = re.search(r"fold(\d+)", str(split_protocol or ""))
    if not match:
        raise ValueError(f"split_protocol must contain fold<N>: {split_protocol!r}")
    return int(match.group(1))

def resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()

# train内部随机划分为单模态客户端
def _split_indices(num_samples: int, num_clients: int, seed: int):
    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(num_samples, generator=generator)
    return [chunk.clone() for chunk in torch.tensor_split(perm, int(num_clients))]


# 按照train的lable比例划分单模态客户端，保证每个客户端的label比例和全局train一致
def _split_indices_stratified(
    labels: torch.Tensor,
    num_clients: int,
    seed: int
):
    generator = torch.Generator().manual_seed(int(seed))

    client_indices = [
        []
        for _ in range(num_clients)
    ]

    unique_labels = torch.unique(labels)

    for label in unique_labels:

        label_indices = torch.where(
            labels == label
        )[0]

        perm = torch.randperm(
            len(label_indices),
            generator=generator
        )

        label_indices = label_indices[perm]

        chunks = torch.tensor_split(
            label_indices,
            num_clients
        )

        for client_id, chunk in enumerate(chunks):
            client_indices[client_id].extend(
                chunk.tolist()
            )

    result = []

    for indices in client_indices:

        indices = torch.tensor(
            indices,
            dtype=torch.long
        )

        perm = torch.randperm(
            len(indices),
            generator=generator
        )

        result.append(indices[perm])

    return result


# debug调式信息，输出训练客户端的label分布情况
def _debug_client_label_distribution(
    labels: torch.Tensor,
    splits,
    modality_name: str
):
    print("\n" + "=" * 60)
    print(f"Modality: {modality_name}")
    print("=" * 60)

    unique_labels = torch.unique(labels)

    # 全局train分布
    total = len(labels)

    print("\nGlobal train distribution:")
    for label in unique_labels:
        count = torch.sum(labels == label).item()
        ratio = count / total * 100

        print(
            f"Label {label.item()}: "
            f"{count} samples ({ratio:.2f}%)"
        )


    # client分布
    for client_id, idx in enumerate(splits):

        client_labels = labels[idx]
        client_total = len(client_labels)

        print(f"\nclient_{client_id:03d}")

        for label in unique_labels:
            count = torch.sum(
                client_labels == label
            ).item()

            ratio = count / client_total * 100

            print(
                f"  Label {label.item()}: "
                f"{ratio:.2f}%"
            )


def prepare_clients(cfg: dict, project_root: Path):
    dataset = load_dataset(cfg, project_root)
    partition_cfg = cfg.get("partition", {})
    output_dir = resolve_project_path(project_root, partition_cfg.get("output_dir", "results/pipeline/clients"))
    train = dataset["train"]
    test = dataset["test"]
    modality_names = dataset["modality_names"]
    input_shapes = dataset.get("modality_input_shapes", [list(train["modalities"][i].shape[1:]) for i in range(len(modality_names))])
    modality_encoder_types = dataset.get("modality_encoder_types")
    if modality_encoder_types is None:
        default_encoder_type = cfg.get("partition", {}).get("encoder_type") or cfg.get("model", {}).get("encoder", {}).get("type", "time_series")
        modality_encoder_types = [str(default_encoder_type)] * len(modality_names)
    clients_per_modality = int(partition_cfg.get("clients_per_modality", cfg.get("clients_per_modality", 10)))
    split_protocol = str(cfg.get("dataset", {}).get("split_protocol", "subject_disjoint"))
    if bool(partition_cfg.get("auto_signature_dir", False)):
        output_dir = output_dir / partition_signature(modality_names, clients_per_modality, split_protocol)
    if output_dir.exists() and any(output_dir.iterdir()) and not bool(partition_cfg.get("allow_existing", False)):
        raise FileExistsError(f"Refusing to overwrite existing client partition directory: {output_dir}")
    train_clients_dir = output_dir / "train_clients"
    train_clients_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 42))

    client_rows = []
    client_id_num = 0
    for modality_id, modality_name in enumerate(modality_names):
        # splits = _split_indices(int(train["labels"].shape[0]), clients_per_modality, seed + modality_id)
        splits = _split_indices_stratified(train["labels"], clients_per_modality, seed + modality_id)
        _debug_client_label_distribution(train["labels"], splits, modality_name)
        for idx in splits:
            client_id = f"client_{client_id_num:03d}"
            encoder_type = str(modality_encoder_types[modality_id])
            modality_lengths = train.get("modality_lengths")
            sequence_lengths = None if modality_lengths is None else modality_lengths[modality_id][idx].contiguous()
            payload = Client(
                client_id=client_id,
                hidden_modality_id=int(modality_id),
                samples=train["modalities"][modality_id][idx].contiguous(),
                labels=train["labels"][idx].contiguous(),
                encoder_type=str(encoder_type),
                input_shape=[int(v) for v in input_shapes[modality_id]],
                sequence_lengths=sequence_lengths,
            ).to_payload()
            payload["hidden_modality_name"] = modality_name
            torch.save(payload, train_clients_dir / f"{client_id}.pt")
            client_rows.append(
                {
                    "client_id": client_id,
                    "hidden_modality_id": int(modality_id),
                    "hidden_modality_name": modality_name,
                    "num_samples": int(idx.numel()),
                    "encoder_type": str(encoder_type),
                }
            )
            client_id_num += 1

    test_modalities = {
        name: test["modalities"][idx].contiguous()
        for idx, name in enumerate(modality_names)
    }
    test_payload = {
        "label": test["labels"].contiguous(),
        "modalities": test_modalities,
        "modality_names": list(modality_names),
        "modality_input_shapes": {
            name: [int(v) for v in shape]
            for name, shape in zip(modality_names, input_shapes)
        },
        "split": "test",
    }
    if test.get("sample_ids") is not None:
        test_payload["sample_ids"] = list(test["sample_ids"])
    for group_key in ("subject_ids", "session_ids"):
        if test.get(group_key) is not None:
            test_payload[group_key] = test[group_key].contiguous()
    if test.get("modality_lengths") is not None:
        test_payload["modality_lengths"] = {
            name: test["modality_lengths"][idx].contiguous()
            for idx, name in enumerate(modality_names)
        }
    for name, tensor in test_modalities.items():
        test_payload[name] = tensor
    torch.save(test_payload, output_dir / "test_multimodal.pt")

    with (output_dir / "client_meta.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["client_id", "hidden_modality_id", "hidden_modality_name", "num_samples", "encoder_type"],
        )
        writer.writeheader()
        writer.writerows(client_rows)

    partition_config = {
        "dataset_type": cfg.get("dataset", {}).get("type", "uci_har"),
        "dataset_root": dataset["root"],
        "dataset_config": cfg.get("dataset", {}),
        "label_mapping": dataset.get("label_mapping"),
        "output_dir": str(output_dir),
        "clients_per_modality": clients_per_modality,
        "num_clients": len(client_rows),
        "split_protocol": split_protocol,
        "split_subjects": dataset.get("split_subjects", {}),
        "split_num_samples": {
            "train": int(train["labels"].shape[0]),
            "test": int(test["labels"].shape[0]),
        },
        "split_sample_ids": {
            "train": list(train.get("sample_ids", [])),
            "test": list(test.get("sample_ids", [])),
        },
        "split_group_ids": {
            "train_subjects": (
                sorted({int(v) for v in train["subject_ids"].tolist()})
                if train.get("subject_ids") is not None
                else None
            ),
            "test_subjects": (
                sorted({int(v) for v in test["subject_ids"].tolist()})
                if test.get("subject_ids") is not None
                else None
            ),
            "train_sessions": (
                sorted({int(v) for v in train["session_ids"].tolist()})
                if train.get("session_ids") is not None
                else None
            ),
            "test_sessions": (
                sorted({int(v) for v in test["session_ids"].tolist()})
                if test.get("session_ids") is not None
                else None
            ),
        },
        "modalities": [
            {
                "hidden_modality_id": i,
                "hidden_modality_name": name,
                "input_shape": [int(v) for v in input_shapes[i]],
                "encoder_type": str(modality_encoder_types[i]),
            }
            for i, name in enumerate(modality_names)
        ],
        "seed": seed,
    }
    with (output_dir / "partition_config.json").open("w", encoding="utf-8") as f:
        json.dump(partition_config, f, indent=2)

    return partition_config

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
    required_top = {"train", "test", "modality_names", "modality_input_shapes"}
    missing_top = sorted(required_top - set(dataset))
    if missing_top:
        raise ValueError(f"Dataset '{dataset_type}' is missing required keys: {missing_top}")

    modality_names = list(dataset["modality_names"])
    input_shapes = list(dataset["modality_input_shapes"])
    encoder_types = list(dataset.get("modality_encoder_types", []))
    if len(modality_names) == 0:
        raise ValueError(f"Dataset '{dataset_type}' must define at least one modality.")
    if len(input_shapes) != len(modality_names):
        raise ValueError(
            f"Dataset '{dataset_type}' has mismatched modality_names and modality_input_shapes lengths: "
            f"{len(modality_names)} vs {len(input_shapes)}"
        )
    if encoder_types and len(encoder_types) != len(modality_names):
        raise ValueError(
            f"Dataset '{dataset_type}' has mismatched modality_names and modality_encoder_types lengths: "
            f"{len(modality_names)} vs {len(encoder_types)}"
        )

    for split_name in ("train", "test"):
        split = dataset[split_name]
        if "modalities" not in split or "labels" not in split:
            raise ValueError(f"Dataset '{dataset_type}' split '{split_name}' must contain modalities and labels.")
        modalities = split["modalities"]
        labels = split["labels"]
        modality_lengths = split.get("modality_lengths")
        if len(modalities) != len(modality_names):
            raise ValueError(
                f"Dataset '{dataset_type}' split '{split_name}' has {len(modalities)} modalities, "
                f"expected {len(modality_names)}."
            )
        if modality_lengths is not None and len(modality_lengths) != len(modality_names):
            raise ValueError(
                f"Dataset '{dataset_type}' split '{split_name}' has invalid modality_lengths count."
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
            if n > 0 and actual_shape != expected_shape:
                raise ValueError(
                    f"Dataset '{dataset_type}' split '{split_name}' modality {idx} shape "
                    f"{actual_shape} does not match modality_input_shapes {expected_shape}."
                )
            if modality_lengths is not None:
                lengths = modality_lengths[idx]
                if not torch.is_tensor(lengths) or lengths.ndim != 1 or int(lengths.shape[0]) != n:
                    raise ValueError(
                        f"Dataset '{dataset_type}' split '{split_name}' modality {idx} lengths "
                        "must be a 1D tensor aligned with labels."
                    )
                if x.ndim < 3 or torch.any(lengths <= 0) or torch.any(lengths > int(x.shape[1])):
                    raise ValueError(
                        f"Dataset '{dataset_type}' split '{split_name}' modality {idx} has invalid lengths."
                    )


from MSL.datasets.uci_har import load_uci_har_dataset
from MSL.datasets.mhealth import load_mhealth_dataset
from MSL.datasets.pamap2 import load_pamap2_dataset
from MSL.datasets.iemocap import load_iemocap_dataset


register_dataset_loader("uci_har", load_uci_har_dataset)
register_dataset_loader("mhealth", load_mhealth_dataset)
register_dataset_loader("pamap2", load_pamap2_dataset)
register_dataset_loader("iemocap", load_iemocap_dataset)
