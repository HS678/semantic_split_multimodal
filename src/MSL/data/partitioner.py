import csv
import json
from pathlib import Path

import torch

from MSL.data.client import Client
from MSL.data.registry import load_dataset
from MSL.utils.results import partition_signature


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


def run_stage1_partition(cfg: dict, project_root: Path):
    dataset = load_dataset(cfg, project_root)
    partition_cfg = cfg.get("partition", {})
    output_dir = resolve_project_path(project_root, partition_cfg.get("output_dir", "results/MSL/partition"))
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
        raise FileExistsError(f"Refusing to overwrite existing Stage1 partition directory: {output_dir}")
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
