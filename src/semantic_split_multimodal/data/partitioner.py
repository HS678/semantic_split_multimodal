import csv
import json
from pathlib import Path

import torch

from semantic_split_multimodal.data.client import Client
from semantic_split_multimodal.data.registry import load_dataset
from semantic_split_multimodal.utils.results import partition_signature


def resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _split_indices(num_samples: int, num_clients: int, seed: int):
    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(num_samples, generator=generator)
    return [chunk.clone() for chunk in torch.tensor_split(perm, int(num_clients))]


def run_stage1_partition(cfg: dict, project_root: Path):
    dataset = load_dataset(cfg, project_root)
    partition_cfg = cfg.get("partition", {})
    output_dir = resolve_project_path(project_root, partition_cfg.get("output_dir", "local/results/data_partition"))
    train = dataset["train"]
    validation = dataset["validation"]
    test = dataset["test"]
    modality_names = dataset["modality_names"]
    input_shapes = dataset.get("modality_input_shapes", [list(train["modalities"][i].shape[1:]) for i in range(len(modality_names))])
    clients_per_modality = int(partition_cfg.get("clients_per_modality", cfg.get("clients_per_modality", 10)))
    split_protocol = str(cfg.get("dataset", {}).get("split_protocol", "subject_disjoint_tvt_v1"))
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
        splits = _split_indices(int(train["labels"].shape[0]), clients_per_modality, seed + modality_id)
        for idx in splits:
            client_id = f"client_{client_id_num:03d}"
            encoder_type = cfg.get("partition", {}).get("encoder_type") or cfg.get("model", {}).get("encoder", {}).get("type", "time_series")
            payload = Client(
                client_id=client_id,
                hidden_modality_id=int(modality_id),
                samples=train["modalities"][modality_id][idx].contiguous(),
                labels=train["labels"][idx].contiguous(),
                encoder_type=str(encoder_type),
                input_shape=[int(v) for v in input_shapes[modality_id]],
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

    for split_name, split in [("validation", validation), ("test", test)]:
        split_modalities = {
            name: split["modalities"][idx].contiguous()
            for idx, name in enumerate(modality_names)
        }
        split_payload = {
            "label": split["labels"].contiguous(),
            "modalities": split_modalities,
            "modality_names": list(modality_names),
            "modality_input_shapes": {
                name: [int(v) for v in shape]
                for name, shape in zip(modality_names, input_shapes)
            },
            "split": split_name,
        }
        for name, tensor in split_modalities.items():
            split_payload[name] = tensor
        torch.save(split_payload, output_dir / f"{split_name}_multimodal.pt")

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
        "split_subjects": {
            split_name: [int(v) for v in cfg.get("dataset", {}).get(f"{split_name}_subjects", [])]
            for split_name in ("train", "validation", "test")
        },
        "split_num_samples": {
            "train": int(train["labels"].shape[0]),
            "validation": int(validation["labels"].shape[0]),
            "test": int(test["labels"].shape[0]),
        },
        "modalities": [
            {
                "hidden_modality_id": i,
                "hidden_modality_name": name,
                "input_shape": [int(v) for v in input_shapes[i]],
            }
            for i, name in enumerate(modality_names)
        ],
        "seed": seed,
    }
    with (output_dir / "partition_config.json").open("w", encoding="utf-8") as f:
        json.dump(partition_config, f, indent=2)

    return partition_config
