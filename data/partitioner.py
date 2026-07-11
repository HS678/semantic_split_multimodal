import csv
import json
from pathlib import Path

import torch

from data.uci_har_adapter import load_uci_har_dataset


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
    dataset = load_uci_har_dataset(cfg, project_root)
    partition_cfg = cfg.get("partition", {})
    output_dir = resolve_project_path(project_root, partition_cfg.get("output_dir", "data_partition"))
    train_clients_dir = output_dir / "train_clients"
    train_clients_dir.mkdir(parents=True, exist_ok=True)

    train = dataset["train"]
    test = dataset["test"]
    modality_names = dataset["modality_names"]
    input_dims = dataset["modality_input_dims"]
    clients_per_modality = int(partition_cfg.get("clients_per_modality", cfg.get("clients_per_modality", 10)))
    seed = int(cfg.get("seed", 42))

    client_rows = []
    client_id_num = 0
    for modality_id, modality_name in enumerate(modality_names):
        splits = _split_indices(int(train["labels"].shape[0]), clients_per_modality, seed + modality_id)
        for idx in splits:
            client_id = f"client_{client_id_num:03d}"
            payload = {
                "client_id": client_id,
                "modality_id": int(modality_id),
                "modality_name": modality_name,
                "x": train["modalities"][modality_id][idx].contiguous(),
                "y": train["labels"][idx].contiguous(),
                "input_dim": int(input_dims[modality_id]),
            }
            torch.save(payload, train_clients_dir / f"{client_id}.pt")
            client_rows.append(
                {
                    "client_id": client_id,
                    "modality_id": int(modality_id),
                    "modality_name": modality_name,
                    "num_samples": int(idx.numel()),
                }
            )
            client_id_num += 1

    test_payload = {
        "acc": test["modalities"][0].contiguous(),
        "gyro": test["modalities"][1].contiguous(),
        "label": test["labels"].contiguous(),
        "modalities": {
            "acc": test["modalities"][0].contiguous(),
            "gyro": test["modalities"][1].contiguous(),
        },
        "modality_input_dims": {name: int(dim) for name, dim in zip(modality_names, input_dims)},
    }
    torch.save(test_payload, output_dir / "test_multimodal.pt")

    with (output_dir / "client_meta.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["client_id", "modality_id", "modality_name", "num_samples"])
        writer.writeheader()
        writer.writerows(client_rows)

    partition_config = {
        "dataset_root": dataset["root"],
        "output_dir": str(output_dir),
        "clients_per_modality": clients_per_modality,
        "num_clients": len(client_rows),
        "modalities": [
            {"modality_id": i, "modality_name": name, "input_dim": int(input_dims[i])}
            for i, name in enumerate(modality_names)
        ],
        "seed": seed,
    }
    with (output_dir / "partition_config.json").open("w", encoding="utf-8") as f:
        json.dump(partition_config, f, indent=2)

    return partition_config
