from pathlib import Path
import numpy as np
import torch


def _read_signal_matrix(path: Path):
    return np.loadtxt(path, dtype=np.float32)


def _build_modality_vectors(root: Path, split: str):
    split_dir = root / split
    sig = split_dir / "Inertial Signals"

    acc_x = _read_signal_matrix(sig / f"body_acc_x_{split}.txt")
    acc_y = _read_signal_matrix(sig / f"body_acc_y_{split}.txt")
    acc_z = _read_signal_matrix(sig / f"body_acc_z_{split}.txt")
    gyro_x = _read_signal_matrix(sig / f"body_gyro_x_{split}.txt")
    gyro_y = _read_signal_matrix(sig / f"body_gyro_y_{split}.txt")
    gyro_z = _read_signal_matrix(sig / f"body_gyro_z_{split}.txt")

    labels = np.loadtxt(split_dir / f"y_{split}.txt", dtype=np.int64) - 1

    n = labels.shape[0]
    t = acc_x.shape[1]

    # Shared 9-channel layout for both modalities to keep one encoder input_dim.
    acc_only = np.zeros((n, 9, t), dtype=np.float32)
    gyro_only = np.zeros((n, 9, t), dtype=np.float32)

    acc_only[:, 0, :] = acc_x
    acc_only[:, 1, :] = acc_y
    acc_only[:, 2, :] = acc_z
    acc_only[:, 3, :] = acc_x
    acc_only[:, 4, :] = acc_y
    acc_only[:, 5, :] = acc_z

    gyro_only[:, 6, :] = gyro_x
    gyro_only[:, 7, :] = gyro_y
    gyro_only[:, 8, :] = gyro_z

    acc_vec = acc_only.reshape(n, -1)
    gyro_vec = gyro_only.reshape(n, -1)

    return {
        "modalities": [torch.tensor(acc_vec, dtype=torch.float32), torch.tensor(gyro_vec, dtype=torch.float32)],
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def load_uci_har_dataset(cfg, project_root: Path):
    dataset_cfg = cfg.get("dataset", {})
    root_cfg = dataset_cfg.get("root", "./data/uci-har")
    root = Path(root_cfg)
    if not root.is_absolute():
        root = project_root / root
    root = root.resolve()

    if not root.exists():
        raise FileNotFoundError(f"UCI-HAR root not found: {root}")

    train = _build_modality_vectors(root, "train")
    test = _build_modality_vectors(root, "test")

    return {"train": train, "test": test, "root": str(root)}
