from pathlib import Path
import numpy as np
import torch


REQUIRED_SIGNAL_FILES = {
    "train": [
        "body_acc_x_train.txt",
        "body_acc_y_train.txt",
        "body_acc_z_train.txt",
        "total_acc_x_train.txt",
        "total_acc_y_train.txt",
        "total_acc_z_train.txt",
        "body_gyro_x_train.txt",
        "body_gyro_y_train.txt",
        "body_gyro_z_train.txt",
    ],
    "test": [
        "body_acc_x_test.txt",
        "body_acc_y_test.txt",
        "body_acc_z_test.txt",
        "total_acc_x_test.txt",
        "total_acc_y_test.txt",
        "total_acc_z_test.txt",
        "body_gyro_x_test.txt",
        "body_gyro_y_test.txt",
        "body_gyro_z_test.txt",
    ],
}


def _read_signal_matrix(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required UCI-HAR inertial signal file not found: {path}")
    return np.loadtxt(path, dtype=np.float32)


def validate_uci_har_root(root: Path):
    if not root.exists():
        raise FileNotFoundError(
            f"UCI-HAR root not found: {root}. Expected train/test folders with 'Inertial Signals' files."
        )
    missing = []
    for split, files in REQUIRED_SIGNAL_FILES.items():
        split_dir = root / split
        sig_dir = split_dir / "Inertial Signals"
        if not split_dir.exists():
            missing.append(str(split_dir))
        if not sig_dir.exists():
            missing.append(str(sig_dir))
        for name in files:
            path = sig_dir / name
            if not path.exists():
                missing.append(str(path))
        label_path = split_dir / f"y_{split}.txt"
        if not label_path.exists():
            missing.append(str(label_path))
    if missing:
        preview = "\n".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f"\n... and {len(missing) - 20} more"
        raise FileNotFoundError(f"UCI-HAR dataset is incomplete under {root}. Missing:\n{preview}{suffix}")


def _build_modality_vectors(root: Path, split: str):
    split_dir = root / split
    sig = split_dir / "Inertial Signals"

    body_acc_x = _read_signal_matrix(sig / f"body_acc_x_{split}.txt")
    body_acc_y = _read_signal_matrix(sig / f"body_acc_y_{split}.txt")
    body_acc_z = _read_signal_matrix(sig / f"body_acc_z_{split}.txt")
    total_acc_x = _read_signal_matrix(sig / f"total_acc_x_{split}.txt")
    total_acc_y = _read_signal_matrix(sig / f"total_acc_y_{split}.txt")
    total_acc_z = _read_signal_matrix(sig / f"total_acc_z_{split}.txt")

    body_gyro_x = _read_signal_matrix(sig / f"body_gyro_x_{split}.txt")
    body_gyro_y = _read_signal_matrix(sig / f"body_gyro_y_{split}.txt")
    body_gyro_z = _read_signal_matrix(sig / f"body_gyro_z_{split}.txt")

    labels = np.loadtxt(split_dir / f"y_{split}.txt", dtype=np.int64) - 1
    acc = np.stack(
        [body_acc_x, body_acc_y, body_acc_z, total_acc_x, total_acc_y, total_acc_z],
        axis=1,
    ).astype(np.float32)
    gyro = np.stack([body_gyro_x, body_gyro_y, body_gyro_z], axis=1).astype(np.float32)

    return {
        "modalities": [torch.tensor(acc, dtype=torch.float32), torch.tensor(gyro, dtype=torch.float32)],
        "labels": torch.tensor(labels, dtype=torch.long),
        "modality_input_shapes": [[int(acc.shape[1]), int(acc.shape[2])], [int(gyro.shape[1]), int(gyro.shape[2])]],
        "modality_names": ["acc", "gyro"],
    }


def load_uci_har_dataset(cfg, project_root: Path):
    dataset_cfg = cfg.get("dataset", {})
    root_cfg = dataset_cfg.get("root", "./data/uci-har")
    root = Path(root_cfg)
    if not root.is_absolute():
        root = project_root / root
    root = root.resolve()

    validate_uci_har_root(root)
    train = _build_modality_vectors(root, "train")
    test = _build_modality_vectors(root, "test")
    return {
        "train": train,
        "test": test,
        "root": str(root),
        "modality_input_shapes": train["modality_input_shapes"],
        "modality_names": train["modality_names"],
    }
