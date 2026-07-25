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
    root_cfg = dataset_cfg.get("root", "./local/datasets/uci_har")
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


MHEALTH_POSITION_MODALITIES = {
    "chest": list(range(0, 5)),
    "left_ankle": list(range(5, 14)),
    "right_lower_arm": list(range(14, 23)),
}

MHEALTH_SENSOR_TYPE_MODALITIES = {
    "accelerometer": [0, 1, 2, 5, 6, 7, 14, 15, 16],
    "gyroscope": [8, 9, 10, 17, 18, 19],
    "magnetometer": [11, 12, 13, 20, 21, 22],
    "ecg": [3, 4],
}


def _mhealth_resolve_modalities(dataset_cfg):
    scheme = str(dataset_cfg.get("modality_scheme", "sensor_type")).lower()
    if scheme == "sensor_type":
        return MHEALTH_SENSOR_TYPE_MODALITIES
    if scheme == "position":
        return MHEALTH_POSITION_MODALITIES
    raise ValueError("dataset.modality_scheme must be 'sensor_type' or 'position'.")


def _mhealth_resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _mhealth_subject_path(root: Path, subject_id: int) -> Path:
    return root / f"mHealth_subject{int(subject_id)}.log"


def validate_mhealth_root(root: Path, subjects):
    if not root.exists():
        raise FileNotFoundError(f"MHEALTH root not found: {root}. Expected mHealth_subject<SUBJECT_ID>.log files.")
    missing = [str(_mhealth_subject_path(root, sid)) for sid in subjects if not _mhealth_subject_path(root, sid).exists()]
    if missing:
        preview = "\n".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f"\n... and {len(missing) - 20} more"
        raise FileNotFoundError(f"MHEALTH dataset is incomplete under {root}. Missing:\n{preview}{suffix}")


def _mhealth_read_subject_file(path: Path):
    data = np.loadtxt(path, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 24:
        raise ValueError(f"MHEALTH file {path} must have shape [num_samples, 24], got {data.shape}.")
    return data[:, :23], data[:, 23].astype(np.int64)


def _mhealth_window_subject(features, labels, window_size, stride, drop_null, min_label_purity, modality_columns):
    modality_windows = {name: [] for name in modality_columns}
    y_windows = []
    total = int(labels.shape[0])
    for start in range(0, total - window_size + 1, stride):
        end = start + window_size
        y = labels[start:end]
        values, counts = np.unique(y, return_counts=True)
        majority = int(values[int(np.argmax(counts))])
        purity = float(counts.max() / window_size)
        if purity < min_label_purity:
            continue
        if drop_null and majority == 0:
            continue
        for name, columns in modality_columns.items():
            # Return [channels, time] tensors to reuse the CNN-GRU encoder.
            modality_windows[name].append(features[start:end, columns].T)
        y_windows.append(majority - 1 if drop_null else majority)

    if not y_windows:
        raise RuntimeError(
            "No MHEALTH windows were produced. Try decreasing dataset.window_size, "
            "dataset.stride, or dataset.min_label_purity."
        )
    modalities = [np.stack(modality_windows[name]).astype(np.float32) for name in modality_columns]
    return modalities, np.asarray(y_windows, dtype=np.int64)


def _mhealth_build_split(root: Path, subjects, dataset_cfg):
    window_size = int(dataset_cfg.get("window_size", 128))
    stride = int(dataset_cfg.get("stride", 64))
    drop_null = bool(dataset_cfg.get("drop_null", True))
    min_label_purity = float(dataset_cfg.get("min_label_purity", 0.6))
    modality_columns = _mhealth_resolve_modalities(dataset_cfg)

    split_modalities = [[] for _ in modality_columns]
    split_labels = []
    for subject_id in subjects:
        features, labels = _mhealth_read_subject_file(_mhealth_subject_path(root, int(subject_id)))
        modalities, y = _mhealth_window_subject(
            features,
            labels,
            window_size,
            stride,
            drop_null,
            min_label_purity,
            modality_columns,
        )
        for idx, x in enumerate(modalities):
            split_modalities[idx].append(x)
        split_labels.append(y)

    x_all = [torch.tensor(np.concatenate(parts, axis=0), dtype=torch.float32) for parts in split_modalities]
    y_all = torch.tensor(np.concatenate(split_labels, axis=0), dtype=torch.long)
    return {"modalities": x_all, "labels": y_all}


def _mhealth_normalize(train, test):
    norm_train = []
    norm_test = []
    for x_train, x_test in zip(train["modalities"], test["modalities"]):
        mean = x_train.mean(dim=(0, 2), keepdim=True)
        std = x_train.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
        norm_train.append((x_train - mean) / std)
        norm_test.append((x_test - mean) / std)
    return {"modalities": norm_train, "labels": train["labels"]}, {"modalities": norm_test, "labels": test["labels"]}


def _mhealth_input_shapes(modalities):
    return [[int(v) for v in x.shape[1:]] for x in modalities]


def load_mhealth_dataset(cfg, project_root: Path):
    dataset_cfg = cfg.get("dataset", {})
    root = _mhealth_resolve_project_path(project_root, dataset_cfg.get("root", "./local/datasets/mhealth"))
    train_subjects = list(dataset_cfg.get("train_subjects", [1, 2, 3, 4, 5, 6, 7, 8]))
    test_subjects = list(dataset_cfg.get("test_subjects", [9, 10]))
    all_subjects = sorted(set(int(s) for s in train_subjects + test_subjects))

    validate_mhealth_root(root, all_subjects)
    train = _mhealth_build_split(root, train_subjects, dataset_cfg)
    test = _mhealth_build_split(root, test_subjects, dataset_cfg)
    if bool(dataset_cfg.get("normalize", True)):
        train, test = _mhealth_normalize(train, test)

    modality_names = list(_mhealth_resolve_modalities(dataset_cfg).keys())
    return {
        "train": train,
        "test": test,
        "root": str(root),
        "modality_names": modality_names,
        "modality_mhealth_input_shapes": _mhealth_input_shapes(train["modalities"]),
    }


PAMAP2_ACTIVITY_IDS = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]

PAMAP2_POSITION_MODALITIES = {
    "heart_rate": [2],
    "hand_imu": [3, 4, 5, 6, 10, 11, 12, 13, 14, 15],
    "chest_imu": [20, 21, 22, 23, 27, 28, 29, 30, 31, 32],
    "ankle_imu": [37, 38, 39, 40, 44, 45, 46, 47, 48, 49],
}

# Columns are zero-based. The readme marks orientation as invalid and
# recommends the 16g accelerometer over the 6g accelerometer.
PAMAP2_SENSOR_TYPE_MODALITIES = {
    "heart_rate": [2],
    "accelerometer": [4, 5, 6, 21, 22, 23, 38, 39, 40],
    "gyroscope": [10, 11, 12, 27, 28, 29, 44, 45, 46],
    "magnetometer": [13, 14, 15, 30, 31, 32, 47, 48, 49],
}


def _pamap2_modality_columns(dataset_cfg):
    scheme = str(dataset_cfg.get("modality_scheme", "sensor_type")).lower()
    if scheme in {"sensor_type", "sensor", "type"}:
        modalities = dict(PAMAP2_SENSOR_TYPE_MODALITIES)
        if not bool(dataset_cfg.get("include_heart_rate", False)):
            modalities.pop("heart_rate", None)
        return modalities
    if scheme in {"position", "device_position", "body_position"}:
        modalities = dict(PAMAP2_POSITION_MODALITIES)
        if not bool(dataset_cfg.get("include_heart_rate", True)):
            modalities.pop("heart_rate", None)
        return modalities
    raise ValueError("dataset.modality_scheme must be 'sensor_type' or 'position'.")


def _pamap2_resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _pamap2_protocol_dir(root: Path) -> Path:
    nested = root / "PAMAP2_Dataset" / "PAMAP2_Dataset" / "Protocol"
    if nested.exists():
        return nested
    direct = root / "PAMAP2_Dataset" / "Protocol"
    if direct.exists():
        return direct
    if (root / "Protocol").exists():
        return root / "Protocol"
    raise FileNotFoundError(
        f"Cannot find PAMAP2 Protocol directory under {root}. Expected Protocol/subject10*.dat files."
    )


def _pamap2_subject_path(protocol_dir: Path, subject_id: int) -> Path:
    return protocol_dir / f"subject{int(subject_id)}.dat"


def validate_pamap2_root(root: Path, subjects):
    protocol_dir = _pamap2_protocol_dir(root)
    missing = [str(_pamap2_subject_path(protocol_dir, sid)) for sid in subjects if not _pamap2_subject_path(protocol_dir, sid).exists()]
    if missing:
        preview = "\n".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f"\n... and {len(missing) - 20} more"
        raise FileNotFoundError(f"PAMAP2 dataset is incomplete under {root}. Missing:\n{preview}{suffix}")
    return protocol_dir


def _pamap2_read_subject_file(path: Path):
    data = np.loadtxt(path, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 54:
        raise ValueError(f"PAMAP2 file {path} must have shape [num_samples, 54], got {data.shape}.")
    return data


def _pamap2_fill_nan_columns(x):
    x = np.asarray(x, dtype=np.float32)
    if not np.isnan(x).any():
        return x
    col_mean = np.nanmean(x, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean).astype(np.float32)
    rows, cols = np.where(np.isnan(x))
    x[rows, cols] = col_mean[cols]
    return x


def _pamap2_window_subject(data, window_size, stride, drop_other, min_label_purity, modality_columns):
    labels = data[:, 1].astype(np.int64)
    features = _pamap2_fill_nan_columns(data)

    modality_windows = {name: [] for name in modality_columns}
    y_windows = []
    total = int(labels.shape[0])
    valid_ids = set(PAMAP2_ACTIVITY_IDS)
    for start in range(0, total - window_size + 1, stride):
        end = start + window_size
        y = labels[start:end]
        values, counts = np.unique(y, return_counts=True)
        majority = int(values[int(np.argmax(counts))])
        purity = float(counts.max() / window_size)
        if purity < min_label_purity:
            continue
        if drop_other and majority == 0:
            continue
        if majority not in valid_ids:
            continue
        for name, columns in modality_columns.items():
            modality_windows[name].append(features[start:end, columns].T)
        y_windows.append(majority)

    if not y_windows:
        raise RuntimeError(
            "No PAMAP2 windows were produced. Try decreasing dataset.window_size, "
            "dataset.stride, or dataset.min_label_purity."
        )
    modalities = [np.stack(modality_windows[name]).astype(np.float32) for name in modality_columns]
    return modalities, np.asarray(y_windows, dtype=np.int64)


def _pamap2_build_split(protocol_dir: Path, subjects, dataset_cfg):
    window_size = int(dataset_cfg.get("window_size", 128))
    stride = int(dataset_cfg.get("stride", 128))
    drop_other = bool(dataset_cfg.get("drop_other", True))
    min_label_purity = float(dataset_cfg.get("min_label_purity", 0.6))
    modality_columns = _pamap2_modality_columns(dataset_cfg)

    split_modalities = [[] for _ in modality_columns]
    split_labels = []
    for subject_id in subjects:
        data = _pamap2_read_subject_file(_pamap2_subject_path(protocol_dir, int(subject_id)))
        modalities, y = _pamap2_window_subject(data, window_size, stride, drop_other, min_label_purity, modality_columns)
        for idx, x in enumerate(modalities):
            split_modalities[idx].append(x)
        split_labels.append(y)

    x_all = [torch.tensor(np.concatenate(parts, axis=0), dtype=torch.float32) for parts in split_modalities]
    y_all = torch.tensor(np.concatenate(split_labels, axis=0), dtype=torch.long)
    return {"modalities": x_all, "labels": y_all}


def _pamap2_normalize(train, test):
    norm_train = []
    norm_test = []
    for x_train, x_test in zip(train["modalities"], test["modalities"]):
        mean = x_train.mean(dim=(0, 2), keepdim=True)
        std = x_train.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
        norm_train.append((x_train - mean) / std)
        norm_test.append((x_test - mean) / std)
    return {"modalities": norm_train, "labels": train["labels"]}, {"modalities": norm_test, "labels": test["labels"]}


def _pamap2_remap_labels(train, test):
    observed = sorted(set(train["labels"].tolist()) | set(test["labels"].tolist()))
    mapping = {label: idx for idx, label in enumerate(observed)}
    train_y = torch.tensor([mapping[int(v)] for v in train["labels"].tolist()], dtype=torch.long)
    test_y = torch.tensor([mapping[int(v)] for v in test["labels"].tolist()], dtype=torch.long)
    return {**train, "labels": train_y}, {**test, "labels": test_y}, mapping


def _pamap2_input_shapes(modalities):
    return [[int(v) for v in x.shape[1:]] for x in modalities]


def load_pamap2_dataset(cfg, project_root: Path):
    dataset_cfg = cfg.get("dataset", {})
    root = _pamap2_resolve_project_path(project_root, dataset_cfg.get("root", "./local/datasets/pamap2"))
    train_subjects = list(dataset_cfg.get("train_subjects", [101, 102, 103, 104, 105, 106, 107]))
    test_subjects = list(dataset_cfg.get("test_subjects", [108, 109]))
    all_subjects = sorted(set(int(s) for s in train_subjects + test_subjects))

    protocol_dir = validate_pamap2_root(root, all_subjects)
    train = _pamap2_build_split(protocol_dir, train_subjects, dataset_cfg)
    test = _pamap2_build_split(protocol_dir, test_subjects, dataset_cfg)
    train, test, label_mapping = _pamap2_remap_labels(train, test)
    if bool(dataset_cfg.get("normalize", True)):
        train, test = _pamap2_normalize(train, test)

    modality_names = list(_pamap2_modality_columns(dataset_cfg).keys())
    return {
        "train": train,
        "test": test,
        "root": str(root),
        "modality_names": modality_names,
        "modality_pamap2_input_shapes": _pamap2_input_shapes(train["modalities"]),
        "label_mapping": {str(k): int(v) for k, v in label_mapping.items()},
    }
