from pathlib import Path

import numpy as np
import torch


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


def _modality_columns(dataset_cfg):
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


def _resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _protocol_dir(root: Path) -> Path:
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


def _subject_path(protocol_dir: Path, subject_id: int) -> Path:
    return protocol_dir / f"subject{int(subject_id)}.dat"


def validate_pamap2_root(root: Path, subjects):
    protocol_dir = _protocol_dir(root)
    missing = [str(_subject_path(protocol_dir, sid)) for sid in subjects if not _subject_path(protocol_dir, sid).exists()]
    if missing:
        preview = "\n".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f"\n... and {len(missing) - 20} more"
        raise FileNotFoundError(f"PAMAP2 dataset is incomplete under {root}. Missing:\n{preview}{suffix}")
    return protocol_dir


def _read_subject_file(path: Path):
    data = np.loadtxt(path, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 54:
        raise ValueError(f"PAMAP2 file {path} must have shape [num_samples, 54], got {data.shape}.")
    return data


def _fill_nan_columns(x):
    x = np.asarray(x, dtype=np.float32)
    if not np.isnan(x).any():
        return x
    col_mean = np.nanmean(x, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean).astype(np.float32)
    rows, cols = np.where(np.isnan(x))
    x[rows, cols] = col_mean[cols]
    return x


def _window_subject(data, window_size, stride, drop_other, min_label_purity, modality_columns):
    labels = data[:, 1].astype(np.int64)
    features = _fill_nan_columns(data)

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


def _build_split(protocol_dir: Path, subjects, dataset_cfg):
    window_size = int(dataset_cfg.get("window_size", 128))
    stride = int(dataset_cfg.get("stride", 128))
    drop_other = bool(dataset_cfg.get("drop_other", True))
    min_label_purity = float(dataset_cfg.get("min_label_purity", 0.6))
    modality_columns = _modality_columns(dataset_cfg)

    split_modalities = [[] for _ in modality_columns]
    split_labels = []
    for subject_id in subjects:
        data = _read_subject_file(_subject_path(protocol_dir, int(subject_id)))
        modalities, y = _window_subject(data, window_size, stride, drop_other, min_label_purity, modality_columns)
        for idx, x in enumerate(modalities):
            split_modalities[idx].append(x)
        split_labels.append(y)

    x_all = [torch.tensor(np.concatenate(parts, axis=0), dtype=torch.float32) for parts in split_modalities]
    y_all = torch.tensor(np.concatenate(split_labels, axis=0), dtype=torch.long)
    return {"modalities": x_all, "labels": y_all}


def _normalize(train, test):
    norm_train = []
    norm_test = []
    for x_train, x_test in zip(train["modalities"], test["modalities"]):
        mean = x_train.mean(dim=(0, 2), keepdim=True)
        std = x_train.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
        norm_train.append((x_train - mean) / std)
        norm_test.append((x_test - mean) / std)
    return {"modalities": norm_train, "labels": train["labels"]}, {"modalities": norm_test, "labels": test["labels"]}


def _remap_labels(train, test):
    observed = sorted(set(train["labels"].tolist()) | set(test["labels"].tolist()))
    mapping = {label: idx for idx, label in enumerate(observed)}
    train_y = torch.tensor([mapping[int(v)] for v in train["labels"].tolist()], dtype=torch.long)
    test_y = torch.tensor([mapping[int(v)] for v in test["labels"].tolist()], dtype=torch.long)
    return {**train, "labels": train_y}, {**test, "labels": test_y}, mapping


def _input_shapes(modalities):
    return [[int(v) for v in x.shape[1:]] for x in modalities]


def load_pamap2_dataset(cfg, project_root: Path):
    dataset_cfg = cfg.get("dataset", {})
    root = _resolve_project_path(project_root, dataset_cfg.get("root", "./data/pamap2+physical+activity+monitoring"))
    train_subjects = list(dataset_cfg.get("train_subjects", [101, 102, 103, 104, 105, 106, 107]))
    test_subjects = list(dataset_cfg.get("test_subjects", [108, 109]))
    all_subjects = sorted(set(int(s) for s in train_subjects + test_subjects))

    protocol_dir = validate_pamap2_root(root, all_subjects)
    train = _build_split(protocol_dir, train_subjects, dataset_cfg)
    test = _build_split(protocol_dir, test_subjects, dataset_cfg)
    train, test, label_mapping = _remap_labels(train, test)
    if bool(dataset_cfg.get("normalize", True)):
        train, test = _normalize(train, test)

    modality_names = list(_modality_columns(dataset_cfg).keys())
    return {
        "train": train,
        "test": test,
        "root": str(root),
        "modality_names": modality_names,
        "modality_input_shapes": _input_shapes(train["modalities"]),
        "label_mapping": {str(k): int(v) for k, v in label_mapping.items()},
    }
