from pathlib import Path

import numpy as np
import torch


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


def _resolve_modalities(dataset_cfg):
    scheme = str(dataset_cfg.get("modality_scheme", "sensor_type")).lower()
    if scheme == "sensor_type":
        return MHEALTH_SENSOR_TYPE_MODALITIES
    if scheme == "position":
        return MHEALTH_POSITION_MODALITIES
    raise ValueError("dataset.modality_scheme must be 'sensor_type' or 'position'.")


def _resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _subject_path(root: Path, subject_id: int) -> Path:
    return root / f"mHealth_subject{int(subject_id)}.log"


def validate_mhealth_root(root: Path, subjects):
    if not root.exists():
        raise FileNotFoundError(f"MHEALTH root not found: {root}. Expected mHealth_subject<SUBJECT_ID>.log files.")
    missing = [str(_subject_path(root, sid)) for sid in subjects if not _subject_path(root, sid).exists()]
    if missing:
        preview = "\n".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f"\n... and {len(missing) - 20} more"
        raise FileNotFoundError(f"MHEALTH dataset is incomplete under {root}. Missing:\n{preview}{suffix}")


def _read_subject_file(path: Path):
    data = np.loadtxt(path, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 24:
        raise ValueError(f"MHEALTH file {path} must have shape [num_samples, 24], got {data.shape}.")
    return data[:, :23], data[:, 23].astype(np.int64)


def _window_subject(features, labels, window_size, stride, drop_null, min_label_purity, modality_columns):
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


def _build_split(root: Path, subjects, dataset_cfg):
    window_size = int(dataset_cfg.get("window_size", 128))
    stride = int(dataset_cfg.get("stride", 64))
    drop_null = bool(dataset_cfg.get("drop_null", True))
    min_label_purity = float(dataset_cfg.get("min_label_purity", 0.6))
    modality_columns = _resolve_modalities(dataset_cfg)

    split_modalities = [[] for _ in modality_columns]
    split_labels = []
    for subject_id in subjects:
        features, labels = _read_subject_file(_subject_path(root, int(subject_id)))
        modalities, y = _window_subject(
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


def _normalize(train, test):
    norm_train = []
    norm_test = []
    for x_train, x_test in zip(train["modalities"], test["modalities"]):
        mean = x_train.mean(dim=(0, 2), keepdim=True)
        std = x_train.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
        norm_train.append((x_train - mean) / std)
        norm_test.append((x_test - mean) / std)
    return {"modalities": norm_train, "labels": train["labels"]}, {"modalities": norm_test, "labels": test["labels"]}


def _input_dims(modalities):
    return [int(x.reshape(int(x.shape[0]), -1).shape[1]) for x in modalities]


def _input_shapes(modalities):
    return [[int(v) for v in x.shape[1:]] for x in modalities]


def load_mhealth_dataset(cfg, project_root: Path):
    dataset_cfg = cfg.get("dataset", {})
    root = _resolve_project_path(project_root, dataset_cfg.get("root", "./data/MHEALTHDATASET"))
    train_subjects = list(dataset_cfg.get("train_subjects", [1, 2, 3, 4, 5, 6, 7, 8]))
    test_subjects = list(dataset_cfg.get("test_subjects", [9, 10]))
    all_subjects = sorted(set(int(s) for s in train_subjects + test_subjects))

    validate_mhealth_root(root, all_subjects)
    train = _build_split(root, train_subjects, dataset_cfg)
    test = _build_split(root, test_subjects, dataset_cfg)
    if bool(dataset_cfg.get("normalize", True)):
        train, test = _normalize(train, test)

    modality_names = list(_resolve_modalities(dataset_cfg).keys())
    return {
        "train": train,
        "test": test,
        "root": str(root),
        "modality_names": modality_names,
        "modality_input_dims": _input_dims(train["modalities"]),
        "modality_input_shapes": _input_shapes(train["modalities"]),
    }
