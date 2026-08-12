import csv
import os
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
        subject_path = split_dir / f"subject_{split}.txt"
        if not subject_path.exists():
            missing.append(str(subject_path))
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
    subjects = np.loadtxt(split_dir / f"subject_{split}.txt", dtype=np.int64)
    acc = np.stack(
        [body_acc_x, body_acc_y, body_acc_z, total_acc_x, total_acc_y, total_acc_z],
        axis=1,
    ).astype(np.float32)
    gyro = np.stack([body_gyro_x, body_gyro_y, body_gyro_z], axis=1).astype(np.float32)

    return {
        "modalities": [torch.tensor(acc, dtype=torch.float32), torch.tensor(gyro, dtype=torch.float32)],
        "labels": torch.tensor(labels, dtype=torch.long),
        "subjects": torch.tensor(subjects, dtype=torch.long),
        "modality_input_shapes": [[int(acc.shape[1]), int(acc.shape[2])], [int(gyro.shape[1]), int(gyro.shape[2])]],
        "modality_names": ["acc", "gyro"],
    }


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


UCI_HAR_TRAIN_SUBJECTS = [1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]
UCI_HAR_TEST_SUBJECTS = [2, 4, 9, 10, 12, 13, 18, 20, 24]

# MHEALTH 5 折：每折 train 8 人 / test 2 人（与正式方案一致）。
MHEALTH_FOLD_SUBJECTS = {
    1: ([2, 3, 4, 5, 6, 7, 8, 9], [1, 10]),
    2: ([1, 2, 3, 4, 5, 7, 8, 10], [6, 9]),
    3: ([1, 3, 4, 5, 6, 8, 9, 10], [2, 7]),
    4: ([1, 2, 3, 5, 6, 7, 9, 10], [4, 8]),
    5: ([1, 2, 4, 6, 7, 8, 9, 10], [3, 5]),
}

# PAMAP2 9 折 LOSO：每折 test 1 人（101~109），train 为其余 8 人。
PAMAP2_ALL_SUBJECTS = [101, 102, 103, 104, 105, 106, 107, 108, 109]


def _select_subjects(split, subjects, split_name):
    requested = {int(v) for v in subjects}
    available = {int(v) for v in split["subjects"].tolist()}
    missing = sorted(requested - available)
    if missing:
        raise ValueError(f"UCI-HAR {split_name}_subjects are not present in the source split: {missing}.")
    mask = torch.tensor([int(v) in requested for v in split["subjects"].tolist()], dtype=torch.bool)
    return {
        "modalities": [x[mask].contiguous() for x in split["modalities"]],
        "labels": split["labels"][mask].contiguous(),
    }


def load_uci_har_dataset(cfg, project_root: Path):
    dataset_cfg = cfg.get("dataset", {})
    root_cfg = dataset_cfg.get("root", "./local/datasets/UCI-HAR")
    root = Path(root_cfg)
    if not root.is_absolute():
        root = project_root / root
    root = root.resolve()

    validate_uci_har_root(root)
    train_subjects = list(UCI_HAR_TRAIN_SUBJECTS)
    test_subjects = list(UCI_HAR_TEST_SUBJECTS)
    _validate_subject_splits(train_subjects, test_subjects, "UCI-HAR")

    official_train = _build_modality_vectors(root, "train")
    official_test = _build_modality_vectors(root, "test")
    train = _select_subjects(official_train, train_subjects, "train")
    test = _select_subjects(official_test, test_subjects, "test")
    return {
        "train": train,
        "test": test,
        "root": str(root),
        "split_subjects": {
            "train": list(UCI_HAR_TRAIN_SUBJECTS),
            "test": list(UCI_HAR_TEST_SUBJECTS),
        },
        "modality_input_shapes": official_train["modality_input_shapes"],
        "modality_names": official_train["modality_names"],
    }


MHEALTH_SENSOR_TYPE_MODALITIES = {
    "acc": [0, 1, 2, 5, 6, 7, 14, 15, 16],
    "gyro": [8, 9, 10, 17, 18, 19],
    "mag": [11, 12, 13, 20, 21, 22],
    "ecg": [3, 4],
}


def _mhealth_resolve_modalities(_dataset_cfg=None):
    # 固定按传感器类型划分模态（acc / gyro / mag / ecg）。
    return MHEALTH_SENSOR_TYPE_MODALITIES


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
    # 滑窗与预处理参数固定内置：window=128, stride=64, 丢弃空标签, 标签纯度>=0.6。
    window_size = 128
    stride = 64
    drop_null = True
    min_label_purity = 0.6
    modality_columns = _mhealth_resolve_modalities(dataset_cfg)

    if not subjects:
        return {
            "modalities": [torch.zeros((0, 1, 1), dtype=torch.float32) for _ in modality_columns],
            "labels": torch.zeros((0,), dtype=torch.long),
        }

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


def _normalize_from_train(train, *evaluation_splits):
    normalized = [[] for _ in range(len(evaluation_splits) + 1)]
    for modality_idx, x_train in enumerate(train["modalities"]):
        mean = x_train.mean(dim=(0, 2), keepdim=True)
        std = x_train.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
        normalized[0].append((x_train - mean) / std)
        for split_idx, split in enumerate(evaluation_splits, start=1):
            normalized[split_idx].append((split["modalities"][modality_idx] - mean) / std)
    source_splits = (train, *evaluation_splits)
    return tuple(
        {"modalities": modalities, "labels": source["labels"]}
        for modalities, source in zip(normalized, source_splits)
    )


def _mhealth_input_shapes(modalities):
    return [[int(v) for v in x.shape[1:]] for x in modalities]


def load_mhealth_dataset(cfg, project_root: Path):
    dataset_cfg = cfg.get("dataset", {})
    root = _mhealth_resolve_project_path(project_root, dataset_cfg.get("root", "./local/datasets/MHEALTH"))
    fold = _fold_number(dataset_cfg.get("split_protocol", ""))
    train_subjects, test_subjects = MHEALTH_FOLD_SUBJECTS[fold]
    _validate_subject_splits(train_subjects, test_subjects, "MHEALTH")
    all_subjects = sorted(set(int(s) for s in train_subjects + test_subjects))

    validate_mhealth_root(root, all_subjects)
    train = _mhealth_build_split(root, train_subjects, dataset_cfg)
    test = _mhealth_build_split(root, test_subjects, dataset_cfg)
    if bool(dataset_cfg.get("normalize", True)):
        train, test = _normalize_from_train(train, test)

    modality_names = list(_mhealth_resolve_modalities(dataset_cfg).keys())
    input_shapes = _mhealth_input_shapes(train["modalities"])
    return {
        "train": train,
        "test": test,
        "root": str(root),
        "split_subjects": {"train": list(train_subjects), "test": list(test_subjects)},
        "modality_names": modality_names,
        "modality_input_shapes": input_shapes,
        "modality_mhealth_input_shapes": input_shapes,
    }


PAMAP2_ACTIVITY_IDS = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]

# Columns are zero-based. The readme marks orientation as invalid and
# recommends the 16g accelerometer over the 6g accelerometer.
# 固定不含心率模态（heart_rate 列不使用）。
PAMAP2_SENSOR_TYPE_MODALITIES = {
    "acc": [4, 5, 6, 21, 22, 23, 38, 39, 40],
    "gyro": [10, 11, 12, 27, 28, 29, 44, 45, 46],
    "mag": [13, 14, 15, 30, 31, 32, 47, 48, 49],
}
PAMAP2_WINDOW_CACHE_VERSION = 1


def _pamap2_modality_columns(_dataset_cfg=None):
    # 固定按传感器类型划分模态（acc / gyro / mag），不含心率。
    return dict(PAMAP2_SENSOR_TYPE_MODALITIES)


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


def _pamap2_window_params():
    # 滑窗与预处理参数固定内置：window=200, stride=100, 丢弃"其他"标签, 标签纯度>=0.6。
    return {
        "window_size": 200,
        "stride": 100,
        "drop_other": True,
        "min_label_purity": 0.6,
    }


def _pamap2_cache_root(root: Path, dataset_cfg: dict) -> Path:
    cache_cfg = dataset_cfg.get("processed_root")
    if cache_cfg:
        cache_root = Path(str(cache_cfg))
        if not cache_root.is_absolute():
            cache_root = root / cache_root
    else:
        cache_root = root / "processed" / "window_cache"
    return cache_root / "sensor_type_w200_s100_purity0p6_dropother"


def _pamap2_expected_cache_metadata(source_path: Path, modality_columns: dict, params: dict) -> dict:
    stat = source_path.stat()
    return {
        "version": PAMAP2_WINDOW_CACHE_VERSION,
        "source_path": str(source_path.resolve()),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "source_size": int(stat.st_size),
        "activity_ids": list(PAMAP2_ACTIVITY_IDS),
        "modality_columns": {name: list(columns) for name, columns in modality_columns.items()},
        "window": dict(params),
    }


def _pamap2_cache_path(cache_root: Path, subject_id: int) -> Path:
    return cache_root / f"subject{int(subject_id)}.pt"


def _pamap2_payload_matches_cache(payload: dict, expected_metadata: dict, modality_columns: dict) -> bool:
    if not isinstance(payload, dict) or payload.get("metadata") != expected_metadata:
        return False
    modalities = payload.get("modalities")
    labels = payload.get("labels")
    if not isinstance(modalities, list) or len(modalities) != len(modality_columns):
        return False
    if not torch.is_tensor(labels) or labels.dtype != torch.long:
        return False
    return all(torch.is_tensor(x) and x.dtype == torch.float32 for x in modalities)


def _pamap2_load_or_build_subject_windows(protocol_dir: Path, subject_id: int, modality_columns: dict, params: dict, cache_root: Path):
    source_path = _pamap2_subject_path(protocol_dir, int(subject_id))
    expected_metadata = _pamap2_expected_cache_metadata(source_path, modality_columns, params)
    cache_path = _pamap2_cache_path(cache_root, int(subject_id))
    if cache_path.exists():
        try:
            payload = torch.load(cache_path, map_location="cpu")
            if _pamap2_payload_matches_cache(payload, expected_metadata, modality_columns):
                return payload["modalities"], payload["labels"]
        except Exception:
            pass

    data = _pamap2_read_subject_file(source_path)
    modalities_np, labels_np = _pamap2_window_subject(
        data,
        int(params["window_size"]),
        int(params["stride"]),
        bool(params["drop_other"]),
        float(params["min_label_purity"]),
        modality_columns,
    )
    modalities = [torch.tensor(x, dtype=torch.float32) for x in modalities_np]
    labels = torch.tensor(labels_np, dtype=torch.long)
    payload = {
        "metadata": expected_metadata,
        "modalities": modalities,
        "labels": labels,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
    torch.save(payload, temp_path)
    temp_path.replace(cache_path)
    return modalities, labels


def _pamap2_build_split(protocol_dir: Path, subjects, dataset_cfg, cache_root: Path):
    params = _pamap2_window_params()
    modality_columns = _pamap2_modality_columns(dataset_cfg)

    if not subjects:
        return {
            "modalities": [torch.zeros((0, 1, 1), dtype=torch.float32) for _ in modality_columns],
            "labels": torch.zeros((0,), dtype=torch.long),
        }

    split_modalities = [[] for _ in modality_columns]
    split_labels = []
    for subject_id in subjects:
        modalities, y = _pamap2_load_or_build_subject_windows(protocol_dir, int(subject_id), modality_columns, params, cache_root)
        for idx, x in enumerate(modalities):
            split_modalities[idx].append(x)
        split_labels.append(y)

    x_all = [torch.cat(parts, dim=0).to(dtype=torch.float32) for parts in split_modalities]
    y_all = torch.cat(split_labels, dim=0).to(dtype=torch.long)
    return {"modalities": x_all, "labels": y_all}


def _pamap2_remap_labels(*splits):
    mapping = {label: idx for idx, label in enumerate(PAMAP2_ACTIVITY_IDS)}
    remapped = []
    for split in splits:
        unexpected = sorted(set(int(v) for v in split["labels"].tolist()) - set(mapping))
        if unexpected:
            raise ValueError(f"PAMAP2 split contains unsupported activity IDs: {unexpected}.")
        labels = torch.tensor([mapping[int(v)] for v in split["labels"].tolist()], dtype=torch.long)
        remapped.append({**split, "labels": labels})
    return (*remapped, mapping)


def _pamap2_input_shapes(modalities):
    return [[int(v) for v in x.shape[1:]] for x in modalities]


def load_pamap2_dataset(cfg, project_root: Path):
    dataset_cfg = cfg.get("dataset", {})
    root = _pamap2_resolve_project_path(project_root, dataset_cfg.get("root", "./local/datasets/PAMAP2"))
    fold = _fold_number(dataset_cfg.get("split_protocol", ""))
    test_subjects = [PAMAP2_ALL_SUBJECTS[fold - 1]]
    train_subjects = [s for s in PAMAP2_ALL_SUBJECTS if s != test_subjects[0]]
    _validate_subject_splits(train_subjects, test_subjects, "PAMAP2")
    all_subjects = sorted(set(int(s) for s in train_subjects + test_subjects))

    protocol_dir = validate_pamap2_root(root, all_subjects)
    cache_root = _pamap2_cache_root(root, dataset_cfg)
    train = _pamap2_build_split(protocol_dir, train_subjects, dataset_cfg, cache_root)
    test = _pamap2_build_split(protocol_dir, test_subjects, dataset_cfg, cache_root)
    train, test, label_mapping = _pamap2_remap_labels(train, test)
    if bool(dataset_cfg.get("normalize", True)):
        train, test = _normalize_from_train(train, test)

    modality_names = list(_pamap2_modality_columns(dataset_cfg).keys())
    input_shapes = _pamap2_input_shapes(train["modalities"])
    return {
        "train": train,
        "test": test,
        "root": str(root),
        "split_subjects": {"train": list(train_subjects), "test": list(test_subjects)},
        "modality_names": modality_names,
        "modality_input_shapes": input_shapes,
        "modality_pamap2_input_shapes": input_shapes,
        "label_mapping": {str(k): int(v) for k, v in label_mapping.items()},
    }
