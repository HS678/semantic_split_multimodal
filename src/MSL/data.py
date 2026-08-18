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

# PAMAP2 8-fold LOSO：subject 109 is excluded because it has insufficient
# coverage of the selected 12-activity protocol.
PAMAP2_ALL_SUBJECTS = [101, 102, 103, 104, 105, 106, 107, 108, 109]
PAMAP2_EVALUATION_SUBJECTS = [101, 102, 103, 104, 105, 106, 107, 108]


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
        "subject_ids": split["subjects"][mask].contiguous(),
        "sample_ids": [f"official_{split_name}_{index}" for index, keep in enumerate(mask.tolist()) if keep],
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
            "subject_ids": torch.zeros((0,), dtype=torch.long),
            "sample_ids": [],
        }

    split_modalities = [[] for _ in modality_columns]
    split_labels = []
    split_subject_ids = []
    split_sample_ids = []
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
        split_subject_ids.extend([int(subject_id)] * int(len(y)))
        split_sample_ids.extend(
            f"subject{int(subject_id)}:window{window_index}"
            for window_index in range(int(len(y)))
        )

    x_all = [torch.tensor(np.concatenate(parts, axis=0), dtype=torch.float32) for parts in split_modalities]
    y_all = torch.tensor(np.concatenate(split_labels, axis=0), dtype=torch.long)
    return {
        "modalities": x_all,
        "labels": y_all,
        "subject_ids": torch.tensor(split_subject_ids, dtype=torch.long),
        "sample_ids": split_sample_ids,
    }


def _normalize_from_train(train, *evaluation_splits):
    normalized = [[] for _ in range(len(evaluation_splits) + 1)]
    for modality_idx, x_train in enumerate(train["modalities"]):
        mean = x_train.mean(dim=(0, 2), keepdim=True)
        std = x_train.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
        normalized[0].append((x_train - mean) / std)
        for split_idx, split in enumerate(evaluation_splits, start=1):
            normalized[split_idx].append((split["modalities"][modality_idx] - mean) / std)
    source_splits = (train, *evaluation_splits)
    out = []
    for modalities, source in zip(normalized, source_splits):
        preserved = {key: value for key, value in source.items() if key not in {"modalities", "labels"}}
        out.append({"modalities": modalities, "labels": source["labels"], **preserved})
    return tuple(out)


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
            "subject_ids": torch.zeros((0,), dtype=torch.long),
            "sample_ids": [],
        }

    split_modalities = [[] for _ in modality_columns]
    split_labels = []
    split_subject_ids = []
    split_sample_ids = []
    for subject_id in subjects:
        modalities, y = _pamap2_load_or_build_subject_windows(protocol_dir, int(subject_id), modality_columns, params, cache_root)
        for idx, x in enumerate(modalities):
            split_modalities[idx].append(x)
        split_labels.append(y)
        split_subject_ids.extend([int(subject_id)] * int(y.shape[0]))
        split_sample_ids.extend(
            f"subject{int(subject_id)}:window{window_index}"
            for window_index in range(int(y.shape[0]))
        )

    x_all = [torch.cat(parts, dim=0).to(dtype=torch.float32) for parts in split_modalities]
    y_all = torch.cat(split_labels, dim=0).to(dtype=torch.long)
    return {
        "modalities": x_all,
        "labels": y_all,
        "subject_ids": torch.tensor(split_subject_ids, dtype=torch.long),
        "sample_ids": split_sample_ids,
    }


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
    if fold < 1 or fold > len(PAMAP2_EVALUATION_SUBJECTS):
        raise ValueError(
            f"PAMAP2 8-fold LOSO uses folds 1-{len(PAMAP2_EVALUATION_SUBJECTS)} "
            "after excluding subject 109."
        )
    test_subjects = [PAMAP2_EVALUATION_SUBJECTS[fold - 1]]
    train_subjects = [s for s in PAMAP2_EVALUATION_SUBJECTS if s != test_subjects[0]]
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


import csv
import json
from pathlib import Path

import torch

from MSL.utils import partition_signature


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


from pathlib import Path
from typing import Callable

import torch

from MSL.iemocap import load_iemocap_dataset


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


register_dataset_loader("uci_har", load_uci_har_dataset)
register_dataset_loader("mhealth", load_mhealth_dataset)
register_dataset_loader("pamap2", load_pamap2_dataset)
register_dataset_loader("iemocap", load_iemocap_dataset)
