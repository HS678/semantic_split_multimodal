import os
from pathlib import Path

import numpy as np
import torch

from MSL.datasets.mhealth import _normalize_from_train


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
    import re
    match = re.search(r"fold(\d+)", str(split_protocol or ""))
    if not match:
        raise ValueError(f"split_protocol must contain fold<N>: {split_protocol!r}")
    return int(match.group(1))


PAMAP2_ACTIVITY_IDS = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]

# PAMAP2 8-fold LOSO：subject 109 is excluded because it has insufficient
# coverage of the selected 12-activity protocol.
PAMAP2_ALL_SUBJECTS = [101, 102, 103, 104, 105, 106, 107, 108, 109]
PAMAP2_EVALUATION_SUBJECTS = [101, 102, 103, 104, 105, 106, 107, 108]

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
