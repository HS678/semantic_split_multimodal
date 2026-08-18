from pathlib import Path

import numpy as np
import torch


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


MHEALTH_SENSOR_TYPE_MODALITIES = {
    "acc": [0, 1, 2, 5, 6, 7, 14, 15, 16],
    "gyro": [8, 9, 10, 17, 18, 19],
    "mag": [11, 12, 13, 20, 21, 22],
    "ecg": [3, 4],
}

# MHEALTH 5 折：每折 train 8 人 / test 2 人（与正式方案一致）。
MHEALTH_FOLD_SUBJECTS = {
    1: ([2, 3, 4, 5, 6, 7, 8, 9], [1, 10]),
    2: ([1, 2, 3, 4, 5, 7, 8, 10], [6, 9]),
    3: ([1, 3, 4, 5, 6, 8, 9, 10], [2, 7]),
    4: ([1, 2, 3, 5, 6, 7, 9, 10], [4, 8]),
    5: ([1, 2, 4, 6, 7, 8, 9, 10], [3, 5]),
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
