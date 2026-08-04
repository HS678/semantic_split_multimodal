import csv
import io
import pickle
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


def _validate_subject_splits(train_subjects, validation_subjects, test_subjects, dataset_name):
    splits = {
        "train": {int(v) for v in train_subjects},
        "validation": {int(v) for v in validation_subjects},
        "test": {int(v) for v in test_subjects},
    }
    for split_name, subjects in splits.items():
        if not subjects:
            raise ValueError(f"{dataset_name} {split_name}_subjects must not be empty.")
    overlaps = {
        "train_validation": sorted(splits["train"] & splits["validation"]),
        "train_test": sorted(splits["train"] & splits["test"]),
        "validation_test": sorted(splits["validation"] & splits["test"]),
    }
    non_empty = {name: values for name, values in overlaps.items() if values}
    if non_empty:
        raise ValueError(f"{dataset_name} subject splits must be disjoint, overlaps={non_empty}.")
    return splits


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
    root_cfg = dataset_cfg.get("root", "./local/datasets/uci_har")
    root = Path(root_cfg)
    if not root.is_absolute():
        root = project_root / root
    root = root.resolve()

    validate_uci_har_root(root)
    train_subjects = list(dataset_cfg.get("train_subjects", [1, 3, 5, 6, 7, 8, 11, 15, 16, 17, 21, 22, 26, 27, 28, 29, 30]))
    validation_subjects = list(dataset_cfg.get("validation_subjects", [14, 19, 23, 25]))
    test_subjects = list(dataset_cfg.get("test_subjects", [2, 4, 9, 10, 12, 13, 18, 20, 24]))
    _validate_subject_splits(train_subjects, validation_subjects, test_subjects, "UCI-HAR")

    official_train = _build_modality_vectors(root, "train")
    official_test = _build_modality_vectors(root, "test")
    train = _select_subjects(official_train, train_subjects, "train")
    validation = _select_subjects(official_train, validation_subjects, "validation")
    test = _select_subjects(official_test, test_subjects, "test")
    return {
        "train": train,
        "validation": validation,
        "test": test,
        "root": str(root),
        "modality_input_shapes": official_train["modality_input_shapes"],
        "modality_names": official_train["modality_names"],
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
    root = _mhealth_resolve_project_path(project_root, dataset_cfg.get("root", "./local/datasets/mhealth"))
    train_subjects = list(dataset_cfg.get("train_subjects", [2, 3, 4, 6, 7, 8]))
    validation_subjects = list(dataset_cfg.get("validation_subjects", [1, 5]))
    test_subjects = list(dataset_cfg.get("test_subjects", [9, 10]))
    _validate_subject_splits(train_subjects, validation_subjects, test_subjects, "MHEALTH")
    all_subjects = sorted(set(int(s) for s in train_subjects + validation_subjects + test_subjects))

    validate_mhealth_root(root, all_subjects)
    train = _mhealth_build_split(root, train_subjects, dataset_cfg)
    validation = _mhealth_build_split(root, validation_subjects, dataset_cfg)
    test = _mhealth_build_split(root, test_subjects, dataset_cfg)
    if bool(dataset_cfg.get("normalize", True)):
        train, validation, test = _normalize_from_train(train, validation, test)

    modality_names = list(_mhealth_resolve_modalities(dataset_cfg).keys())
    input_shapes = _mhealth_input_shapes(train["modalities"])
    return {
        "train": train,
        "validation": validation,
        "test": test,
        "root": str(root),
        "modality_names": modality_names,
        "modality_input_shapes": input_shapes,
        "modality_mhealth_input_shapes": input_shapes,
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
    root = _pamap2_resolve_project_path(project_root, dataset_cfg.get("root", "./local/datasets/pamap2"))
    train_subjects = list(dataset_cfg.get("train_subjects", [101, 102, 103, 104, 105, 106]))
    validation_subjects = list(dataset_cfg.get("validation_subjects", [107, 108]))
    test_subjects = list(dataset_cfg.get("test_subjects", [109]))
    _validate_subject_splits(train_subjects, validation_subjects, test_subjects, "PAMAP2")
    all_subjects = sorted(set(int(s) for s in train_subjects + validation_subjects + test_subjects))

    protocol_dir = validate_pamap2_root(root, all_subjects)
    train = _pamap2_build_split(protocol_dir, train_subjects, dataset_cfg)
    validation = _pamap2_build_split(protocol_dir, validation_subjects, dataset_cfg)
    test = _pamap2_build_split(protocol_dir, test_subjects, dataset_cfg)
    train, validation, test, label_mapping = _pamap2_remap_labels(train, validation, test)
    if bool(dataset_cfg.get("normalize", True)):
        train, validation, test = _normalize_from_train(train, validation, test)

    modality_names = list(_pamap2_modality_columns(dataset_cfg).keys())
    input_shapes = _pamap2_input_shapes(train["modalities"])
    return {
        "train": train,
        "validation": validation,
        "test": test,
        "root": str(root),
        "modality_names": modality_names,
        "modality_input_shapes": input_shapes,
        "modality_pamap2_input_shapes": input_shapes,
        "label_mapping": {str(k): int(v) for k, v in label_mapping.items()},
    }


class _CMUMOSEIComputationalSequence:
    """Minimal pickle target for SDK computational-sequence feature files."""


class _CMUMOSEIUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (
            module == "mmsdk.mmdatasdk.computational_sequence.computational_sequence"
            and name == "computational_sequence"
        ):
            return _CMUMOSEIComputationalSequence
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda payload: torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def _cmu_mosei_resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _cmu_mosei_required_paths(root: Path, dataset_cfg):
    feature_dir = _cmu_mosei_resolve_path(root, dataset_cfg.get("features_dir", "features"))
    split_dir = _cmu_mosei_resolve_path(root, dataset_cfg.get("splits_dir", "splits"))
    return {
        "text": feature_dir / dataset_cfg.get("text_file", "BERT_MOSEI.pkl"),
        "audio": feature_dir / dataset_cfg.get("audio_file", "COAVAREP_aligned_MOSEI.pkl"),
        "visual": feature_dir / dataset_cfg.get("visual_file", "FACET_aligned_MOSEI.pkl"),
        "train": split_dir / dataset_cfg.get("train_split_file", "df_MOSEI.tsv"),
        "validation": split_dir / dataset_cfg.get("validation_split_file", "df_valid_MOSEI.tsv"),
        "test": split_dir / dataset_cfg.get("test_split_file", "df_test_MOSEI.tsv"),
    }


def validate_cmu_mosei_root(root: Path, dataset_cfg):
    if not root.exists():
        raise FileNotFoundError(f"CMU-MOSEI root not found: {root}.")
    paths = _cmu_mosei_required_paths(root, dataset_cfg)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        preview = "\n".join(missing)
        raise FileNotFoundError(f"CMU-MOSEI dataset is incomplete under {root}. Missing:\n{preview}")
    return paths


def _cmu_mosei_read_split(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"CMU-MOSEI split file is empty: {path}")

    sample_ids = []
    polarities = []
    for row_idx, row in enumerate(rows):
        sample_id = str(row.get("", "")).strip()
        if not sample_id:
            raise ValueError(f"CMU-MOSEI split file {path} has an empty sample id at row {row_idx + 2}.")
        try:
            polarity = float(row["polarity"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"CMU-MOSEI split file {path} has an invalid polarity at row {row_idx + 2}."
            ) from exc
        sample_ids.append(sample_id)
        polarities.append(polarity)
    return sample_ids, torch.tensor(polarities, dtype=torch.float32)


def _cmu_mosei_load_pickle(path: Path):
    with path.open("rb") as f:
        return _CMUMOSEIUnpickler(f).load()


def _cmu_mosei_pool_sequence(sequence, sample_id: str, modality_name: str):
    features = np.asarray(sequence, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(
            f"CMU-MOSEI {modality_name} sample {sample_id} must have non-empty [time, feature] data, "
            f"got {features.shape}."
        )
    finite = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return finite.mean(axis=0, dtype=np.float32)


def _cmu_mosei_standardize_from_train(train, validation, test):
    normalized = {"train": [], "validation": [], "test": []}
    for modality_idx, train_tensor in enumerate(train["modalities"]):
        mean = train_tensor.mean(dim=0, keepdim=True)
        std = train_tensor.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        normalized["train"].append((train_tensor - mean) / std)
        normalized["validation"].append((validation["modalities"][modality_idx] - mean) / std)
        normalized["test"].append((test["modalities"][modality_idx] - mean) / std)
    return (
        {"modalities": normalized["train"], "labels": train["labels"]},
        {"modalities": normalized["validation"], "labels": validation["labels"]},
        {"modalities": normalized["test"], "labels": test["labels"]},
    )


def _validate_cmu_mosei_protocol(dataset_cfg, split_rows):
    expected = {
        "task": "binary_sentiment",
        "label_protocol": "negative_vs_non_negative",
        "temporal_pooling": "mean",
    }
    for field, expected_value in expected.items():
        actual = str(dataset_cfg.get(field, expected_value)).strip().lower()
        if actual != expected_value:
            raise ValueError(
                f"CMU-MOSEI dataset.{field} must be '{expected_value}', got '{actual}'."
            )

    sample_sets = {}
    video_sets = {}
    for split_name, (sample_ids, _) in split_rows.items():
        sample_set = set(sample_ids)
        if len(sample_set) != len(sample_ids):
            raise ValueError(f"CMU-MOSEI {split_name} split contains duplicate sample IDs.")
        sample_sets[split_name] = sample_set
        video_sets[split_name] = {sample_id.rsplit("[", 1)[0] for sample_id in sample_ids}

    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        sample_overlap = sample_sets[left] & sample_sets[right]
        video_overlap = video_sets[left] & video_sets[right]
        if sample_overlap or video_overlap:
            raise ValueError(
                f"CMU-MOSEI official splits must be sample- and video-disjoint: "
                f"{left}/{right} sample_overlap={sorted(sample_overlap)[:10]}, "
                f"video_overlap={sorted(video_overlap)[:10]}."
            )


def load_cmu_mosei_dataset(cfg, project_root: Path):
    dataset_cfg = cfg.get("dataset", {})
    root_cfg = dataset_cfg.get("root", "./local/datasets/cmu_mosei")
    root = Path(root_cfg)
    if not root.is_absolute():
        root = project_root / root
    root = root.resolve()
    paths = validate_cmu_mosei_root(root, dataset_cfg)

    split_rows = {
        split_name: _cmu_mosei_read_split(paths[split_name])
        for split_name in ("train", "validation", "test")
    }
    _validate_cmu_mosei_protocol(dataset_cfg, split_rows)
    text_payload = _cmu_mosei_load_pickle(paths["text"])
    audio_payload = _cmu_mosei_load_pickle(paths["audio"])
    visual_payload = _cmu_mosei_load_pickle(paths["visual"])

    if not isinstance(text_payload, dict) or "Data" not in text_payload or "level" not in text_payload:
        raise ValueError("CMU-MOSEI text pickle must contain 'Data' and 'level'.")
    text_features = torch.as_tensor(text_payload["Data"], dtype=torch.float32).cpu()
    text_levels = torch.as_tensor(text_payload["level"], dtype=torch.float32).reshape(-1).cpu()
    expected_samples = sum(len(sample_ids) for sample_ids, _ in split_rows.values())
    if text_features.ndim != 2 or int(text_features.shape[0]) != expected_samples:
        raise ValueError(
            f"CMU-MOSEI text features must have shape [total_split_samples, feature_dim], "
            f"got {tuple(text_features.shape)} for {expected_samples} split samples."
        )
    if int(text_levels.shape[0]) != expected_samples:
        raise ValueError(
            f"CMU-MOSEI text labels contain {int(text_levels.shape[0])} samples, expected {expected_samples}."
        )

    audio_data = getattr(audio_payload, "data", None)
    visual_data = getattr(visual_payload, "data", None)
    if not isinstance(audio_data, dict) or not isinstance(visual_data, dict):
        raise ValueError("CMU-MOSEI audio/visual pickles must contain computational-sequence data dictionaries.")

    splits = {}
    offset = 0
    for split_name in ("train", "validation", "test"):
        sample_ids, polarity = split_rows[split_name]
        end = offset + len(sample_ids)
        stored_levels = text_levels[offset:end]
        if not torch.equal(stored_levels, polarity):
            max_diff = float((stored_levels - polarity).abs().max().item())
            raise ValueError(
                f"CMU-MOSEI {split_name} polarity does not match BERT_MOSEI.pkl level values; "
                f"max_abs_diff={max_diff}."
            )

        missing_audio = [sample_id for sample_id in sample_ids if sample_id not in audio_data]
        missing_visual = [sample_id for sample_id in sample_ids if sample_id not in visual_data]
        if missing_audio or missing_visual:
            raise ValueError(
                f"CMU-MOSEI {split_name} has missing modalities: "
                f"audio={missing_audio[:10]}, visual={missing_visual[:10]}."
            )

        audio = np.stack(
            [
                _cmu_mosei_pool_sequence(audio_data[sample_id]["features"], sample_id, "audio")
                for sample_id in sample_ids
            ]
        )
        visual = np.stack(
            [
                _cmu_mosei_pool_sequence(visual_data[sample_id]["features"], sample_id, "visual")
                for sample_id in sample_ids
            ]
        )
        labels = (polarity >= 0).to(dtype=torch.long)
        splits[split_name] = {
            "modalities": [
                text_features[offset:end].contiguous(),
                torch.from_numpy(audio).to(dtype=torch.float32),
                torch.from_numpy(visual).to(dtype=torch.float32),
            ],
            "labels": labels,
        }
        offset = end

    if bool(dataset_cfg.get("normalize", True)):
        splits["train"], splits["validation"], splits["test"] = _cmu_mosei_standardize_from_train(
            splits["train"],
            splits["validation"],
            splits["test"],
        )

    modality_names = ["text", "audio", "visual"]
    input_shapes = [
        [int(tensor.shape[1])]
        for tensor in splits["train"]["modalities"]
    ]
    return {
        "train": splits["train"],
        "validation": splits["validation"],
        "test": splits["test"],
        "root": str(root),
        "modality_names": modality_names,
        "modality_input_shapes": input_shapes,
        "label_mapping": {"negative": 0, "non_negative": 1},
        "split_num_samples": {
            split_name: int(splits[split_name]["labels"].shape[0])
            for split_name in ("train", "validation", "test")
        },
    }
