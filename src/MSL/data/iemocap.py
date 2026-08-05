import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from MSL.data.datasets import _validate_subject_splits


IEMOCAP_MODALITY_NAMES = ["audio", "video", "text"]
IEMOCAP_ENCODER_TYPES = ["conv_gru", "gru", "gru"]
IEMOCAP_LABEL_MAPPING = {
    "ang": 0,
    "hap": 1,
    "exc": 1,
    "sad": 2,
    "neu": 3,
}
IEMOCAP_CLASS_NAMES = ["angry", "happy", "sad", "neutral"]


def _resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _validate_iemocap_protocol(dataset_cfg: dict) -> None:
    expected = {
        "variant": "full",
        "task": "emotion_4class",
        "label_protocol": "ang_hap_exc_sad_neu_v1",
        "feature_recipe": "mfcc_mobilevit_xs_distilbert_v1",
    }
    for field, expected_value in expected.items():
        actual = str(dataset_cfg.get(field, expected_value)).strip().lower()
        if actual != expected_value:
            raise ValueError(
                f"IEMOCAP dataset.{field} must be '{expected_value}', got '{actual}'."
            )


def _load_feature_payload(path: Path, modality_name: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing IEMOCAP {modality_name} feature cache: {path}. "
            "Run 'python -m MSL.data.prepare_iemocap' first."
        )
    payload = torch.load(path, map_location="cpu")
    required = {"sample_ids", "features", "lengths", "feature_extractor"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"IEMOCAP {modality_name} feature cache is missing keys: {missing}")
    features = payload["features"]
    lengths = payload["lengths"]
    if not torch.is_tensor(features) or features.ndim != 3:
        raise ValueError(
            f"IEMOCAP {modality_name} features must be a [samples, time, dim] tensor."
        )
    if not torch.is_tensor(lengths) or lengths.ndim != 1:
        raise ValueError(f"IEMOCAP {modality_name} lengths must be a 1D tensor.")
    if int(features.shape[0]) != len(payload["sample_ids"]) or int(lengths.shape[0]) != int(features.shape[0]):
        raise ValueError(f"IEMOCAP {modality_name} cache sample counts do not match.")
    if torch.any(lengths <= 0) or torch.any(lengths > int(features.shape[1])):
        raise ValueError(f"IEMOCAP {modality_name} cache contains invalid sequence lengths.")
    payload["features"] = features.to(dtype=torch.float32)
    payload["lengths"] = lengths.to(dtype=torch.long)
    return payload


def _load_manifest(processed_root: Path) -> list[dict]:
    path = processed_root / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing IEMOCAP manifest: {path}. "
            "Run 'python -m MSL.data.prepare_iemocap' first."
        )
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"IEMOCAP manifest must contain a non-empty 'samples' list: {path}")
    sample_ids = [str(row.get("utterance_id", "")) for row in rows]
    if any(not sample_id for sample_id in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("IEMOCAP manifest utterance IDs must be non-empty and unique.")
    return rows


def _standardize_valid_sequences(train, validation, test):
    outputs = {
        "train": {"modalities": [], "modality_lengths": train["modality_lengths"], "labels": train["labels"]},
        "validation": {
            "modalities": [],
            "modality_lengths": validation["modality_lengths"],
            "labels": validation["labels"],
        },
        "test": {"modalities": [], "modality_lengths": test["modality_lengths"], "labels": test["labels"]},
    }
    for modality_idx, x_train in enumerate(train["modalities"]):
        train_lengths = train["modality_lengths"][modality_idx]
        time = torch.arange(x_train.shape[1]).unsqueeze(0)
        train_mask = time < train_lengths.unsqueeze(1)
        valid_train = x_train[train_mask]
        mean = valid_train.mean(dim=0, keepdim=True)
        std = valid_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        for split_name, split in (("train", train), ("validation", validation), ("test", test)):
            x = split["modalities"][modality_idx]
            lengths = split["modality_lengths"][modality_idx]
            mask = (torch.arange(x.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(-1)
            normalized = torch.where(mask, (x - mean) / std, torch.zeros_like(x))
            outputs[split_name]["modalities"].append(normalized.contiguous())
    return outputs["train"], outputs["validation"], outputs["test"]


def _session_loso_group_validation_masks(
    rows: list[dict],
    labels: torch.Tensor,
    sessions: torch.Tensor,
    train_sessions: list[int],
    test_sessions: list[int],
    validation_folds: int,
    validation_fold_index: int,
    validation_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Create a LOSO test split and a dialog-disjoint validation fold."""
    train_set = {int(value) for value in train_sessions}
    test_set = {int(value) for value in test_sessions}
    if not train_set or len(test_set) != 1 or train_set & test_set:
        raise ValueError(
            "IEMOCAP LOSO requires non-empty train_sessions and exactly one disjoint test session."
        )
    available_sessions = {int(value) for value in sessions.tolist()}
    if train_set | test_set != available_sessions:
        raise ValueError(
            "IEMOCAP LOSO train/test sessions must cover exactly the processed sessions; "
            f"configured={sorted(train_set | test_set)}, available={sorted(available_sessions)}."
        )
    if validation_folds < 2:
        raise ValueError("IEMOCAP dataset.validation_folds must be at least 2.")
    if not 0 <= validation_fold_index < validation_folds:
        raise ValueError(
            "IEMOCAP dataset.validation_fold_index must be in "
            f"[0, {validation_folds - 1}]."
        )

    candidate_indices = np.asarray(
        [idx for idx, value in enumerate(sessions.tolist()) if int(value) in train_set],
        dtype=np.int64,
    )
    candidate_labels = labels[candidate_indices].numpy()
    candidate_groups = np.asarray(
        [str(rows[idx].get("dialog_id", "")) for idx in candidate_indices], dtype=object
    )
    if any(not value for value in candidate_groups.tolist()):
        raise ValueError("IEMOCAP LOSO validation requires a non-empty dialog_id for every sample.")
    unique_groups = np.unique(candidate_groups)
    if unique_groups.size < validation_folds:
        raise ValueError(
            "IEMOCAP does not contain enough training dialogs for "
            f"validation_folds={validation_folds}."
        )

    splitter = StratifiedGroupKFold(
        n_splits=validation_folds,
        shuffle=True,
        random_state=validation_seed,
    )
    folds = list(
        splitter.split(
            np.zeros(candidate_indices.shape[0], dtype=np.float32),
            candidate_labels,
            candidate_groups,
        )
    )
    train_relative, validation_relative = folds[validation_fold_index]
    train_indices = candidate_indices[train_relative]
    validation_indices = candidate_indices[validation_relative]
    test_indices = np.asarray(
        [idx for idx, value in enumerate(sessions.tolist()) if int(value) in test_set],
        dtype=np.int64,
    )

    train_groups = set(candidate_groups[train_relative].tolist())
    validation_groups = set(candidate_groups[validation_relative].tolist())
    if train_groups & validation_groups:
        raise RuntimeError("IEMOCAP dialog leakage detected between train and validation splits.")
    expected_labels = set(range(len(IEMOCAP_CLASS_NAMES)))
    for split_name, indices in (
        ("train", train_indices),
        ("validation", validation_indices),
        ("test", test_indices),
    ):
        actual = {int(value) for value in labels[indices].tolist()}
        if actual != expected_labels:
            raise ValueError(
                f"IEMOCAP LOSO {split_name} split does not contain all four classes: "
                f"found={sorted(actual)}."
            )

    def as_mask(indices: np.ndarray) -> torch.Tensor:
        mask = torch.zeros(labels.shape[0], dtype=torch.bool)
        mask[torch.as_tensor(indices, dtype=torch.long)] = True
        return mask

    metadata = {
        "strategy": "session_loso_5fold_group_validation_v1",
        "train_sessions": sorted(train_set),
        "test_sessions": sorted(test_set),
        "validation_folds": int(validation_folds),
        "validation_fold_index": int(validation_fold_index),
        "validation_seed": int(validation_seed),
        "train_dialogs": len(train_groups),
        "validation_dialogs": len(validation_groups),
    }
    return (
        as_mask(train_indices),
        as_mask(validation_indices),
        as_mask(test_indices),
        metadata,
    )


def load_iemocap_dataset(cfg: dict, project_root: Path) -> dict:
    dataset_cfg = cfg.get("dataset", {})
    _validate_iemocap_protocol(dataset_cfg)
    raw_root = _resolve_project_path(
        project_root,
        dataset_cfg.get(
            "root",
            "./local/datasets/IEMOCAP/IEMOCAP_full/IEMOCAP_full_release",
        ),
    )
    processed_root = _resolve_project_path(
        project_root,
        dataset_cfg.get(
            "processed_root",
            "./local/datasets/IEMOCAP/processed/mfcc_mobilevit_xs_distilbert_v1",
        ),
    )
    if not raw_root.exists():
        raise FileNotFoundError(f"IEMOCAP full root not found: {raw_root}")

    rows = _load_manifest(processed_root)
    manifest_ids = [str(row["utterance_id"]) for row in rows]
    payloads = [
        _load_feature_payload(processed_root / f"{name}.pt", name)
        for name in IEMOCAP_MODALITY_NAMES
    ]
    for name, payload in zip(IEMOCAP_MODALITY_NAMES, payloads):
        cache_ids = [str(value) for value in payload["sample_ids"]]
        if cache_ids != manifest_ids:
            raise ValueError(f"IEMOCAP {name} cache sample order does not match manifest.json.")

    labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)
    sessions = torch.tensor([int(row["session_id"]) for row in rows], dtype=torch.long)
    expected_labels = set(range(len(IEMOCAP_CLASS_NAMES)))
    actual_labels = {int(value) for value in labels.tolist()}
    if not actual_labels.issubset(expected_labels):
        raise ValueError(f"IEMOCAP manifest contains unsupported labels: {sorted(actual_labels)}")

    split_strategy = str(dataset_cfg.get("split_strategy", "fixed_session_split_v1"))
    split_metadata = {"strategy": split_strategy}
    if split_strategy == "session_loso_5fold_group_validation_v1":
        train_mask, validation_mask, test_mask, split_metadata = (
            _session_loso_group_validation_masks(
                rows=rows,
                labels=labels,
                sessions=sessions,
                train_sessions=list(dataset_cfg.get("train_sessions", [])),
                test_sessions=list(dataset_cfg.get("test_sessions", [])),
                validation_folds=int(dataset_cfg.get("validation_folds", 8)),
                validation_fold_index=int(dataset_cfg.get("validation_fold_index", 0)),
                validation_seed=int(dataset_cfg.get("validation_seed", cfg.get("seed", 42))),
            )
        )
    elif split_strategy == "fixed_session_split_v1":
        train_sessions = list(dataset_cfg.get("train_sessions", [1, 2, 3]))
        validation_sessions = list(dataset_cfg.get("validation_sessions", [4]))
        test_sessions = list(dataset_cfg.get("test_sessions", [5]))
        session_splits = _validate_subject_splits(
            train_sessions,
            validation_sessions,
            test_sessions,
            "IEMOCAP",
        )
        available_sessions = {int(value) for value in sessions.tolist()}
        requested_sessions = set().union(*session_splits.values())
        if requested_sessions != available_sessions:
            raise ValueError(
                "IEMOCAP configured sessions must cover exactly the processed sessions; "
                f"configured={sorted(requested_sessions)}, available={sorted(available_sessions)}."
            )
        train_mask = torch.tensor(
            [int(value) in session_splits["train"] for value in sessions.tolist()],
            dtype=torch.bool,
        )
        validation_mask = torch.tensor(
            [int(value) in session_splits["validation"] for value in sessions.tolist()],
            dtype=torch.bool,
        )
        test_mask = torch.tensor(
            [int(value) in session_splits["test"] for value in sessions.tolist()],
            dtype=torch.bool,
        )
        split_metadata.update(
            {
                "train_sessions": sorted(session_splits["train"]),
                "validation_sessions": sorted(session_splits["validation"]),
                "test_sessions": sorted(session_splits["test"]),
            }
        )
    else:
        raise ValueError(f"Unsupported IEMOCAP dataset.split_strategy: {split_strategy!r}.")

    def build_split(mask, split_name):
        if not bool(mask.any()) and split_name != "validation":
            raise ValueError(f"IEMOCAP {split_name} split produced no samples.")
        return {
            "modalities": [payload["features"][mask].contiguous() for payload in payloads],
            "modality_lengths": [payload["lengths"][mask].contiguous() for payload in payloads],
            "labels": labels[mask].contiguous(),
        }

    train = build_split(train_mask, "train")
    validation = build_split(validation_mask, "validation")
    test = build_split(test_mask, "test")
    if bool(dataset_cfg.get("normalize", True)):
        train, validation, test = _standardize_valid_sequences(train, validation, test)

    input_shapes = [[int(value) for value in payload["features"].shape[1:]] for payload in payloads]
    return {
        "train": train,
        "validation": validation,
        "test": test,
        "root": str(raw_root),
        "processed_root": str(processed_root),
        "modality_names": list(IEMOCAP_MODALITY_NAMES),
        "modality_input_shapes": input_shapes,
        "modality_encoder_types": list(IEMOCAP_ENCODER_TYPES),
        "label_mapping": {
            "angry": 0,
            "happy_or_excited": 1,
            "sad": 2,
            "neutral": 3,
        },
        "split_num_samples": {
            "train": int(train["labels"].shape[0]),
            "validation": int(validation["labels"].shape[0]),
            "test": int(test["labels"].shape[0]),
        },
        "split_metadata": split_metadata,
        "feature_extractors": {
            name: payload["feature_extractor"]
            for name, payload in zip(IEMOCAP_MODALITY_NAMES, payloads)
        },
    }
