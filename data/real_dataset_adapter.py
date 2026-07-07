'''
将整理好的真实多模态 .npz 数据，加载成训练代码统一使用的格式：
    {
        "train": {
            "modalities": [模态0数据, 模态1数据, ...],
            "labels": 标签
        },
        "test": {
            "modalities": [模态0数据, 模态1数据, ...],
            "labels": 标签
        }
    }
'''
from pathlib import Path
import numpy as np
import torch


def _npz_to_paired_dict(npz_path, num_modalities):
    '''
    从 .npz 文件加载数据，文件中应该包含 "mod_0", "mod_1", ..., "labels" 或 "y" 键。
    返回格式：
        {
            "modalities": [模态0数据, 模态1数据, ...],
            "labels": 标签
        }
    '''
    data = np.load(npz_path)
    labels_key = "labels" if "labels" in data.files else "y"
    if labels_key not in data.files:
        raise ValueError(f"{npz_path} must contain 'labels' or 'y'.")

    modalities = []
    for m in range(num_modalities):
        k = f"mod_{m}"
        if k not in data.files:
            raise ValueError(f"{npz_path} missing modality key: {k}")
        modalities.append(torch.tensor(data[k], dtype=torch.float32))

    labels = torch.tensor(data[labels_key], dtype=torch.long)
    n = labels.shape[0]
    for m, x in enumerate(modalities):
        if x.shape[0] != n:
            raise ValueError(f"Sample size mismatch in modality {m}: {x.shape[0]} vs labels {n}")

    return {"modalities": modalities, "labels": labels}


def load_real_paired_dataset(cfg):
    '''
    从配置指定的路径加载真实多模态数据，数据应该以 .npz 格式存储，并且包含 "mod_0", "mod_1", ..., "labels" 或 "y" 键。
    返回格式：
        {
            "train": train_set,
            "test": test_set
        }
    '''
    dataset_cfg = cfg.get("dataset", {})
    root = Path(dataset_cfg.get("root", "")).resolve()
    if not root.exists():
        raise FileNotFoundError(f"dataset.root not found: {root}")

    num_modalities = int(cfg["num_modalities"])

    train_file = dataset_cfg.get("train_file", "train_paired.npz")
    test_file = dataset_cfg.get("test_file", "test_paired.npz")
    full_file = dataset_cfg.get("full_file", "")

    if full_file:
        full = _npz_to_paired_dict(root / full_file, num_modalities)
        ratio = float(cfg.get("train_split_ratio", 0.8))
        train_n = int(len(full["labels"]) * ratio)
        train_set = {
            "modalities": [x[:train_n] for x in full["modalities"]],
            "labels": full["labels"][:train_n],
        }
        test_set = {
            "modalities": [x[train_n:] for x in full["modalities"]],
            "labels": full["labels"][train_n:],
        }
    else:
        train_set = _npz_to_paired_dict(root / train_file, num_modalities)
        test_set = _npz_to_paired_dict(root / test_file, num_modalities)

    return {"train": train_set, "test": test_set}
