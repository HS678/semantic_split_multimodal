# 读取UCI-HAR数据集，并且数据按照模态分开存储在不同的键下，标签存储在 "labels" 或 "y" 键下。
'''
UCI-HAR数据集的目录结构应该如下：
uci-har/
    train/
        Inertial Signals/
            body_acc_x_train.txt
            body_acc_y_train.txt
            body_acc_z_train.txt
            body_gyro_x_train.txt
            body_gyro_y_train.txt
            body_gyro_z_train.txt
        y_train.txt
    test/   
        Inertial Signals/   
            body_acc_x_test.txt
            body_acc_y_test.txt
            body_acc_z_test.txt     
            body_gyro_x_test.txt
            body_gyro_y_test.txt
            body_gyro_z_test.txt
        y_test.txt  
每个模态的数据被存储在一个单独的键下，标签存储在 "labels" 或 "y" 键下。返回格式如下：
{
    "train": {
        "modalities": [模态0数据, 模态1数据, ...],
        "labels": 标签      
    },
    "test": {
        "modalities": [模态0数据, 模态1数据, ...],
        "labels": 标签              
    },
    "root": 数据集根目录路径,
    "modality_input_dims": [模态0输入维度, 模态1输入维度, ...]  # 可选，提供每个模态的输入维度信息
}   
'''
from pathlib import Path
import numpy as np
import torch


def _read_signal_matrix(path: Path):
    return np.loadtxt(path, dtype=np.float32)


def _build_modality_vectors(root: Path, split: str):
    split_dir = root / split
    sig = split_dir / "Inertial Signals"

    # fed-multimodal compatible channels for UCI-HAR
    acc_x = _read_signal_matrix(sig / f"body_acc_x_{split}.txt")
    acc_y = _read_signal_matrix(sig / f"body_acc_y_{split}.txt")
    acc_z = _read_signal_matrix(sig / f"body_acc_z_{split}.txt")

    gyro_x = _read_signal_matrix(sig / f"body_gyro_x_{split}.txt")
    gyro_y = _read_signal_matrix(sig / f"body_gyro_y_{split}.txt")
    gyro_z = _read_signal_matrix(sig / f"body_gyro_z_{split}.txt")

    labels = np.loadtxt(split_dir / f"y_{split}.txt", dtype=np.int64) - 1
    n = labels.shape[0]

    # modality-isolated tensors; no shared full-channel tensor
    acc_only = np.stack([acc_x, acc_y, acc_z], axis=1).astype(np.float32)  # [N, 3, 128]
    gyro_only = np.stack([gyro_x, gyro_y, gyro_z], axis=1).astype(np.float32)  # [N, 3, 128]

    acc_vec = acc_only.reshape(n, -1)  # [N, 384]
    gyro_vec = gyro_only.reshape(n, -1)  # [N, 384]

    return {
        "modalities": [torch.tensor(acc_vec, dtype=torch.float32), torch.tensor(gyro_vec, dtype=torch.float32)],
        "labels": torch.tensor(labels, dtype=torch.long),
        "modality_input_dims": [int(acc_vec.shape[1]), int(gyro_vec.shape[1])],
    }


def load_uci_har_dataset(cfg, project_root: Path):
    dataset_cfg = cfg.get("dataset", {})
    root_cfg = dataset_cfg.get("root", "./data/uci-har")
    root = Path(root_cfg)
    if not root.is_absolute():
        root = project_root / root
    root = root.resolve()

    if not root.exists():
        raise FileNotFoundError(f"UCI-HAR root not found: {root}")

    train = _build_modality_vectors(root, "train")
    test = _build_modality_vectors(root, "test")
    return {"train": train, "test": test, "root": str(root), "modality_input_dims": train["modality_input_dims"]}
