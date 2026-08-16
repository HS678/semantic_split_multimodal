import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from MSL.data.dataset_defaults import DATASET_DEFAULTS
from MSL.evaluation.metrics import discovery_metrics
from MSL.utils.experiment_args import build_experiment_config, apply_experiment_overrides
from MSL.utils.results import dataset_result_name, partition_signature, safe_result_component


FORMAL_SEEDS = [42, 123, 2025, 3407, 7777]
RQ1_METHODS = ["adaptive_isodata", "kmeans2", "kmeans3", "kmeans5"]
RQ2_METHODS = ["ours", "randomsl", "kmeans2", "kmeans3", "kmeans5", "oracle"]


# 返回当前项目根目录。
def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "MSL").is_dir():
            return parent
    raise RuntimeError("Cannot locate project root containing src/MSL.")


# 读取当前 git commit，缺失时返回 unknown。
def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# 根据 dataset/fold/seed 构建冻结协议配置。
def resolved_cfg(dataset: str, fold: int | None, seed: int) -> dict:
    cfg = build_experiment_config(dataset_type=dataset, seed=seed)
    cfg = apply_experiment_overrides(cfg, fold=fold)
    return cfg


# 返回某个数据集的正式 fold 列表。
def formal_folds(dataset: str) -> list[int | None]:
    fold_count = DATASET_DEFAULTS[str(dataset)]["fold_count"]
    if fold_count is None:
        return [None]
    return list(range(1, int(fold_count) + 1))


# 尝试从已有真实 Stage1 目录中定位数据划分产物。
def find_stage1_dir(root: Path, cfg: dict) -> Path:
    dataset_name = safe_result_component(dataset_result_name(cfg))
    split_protocol = cfg.get("dataset", {}).get("split_protocol")
    candidates = []
    for base in [root / "results" / "MSL", root / "local" / "results_msl"]:
        partition_root = base / "partition" / dataset_name
        if not partition_root.exists():
            continue
        if split_protocol:
            candidates.extend(sorted(partition_root.glob(f"*__{safe_result_component(split_protocol)}")))
        candidates.extend(sorted(partition_root.glob("*")))
    for candidate in candidates:
        if (candidate / "train_clients").is_dir() and (candidate / "test_multimodal.pt").exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Missing real Stage1 partition. Expected train_clients/ and test_multimodal.pt for "
        f"dataset={dataset_name}, split_protocol={split_protocol}."
    )


# 尝试从已有真实 Stage2 目录中定位 discovery 产物。
def find_stage2_dir(root: Path, stage1_dir: Path, method: str = "adaptive_isodata") -> Path:
    dataset_name = stage1_dir.parent.name
    partition_name = stage1_dir.name
    candidates = [
        root / "results" / "MSL" / "cluster" / dataset_name / partition_name / method,
        root / "local" / "results_msl" / "cluster" / dataset_name / partition_name / method,
    ]
    for candidate in candidates:
        if (candidate / "pred_cluster.csv").exists() and (candidate / "pretrained_encoders").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Missing real Stage2 discovery artifacts. Expected pred_cluster.csv and pretrained_encoders/ under "
        f"{dataset_name}/{partition_name}/{method}."
    )


# 读取 adaptive Stage2 保存的 fingerprint 矩阵。
def load_fingerprint_npz(stage2_dir: Path) -> dict:
    path = stage2_dir / "visualization" / "fingerprints.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing fingerprints.npz: {path}")
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}


# 保存 JSON 文件并创建父目录。
def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)


# 构建正式结果 metadata 的公共字段。
def runtime_metadata(root: Path, dataset: str, fold: int | None, seed: int, method: str) -> dict:
    return {
        "dataset": dataset,
        "fold": fold,
        "seed": int(seed),
        "method": method,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": git_commit(root),
    }


# 检查正式实验入口没有导入 mock/synthetic/dummy 数据模块。
def assert_no_mock_pipeline_imports() -> None:
    forbidden = ["synthetic", "mock", "dummy", "random_dataset"]
    loaded = [name for name in sys.modules if any(token in name.lower() for token in forbidden)]
    allowed = [name for name in loaded if name.startswith(("tests", "_pytest", "pytest", "unittest.mock"))]
    violations = sorted(set(loaded) - set(allowed))
    if violations:
        raise RuntimeError(f"Formal experiment imported forbidden mock/synthetic modules: {violations}")


# 给实验脚本添加公共 dataset/fold/seed 参数。
def add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=tuple(DATASET_DEFAULTS), required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-root", default="results")
