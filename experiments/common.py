import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from MSL.protocol import DATASET_DEFAULTS, DEFAULT_ADAPTIVE
from MSL.protocol import build_experiment_config, apply_experiment_overrides
from MSL.utils import dataset_result_name, safe_result_component


FORMAL_SEEDS = [42, 123, 2025, 3407, 7777]
FORMAL_CV_SEED = 42
DISCOVERY_METHODS = ["adaptive_isodata", "kmeans2", "kmeans3", "kmeans4", "kmeans5"]
TRAINING_METHODS = ["ours", "randomsl", "kmeans2", "kmeans3", "kmeans4", "kmeans5", "oracle"]


# 返回当前项目根目录。
def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "MSL").is_dir():
            return parent
    raise RuntimeError("Cannot locate project root containing src/MSL.")


# 读取当前 git commit，缺失时返回 unknown。
def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# 根据 dataset/fold/seed 构建冻结协议配置。
def resolved_cfg(dataset: str, fold: int | None, seed: int) -> dict:
    cfg = build_experiment_config(dataset_type=dataset, seed=seed)
    cfg = apply_experiment_overrides(cfg, fold=fold)
    if fold is None and DATASET_DEFAULTS[str(dataset)]["fold_count"] is None:
        cfg = with_repeated_seed_split_signature(cfg, seed)
    return cfg


# 返回某个数据集的正式 fold 列表。
def formal_folds(dataset: str) -> list[int | None]:
    fold_count = DATASET_DEFAULTS[str(dataset)]["fold_count"]
    if fold_count is None:
        return [None]
    return list(range(1, int(fold_count) + 1))


# 返回正式交叉验证协议下的 (fold, seed) 运行组合。
def formal_run_grid(dataset: str, seeds: list[int] | None = None) -> list[tuple[int | None, int]]:
    folds = formal_folds(dataset)
    if DATASET_DEFAULTS[str(dataset)]["fold_count"] is None:
        run_seeds = FORMAL_SEEDS if seeds is None else [int(seed) for seed in seeds]
        return [(None, int(seed)) for seed in run_seeds]
    if seeds is None:
        return [(fold, FORMAL_CV_SEED) for fold in folds]
    return [(fold, int(seed)) for fold in folds for seed in seeds]


# 无 CV 数据集用多个随机种子重复实验时，把 seed 写入 split signature 以避免 pipeline 产物互相覆盖。
def repeated_seed_split_protocol(dataset: str, seed: int) -> str:
    dataset_cfg = DATASET_DEFAULTS[str(dataset)]["dataset"]
    base_protocol = str(dataset_cfg.get("split_protocol", "subject_disjoint"))
    return f"{base_protocol}_seed{int(seed)}"


# 为无 CV 数据集写入带 seed 的正式结果签名。
def with_repeated_seed_split_signature(cfg: dict, seed: int) -> dict:
    cfg = dict(cfg)
    dataset = dict(cfg.get("dataset", {}))
    dataset_type = str(dataset.get("type"))
    dataset["split_protocol"] = repeated_seed_split_protocol(dataset_type, int(seed))
    cfg["dataset"] = dataset
    cfg["runtime_overrides"] = {
        **dict(cfg.get("runtime_overrides", {})),
        "fold": None,
        "split_protocol": dataset["split_protocol"],
        "repeated_seed": int(seed),
    }
    return cfg


# 返回结果目录中的 fold 层级名。
def fold_result_component(fold: int | None) -> str:
    return "fold_00" if fold is None else f"fold_{int(fold):02d}"


# 返回结果目录中的 seed 层级名。
def seed_result_component(seed: int) -> str:
    return f"seed_{int(seed)}"


# 返回单次实验运行的层级结果目录。
def formal_result_dir(results_root: Path, family: str, dataset: str, method: str, fold: int | None, seed: int) -> Path:
    return (
        Path(results_root)
        / str(family)
        / safe_result_component(dataset)
        / safe_result_component(method)
        / fold_result_component(fold)
        / seed_result_component(seed)
    )


# 尝试从已有真实 client partition 目录中定位数据划分产物。
def find_clients_dir(root: Path, cfg: dict) -> Path:
    dataset_name = safe_result_component(dataset_result_name(cfg))
    split_protocol = cfg.get("dataset", {}).get("split_protocol")
    candidates = []
    for base in [root / "results" / "MSL", root / "local" / "results_msl"]:
        partition_root = base / "partition" / dataset_name
        if not partition_root.exists():
            continue
        if split_protocol:
            candidates.extend(sorted(partition_root.glob(f"*__{safe_result_component(split_protocol)}")))
        else:
            candidates.extend(sorted(partition_root.glob("*")))
    for candidate in candidates:
        if (candidate / "train_clients").is_dir() and (candidate / "test_multimodal.pt").exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Missing real client partition. Expected train_clients/ and test_multimodal.pt for "
        f"dataset={dataset_name}, split_protocol={split_protocol}."
    )


# 尝试从已有真实 modality discovery 目录中定位 discovery 产物。
def find_discovery_dir(root: Path, clients_dir: Path, method: str = "adaptive_isodata") -> Path:
    dataset_name = clients_dir.parent.name
    partition_name = clients_dir.name
    candidates = [
        root / "results" / "MSL" / "cluster" / dataset_name / partition_name / method,
        root / "local" / "results_msl" / "cluster" / dataset_name / partition_name / method,
    ]
    for candidate in candidates:
        if (candidate / "pred_cluster.csv").exists() and (candidate / "pretrained_encoders").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Missing real modality discovery artifacts. Expected pred_cluster.csv and pretrained_encoders/ under "
        f"{dataset_name}/{partition_name}/{method}."
    )


# 读取 adaptive modality discovery 保存的 fingerprint 矩阵。
def load_fingerprint_npz(discovery_dir: Path) -> dict:
    path = discovery_dir / "visualization" / "fingerprints.npz"
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


# 对配置快照计算稳定 hash，供 resume 校验。
def stable_config_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# 构建正式实验协议 manifest；timestamp 只用于记录，不参与 protocol_hash。
def build_protocol_manifest(results_root: Path | None = None) -> dict:
    root = project_root()
    datasets = {}
    for dataset_name, defaults in DATASET_DEFAULTS.items():
        folds = formal_folds(dataset_name)
        representative_fold = folds[0]
        representative_cfg = resolved_cfg(dataset_name, representative_fold, FORMAL_CV_SEED)
        dataset_cfg = representative_cfg["dataset"]
        datasets[dataset_name] = {
            "fold_count": defaults["fold_count"],
            "folds": folds,
            "run_grid": [
                {"fold": fold, "seed": int(seed)}
                for fold, seed in formal_run_grid(dataset_name)
            ],
            "split_protocol": dataset_cfg.get("split_protocol"),
            "split_protocol_template": defaults.get("dataset", {}).get("split_protocol_template"),
            "clients_per_modality": int(representative_cfg["partition"]["clients_per_modality"]),
            "global_rounds": int(representative_cfg["training"]["global_rounds"]),
            "local_steps": int(representative_cfg["training"]["local_steps"]),
            "batch_size": int(representative_cfg["training"]["batch_size"]),
            "eval_batch_size": int(representative_cfg["training"]["eval_batch_size"]),
            "r": int(representative_cfg["training"]["clients_per_cluster_per_round"]),
            "training_objective": representative_cfg["fusion"]["training_objective"],
            "loss_weights": dict(representative_cfg["fusion"]["mmbind"]),
            "fingerprint_type": representative_cfg["fingerprint"]["type"],
            "adaptive_isodata": dict(representative_cfg["cluster"]["adaptive"]),
            "kmeans_configs": {"kmeans2": 2, "kmeans3": 3, "kmeans4": 4, "kmeans5": 5},
            "evaluation_protocol": {
                "selection": "fixed_rounds_no_validation",
                "test_loss": "classification_cross_entropy",
                "routing": "tolerant_activation_ensemble",
            },
            "cv_protocol": {
                "uci_har": "official subject-disjoint train/test, repeated formal seeds",
                "mhealth": "subject_5fold, one fixed seed per fold",
                "pamap2": "subject_8fold_loso excluding subject 109, one fixed seed per fold",
                "iemocap": "session_5fold_loso, one fixed seed per fold",
            }.get(dataset_name),
        }
    manifest = {
        "protocol_version": "cv_revision_iemocap_300_rounds_kmeans4_2026_08_17",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "results_root": None if results_root is None else str(Path(results_root)),
        "git_commit": git_commit(root),
        "formal_seeds": list(FORMAL_SEEDS),
        "formal_cv_seed": int(FORMAL_CV_SEED),
        "discovery_methods": list(DISCOVERY_METHODS),
        "training_methods": list(TRAINING_METHODS),
        "default_adaptive_isodata": dict(DEFAULT_ADAPTIVE),
        "datasets": datasets,
        "yield_definition": {
            "name": "pseudo_samples_over_requested_tuple_budget",
            "formula": "pseudo_batch_size_per_round / (binding.batch_size * attempted_local_steps)",
            "numerator": "actual complete same-label pseudo multimodal tuples built in the round",
            "denominator": "configured requested pseudo tuple budget for that round",
        },
    }
    manifest["protocol_hash"] = protocol_hash(manifest)
    return manifest


# 对正式协议计算 deterministic hash，忽略 timestamp 和输出路径。
def protocol_hash(manifest_or_payload: dict | None = None) -> str:
    payload = build_protocol_manifest(None) if manifest_or_payload is None else dict(manifest_or_payload)
    payload.pop("timestamp", None)
    payload.pop("results_root", None)
    payload.pop("protocol_hash", None)
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# 写出冻结协议 manifest。
def write_protocol_manifest(results_root: Path) -> dict:
    manifest = build_protocol_manifest(results_root)
    write_json(Path(results_root) / "protocol_manifest.json", manifest)
    return manifest


# 标记运行是 formal 还是隔离 smoke。
def run_type_metadata(results_root: Path) -> dict:
    name_parts = {part.lower() for part in Path(results_root).parts}
    is_smoke = any("smoke" in part for part in name_parts)
    return {"run_type": "smoke" if is_smoke else "formal", "formal": not is_smoke}


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
