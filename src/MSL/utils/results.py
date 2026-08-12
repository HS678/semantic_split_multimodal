import hashlib
import json
from pathlib import Path
import re


SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")
_NON_IDENTITY_KEYS = {
    "base_dir",
    "dataset_dir",
    "output_dir",
    "run_dir",
    "run_id",
    "stage1_dir",
    "stage2_dir",
    "output_root",
    "attempt",
    "root",
    "processed_root",
}


def _resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def dataset_result_name(cfg: dict) -> str:
    dataset_cfg = cfg.get("dataset", {})
    return str(dataset_cfg.get("name", dataset_cfg.get("type", "dataset"))).strip().lower()


def safe_result_component(value) -> str:
    text = str(value).strip().lower()
    text = SAFE_COMPONENT.sub("_", text)
    text = text.strip("._-")
    return text or "default"


def partition_signature(modality_names, clients_per_modality: int, split_protocol: str | None = None) -> str:
    signature = "_".join(
        f"{safe_result_component(name)}_{int(clients_per_modality)}clients"
        for name in modality_names
    )
    if split_protocol:
        signature = f"{signature}__{safe_result_component(split_protocol)}"
    return signature


def cluster_assignment_scope(cfg: dict) -> str:
    source = str(
        cfg.get("training", {}).get("cluster_assignment_source", "pred_cluster")
    ).strip().lower()
    if source == "true_cluster":
        return "oracle_true_cluster"
    if source == "pred_cluster":
        return "predicted_cluster"
    raise ValueError(
        "training.cluster_assignment_source must be 'pred_cluster' or 'true_cluster', "
        f"got {source!r}."
    )


def _identity_payload(value):
    if isinstance(value, dict):
        return {
            key: _identity_payload(item)
            for key, item in sorted(value.items())
            if key not in _NON_IDENTITY_KEYS and key not in {"seed", "device"}
        }
    if isinstance(value, (list, tuple)):
        return [_identity_payload(item) for item in value]
    return value


def experiment_config_signature(cfg: dict) -> str:
    payload = _identity_payload(cfg)
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]
    encoder_cfg = cfg.get("model", {}).get("encoder", {})
    dataset_cfg = cfg.get("dataset", {})
    encoder_parts = [encoder_cfg.get("type"), dataset_cfg.get("feature_recipe")]
    objective = cfg.get("fusion", {}).get("training_objective", "objective")
    # 目录名保留 loss 方式 + 配置哈希，便于快速识别训练目标。
    return f"{safe_result_component(objective)}-h-{digest}"


def configure_result_run(cfg: dict, project_root: Path) -> dict:
    """Stage1 专用：解析 base_dir 并生成 partition 输出目录（含协议签名字段）。"""
    cfg = dict(cfg)
    result_cfg = dict(cfg.get("results", {}))
    base_dir_value = cfg.get("base_dir") or result_cfg.get("base_dir", "./results/MSL")
    base_dir = _resolve_project_path(project_root, base_dir_value)
    dataset_name = dataset_result_name(cfg)
    dataset_dir = base_dir / "partition" / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    clients_per_modality = int(
        cfg.get("partition", {}).get("clients_per_modality", cfg.get("clients_per_modality", 10))
    )
    cfg["results"] = {
        **result_cfg,
        "base_dir": str(base_dir),
        "dataset_dir": str(dataset_dir),
        "run_dir": str(dataset_dir),
        "data_partition": str(dataset_dir),
    }
    cfg["partition"] = {
        **cfg.get("partition", {}),
        "output_dir": str(dataset_dir),
        "auto_signature_dir": True,
        "clients_per_modality": clients_per_modality,
    }
    return cfg


def resolve_stage_paths(cfg: dict, project_root: Path) -> dict:
    """从 base_dir + 数据集 + 划分协议自动生成 Stage1/2/3 路径（config 无需手写路径）。"""
    from MSL.data.registry import load_dataset

    result_cfg = dict(cfg.get("results", {}))
    base_dir_value = cfg.get("base_dir") or result_cfg.get("base_dir", "./results/MSL")
    base_dir = _resolve_project_path(project_root, base_dir_value)
    dataset_name = dataset_result_name(cfg)
    dataset = load_dataset(cfg, project_root)
    signature = partition_signature(
        dataset["modality_names"],
        int(cfg.get("partition", {}).get("clients_per_modality", 10)),
        cfg.get("dataset", {}).get("split_protocol"),
    )
    return {
        "stage1_dir": base_dir / "partition" / dataset_name / signature,
        "stage2_dir": base_dir / "cluster" / dataset_name / signature / "adaptive_isodata",
        "output_dir": base_dir / "experiments",
    }
