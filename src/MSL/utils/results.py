from datetime import datetime
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


def _timestamp_ms():
    now = datetime.now()
    return now.strftime("%y_%m_%d_%H_%M_%S_") + f"{now.microsecond // 1000:03d}"


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
    encoder = "-".join(str(part) for part in encoder_parts if part) or "encoder"
    binding = cfg.get("binding", {}).get("type", "binding")
    objective = cfg.get("fusion", {}).get("training_objective", "objective")
    scheduler = cfg.get("training", {}).get("scheduler", "scheduler")
    readable = (
        f"enc-{safe_result_component(encoder)}"
        f"__bind-{safe_result_component(binding)}"
        f"__loss-{safe_result_component(objective)}"
        f"__sched-{safe_result_component(scheduler)}"
    )
    return f"{readable}__h-{digest}"


def configure_result_run(cfg: dict, project_root: Path, stage: str, create_new: bool = False) -> dict:
    cfg = dict(cfg)
    result_cfg = dict(cfg.get("results", {}))
    base_dir = _resolve_project_path(project_root, result_cfg.get("base_dir", "./local/results"))
    dataset_name = dataset_result_name(cfg)
    if stage == "stage1_partition":
        dataset_dir = base_dir / "partition" / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        clients_per_modality = int(
            cfg.get("partition", {}).get("clients_per_modality", cfg.get("clients_per_modality", 10))
        )
        paths = {
            "base_dir": str(base_dir),
            "dataset_dir": str(dataset_dir),
            "run_id": "",
            "run_dir": str(dataset_dir),
            "data_partition": str(dataset_dir),
            "cluster": "",
            "logs": "",
            "models": "",
        }
        cfg["results"] = {**result_cfg, **paths}
        cfg["partition"] = {
            **cfg.get("partition", {}),
            "output_dir": str(dataset_dir),
            "auto_signature_dir": True,
            "clients_per_modality": clients_per_modality,
        }
        return cfg

    dataset_dir = base_dir / "experiments" / dataset_name
    latest_path = dataset_dir / "latest_run.txt"

    run_id = result_cfg.get("run_id")
    if not run_id:
        if create_new:
            run_id = _timestamp_ms()
        else:
            if not latest_path.exists():
                raise FileNotFoundError(
                    f"Missing latest run marker: {latest_path}. Run Stage 1 first or set results.run_id."
                )
            run_id = latest_path.read_text(encoding="utf-8").strip()
    run_dir = dataset_dir / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if create_new:
        latest_path.write_text(str(run_id), encoding="utf-8")

    paths = {
        "base_dir": str(base_dir),
        "dataset_dir": str(dataset_dir),
        "run_id": str(run_id),
        "run_dir": str(run_dir),
        "data_partition": str(base_dir / "partition" / dataset_name),
        "cluster": str(base_dir / "cluster" / dataset_name),
        "logs": str(run_dir),
        "models": str(run_dir),
    }

    cfg["results"] = {**result_cfg, **paths}
    cfg["partition"] = {**cfg.get("partition", {}), "output_dir": paths["data_partition"]}
    cfg["cluster"] = {**cfg.get("cluster", {}), "output_dir": paths["cluster"]}
    cfg["result"] = {**cfg.get("result", {}), "output_dir": paths["logs"]}
    cfg["result_model"] = {**cfg.get("result_model", {}), "output_dir": paths["models"]}

    run_meta = {
        "dataset": dataset_name,
        "run_id": str(run_id),
        "stage": stage,
        "run_dir": str(run_dir),
        "paths": paths,
    }
    with (run_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False, sort_keys=True)
    return cfg
