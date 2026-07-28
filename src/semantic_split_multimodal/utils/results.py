from datetime import datetime
from pathlib import Path
import re

import yaml


SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


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


def partition_signature(modality_names, clients_per_modality: int) -> str:
    modality_part = "-".join(safe_result_component(name) for name in modality_names)
    return f"{modality_part}_{int(clients_per_modality)}clients"


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

    dataset_dir = base_dir / "experiment" / dataset_name
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
        "data_partition": str(run_dir / "01_dataset_partition"),
        "cluster": str(run_dir / "02_cluster_results"),
        "logs": str(run_dir / "03_training_evaluation"),
        "models": str(run_dir / "04_model_artifacts"),
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
    with (run_dir / "run_meta.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(run_meta, f, sort_keys=False)
    return cfg
