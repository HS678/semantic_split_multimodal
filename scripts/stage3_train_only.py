import argparse
import csv
import json
from datetime import datetime, timezone
import math
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic_split_multimodal.learning.fusion_sl import run_mmbind_fusion_stage3_split_training
from semantic_split_multimodal.utils.config import load_config
from semantic_split_multimodal.utils.device import select_device
from semantic_split_multimodal.utils.results import dataset_result_name
from semantic_split_multimodal.utils.seed import set_seed


CODEX_RESULTS_ROOT = (ROOT / "local" / "results" / "codex").resolve()
SAFE_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _resolve(path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _paths_overlap(first: Path, second: Path) -> bool:
    return _is_relative_to(first, second) or _is_relative_to(second, first)


def _validate_path_component(value, label: str) -> str:
    text = str(value or "")
    if not text:
        raise ValueError(f"{label} must not be empty.")
    if Path(text).is_absolute() or "/" in text or "\\" in text or ".." in Path(text).parts:
        raise ValueError(f"{label} must be a single safe path component, got {text!r}.")
    if not SAFE_PATH_COMPONENT.match(text):
        raise ValueError(f"{label} must contain only letters, numbers, '.', '_', and '-', got {text!r}.")
    return text


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _git_output(args):
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_commit():
    return _git_output(["git", "rev-parse", "HEAD"])


def _git_branch():
    return _git_output(["git", "branch", "--show-current"])


def _git_dirty():
    status = _git_output(["git", "status", "--short"])
    return None if status is None else bool(status)


def _read_csv_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _require_readable_file(path: Path, label: str):
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    with path.open("rb"):
        pass


def _load_json(path: Path, label: str):
    _require_readable_file(path, label)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(path: Path, label: str):
    _require_readable_file(path, label)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_stage3_only_run(cfg: dict, stage1_dir, stage2_dir, output_root, tag, run_type="codex_test"):
    run_type = str(run_type)
    if run_type not in {"codex_test", "user_formal"}:
        raise ValueError("run_type must be 'codex_test' or 'user_formal'.")
    if not output_root:
        raise ValueError("--output-root is required.")

    stage1_path = _resolve(stage1_dir)
    stage2_path = _resolve(stage2_dir)
    output_root_path = _resolve(output_root)
    if run_type == "codex_test" and not _is_relative_to(output_root_path, CODEX_RESULTS_ROOT):
        raise ValueError(f"codex_test output_root must be under {CODEX_RESULTS_ROOT}, got {output_root_path}")

    dataset_name = _validate_path_component(dataset_result_name(cfg), "dataset")
    run_tag = _validate_path_component(tag, "tag")
    for input_label, input_path in [("Stage1", stage1_path), ("Stage2", stage2_path)]:
        if _paths_overlap(output_root_path, input_path):
            raise ValueError(f"output_root must not overlap {input_label} input directory: {input_path}")

    run_dir = (output_root_path / dataset_name / run_tag).resolve()
    if not _is_relative_to(run_dir, output_root_path):
        raise ValueError(f"Stage3 run directory escaped output_root: {run_dir}")
    for input_label, input_path in [("Stage1", stage1_path), ("Stage2", stage2_path)]:
        if _paths_overlap(run_dir, input_path):
            raise ValueError(f"Stage3 run directory must not overlap {input_label} input directory: {input_path}")
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing Stage3 run directory: {run_dir}")

    result_dir = run_dir / "03_training_evaluation"
    model_dir = run_dir / "04_model_artifacts"
    run_cfg = dict(cfg)
    run_cfg["partition"] = {**run_cfg.get("partition", {}), "output_dir": str(stage1_path)}
    run_cfg["cluster"] = {**run_cfg.get("cluster", {}), "output_dir": str(stage2_path)}
    run_cfg["result"] = {**run_cfg.get("result", {}), "output_dir": str(result_dir)}
    run_cfg["result_model"] = {**run_cfg.get("result_model", {}), "output_dir": str(model_dir)}
    run_cfg["stage3_only"] = {
        "run_type": run_type,
        "stage1_dir": str(stage1_path),
        "stage2_dir": str(stage2_path),
        "output_root": str(output_root_path),
        "run_dir": str(run_dir),
        "result_dir": str(result_dir),
        "model_dir": str(model_dir),
        "tag": run_tag,
    }
    paths = {
        "stage1_dir": stage1_path,
        "stage2_dir": stage2_path,
        "output_root": output_root_path,
        "run_dir": run_dir,
        "result_dir": result_dir,
        "model_dir": model_dir,
        "config_snapshot": run_dir / "stage3_only_config_used.yaml",
        "metadata": run_dir / "stage3_only_metadata.json",
        "run_type": run_type,
        "tag": run_tag,
        "dataset": dataset_name,
    }
    return run_cfg, paths


def audit_stage3_inputs(cfg: dict, stage1_dir: Path, stage2_dir: Path):
    stage1 = _audit_stage1_inputs(cfg, Path(stage1_dir))
    stage2 = _audit_stage2_inputs(cfg, Path(stage2_dir), stage1)
    return {"stage1": stage1, "stage2": stage2}


def _audit_stage1_inputs(cfg: dict, stage1_dir: Path):
    if not stage1_dir.exists() or not stage1_dir.is_dir():
        raise FileNotFoundError(f"Missing Stage1 partition directory: {stage1_dir}")
    train_dir = stage1_dir / "train_clients"
    if not train_dir.exists() or not train_dir.is_dir():
        raise FileNotFoundError(f"Missing Stage1 train_clients directory: {train_dir}")
    client_meta_path = stage1_dir / "client_meta.csv"
    test_multimodal_path = stage1_dir / "test_multimodal.pt"
    partition_config_path = stage1_dir / "partition_config.json"
    _require_readable_file(client_meta_path, "Stage1 client_meta.csv")
    _require_readable_file(test_multimodal_path, "Stage1 test_multimodal.pt")
    partition_config = _load_json(partition_config_path, "Stage1 partition_config.json")

    cfg_dataset = dataset_result_name(cfg)
    partition_dataset = str(partition_config.get("dataset_type", "")).strip().lower()
    if partition_dataset and partition_dataset != cfg_dataset:
        raise ValueError(f"Stage1 dataset mismatch: config={cfg_dataset}, partition={partition_dataset}")

    meta_rows = _read_csv_rows(client_meta_path)
    meta_ids = [row.get("client_id") for row in meta_rows]
    if not meta_ids or any(not client_id for client_id in meta_ids):
        raise ValueError("Stage1 client_meta.csv must contain non-empty client_id values.")
    if len(set(meta_ids)) != len(meta_ids):
        raise ValueError("Stage1 client_meta.csv contains duplicate client_id values.")

    client_files = sorted(train_dir.glob("client_*.pt"))
    if not client_files:
        raise FileNotFoundError(f"No Stage1 client_*.pt files found under {train_dir}.")
    payload_ids = []
    for path in client_files:
        _require_readable_file(path, f"Stage1 client payload {path.name}")
        payload = torch.load(path, map_location="cpu")
        client_id = str(payload.get("client_id", ""))
        if not client_id:
            raise ValueError(f"Stage1 client payload missing client_id: {path}")
        if "samples" not in payload or "labels" not in payload:
            raise ValueError(f"Stage1 client payload missing samples or labels: {path}")
        samples = payload["samples"]
        labels = payload["labels"]
        if int(samples.shape[0]) == 0:
            raise ValueError(f"Stage1 client payload is empty: {path}")
        if int(samples.shape[0]) != int(labels.shape[0]):
            raise ValueError(f"Stage1 samples/labels length mismatch: {path}")
        payload_ids.append(client_id)
    if len(set(payload_ids)) != len(payload_ids):
        raise ValueError("Stage1 train client payloads contain duplicate client_id values.")
    if set(payload_ids) != set(meta_ids):
        missing_meta = sorted(set(payload_ids) - set(meta_ids))
        missing_payload = sorted(set(meta_ids) - set(payload_ids))
        raise ValueError(f"Stage1 client IDs mismatch: missing_meta={missing_meta}, missing_payload={missing_payload}")

    return {
        "dataset": cfg_dataset,
        "partition_config": partition_config,
        "client_ids": sorted(payload_ids),
        "num_clients": len(payload_ids),
        "client_meta_path": str(client_meta_path),
        "test_multimodal_path": str(test_multimodal_path),
        "train_clients_dir": str(train_dir),
    }


def _audit_stage2_inputs(cfg: dict, stage2_dir: Path, stage1_audit: dict):
    if not stage2_dir.exists() or not stage2_dir.is_dir():
        raise FileNotFoundError(f"Missing Stage2 cluster results directory: {stage2_dir}")
    assignments_path = stage2_dir / "cluster_assignments.csv"
    metrics = _load_json(stage2_dir / "cluster_metrics.json", "Stage2 cluster_metrics.json")
    diagnostics = _load_json(stage2_dir / "adaptive_diagnostics.json", "Stage2 adaptive_diagnostics.json")
    _require_readable_file(stage2_dir / "fingerprints.npy", "Stage2 fingerprints.npy")
    stage2_config_path = stage2_dir / "stage2_only_config_used.yaml"
    stage2_config = _load_yaml(stage2_config_path, "Stage2 config snapshot") if stage2_config_path.exists() else None
    metadata_path = stage2_dir / "stage2_only_metadata.json"
    stage2_metadata = _load_json(metadata_path, "Stage2 metadata") if metadata_path.exists() else None
    encoder_dir = stage2_dir / "pretrained_encoders"
    if not encoder_dir.exists() or not encoder_dir.is_dir():
        raise FileNotFoundError(f"Missing Stage2 pretrained_encoders directory: {encoder_dir}")

    if metrics.get("discovery_status") != "discovery_success":
        raise ValueError(f"Stage2 discovery_status must be discovery_success, got {metrics.get('discovery_status')}")
    if metrics.get("method") not in {"adaptive_isodata", "adaptive"}:
        raise ValueError(f"Stage2 input must come from adaptive discovery, got method={metrics.get('method')}")

    cfg_dataset = dataset_result_name(cfg)
    if stage2_config:
        stage2_dataset = dataset_result_name(stage2_config)
        if stage2_dataset != cfg_dataset:
            raise ValueError(f"Stage2 dataset mismatch: config={cfg_dataset}, stage2={stage2_dataset}")

    rows = _read_csv_rows(assignments_path)
    if not rows:
        raise ValueError("Stage2 cluster_assignments.csv is empty.")
    if "pred_cluster" not in rows[0]:
        raise ValueError("Stage2 cluster_assignments.csv must contain pred_cluster.")
    assignment_ids = [row.get("client_id") for row in rows]
    if any(not client_id for client_id in assignment_ids):
        raise ValueError("Stage2 cluster_assignments.csv must contain non-empty client_id values.")
    if len(set(assignment_ids)) != len(assignment_ids):
        raise ValueError("Stage2 cluster_assignments.csv contains duplicate client_id values.")
    stage1_ids = set(stage1_audit["client_ids"])
    assignment_set = set(assignment_ids)
    if assignment_set != stage1_ids:
        missing = sorted(stage1_ids - assignment_set)
        unknown = sorted(assignment_set - stage1_ids)
        raise ValueError(f"Stage2 client IDs mismatch Stage1: missing={missing}, unknown={unknown}")

    clusters = []
    for row in rows:
        try:
            cluster_id = int(row["pred_cluster"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid pred_cluster for client {row.get('client_id')}: {row.get('pred_cluster')}") from exc
        if cluster_id < 0:
            raise ValueError(f"pred_cluster must be non-negative, got {cluster_id}")
        clusters.append(cluster_id)
    cluster_ids = sorted(set(clusters))
    estimated_q = int(metrics.get("estimated_Q", metrics.get("estimated_num_clusters", 0)))
    if estimated_q <= 0:
        raise ValueError(f"Stage2 estimated_Q must be positive, got {estimated_q}")
    if len(cluster_ids) != estimated_q:
        raise ValueError(f"Stage2 estimated_Q mismatch: unique_pred_clusters={len(cluster_ids)}, estimated_Q={estimated_q}")
    if diagnostics.get("estimated_Q") is not None and int(diagnostics["estimated_Q"]) != len(cluster_ids):
        raise ValueError("Stage2 adaptive diagnostics estimated_Q does not match cluster assignments.")

    fingerprints = np.load(stage2_dir / "fingerprints.npy")
    if int(fingerprints.shape[0]) != len(stage1_ids):
        raise ValueError(f"Stage2 fingerprints row count mismatch: {fingerprints.shape[0]} vs clients={len(stage1_ids)}")

    encoder_files = sorted(encoder_dir.glob("*_encoder.pt"))
    encoder_client_ids = [path.name[: -len("_encoder.pt")] for path in encoder_files]
    if set(encoder_client_ids) != stage1_ids:
        missing = sorted(stage1_ids - set(encoder_client_ids))
        unknown = sorted(set(encoder_client_ids) - stage1_ids)
        raise ValueError(f"Stage2 pretrained encoder IDs mismatch Stage1: missing={missing}, unknown={unknown}")
    for client_id in sorted(stage1_ids):
        _require_readable_file(encoder_dir / f"{client_id}_encoder.pt", f"Stage2 encoder for {client_id}")

    return {
        "cluster_assignments_path": str(assignments_path),
        "cluster_metrics_path": str(stage2_dir / "cluster_metrics.json"),
        "adaptive_diagnostics_path": str(stage2_dir / "adaptive_diagnostics.json"),
        "fingerprints_path": str(stage2_dir / "fingerprints.npy"),
        "pretrained_encoders_dir": str(encoder_dir),
        "client_ids": sorted(assignment_ids),
        "num_clients": len(assignment_ids),
        "cluster_ids": cluster_ids,
        "estimated_Q": estimated_q,
        "discovery_status": metrics.get("discovery_status"),
        "method": metrics.get("method"),
        "stage2_metadata": stage2_metadata,
    }


def _write_yaml(path: Path, data: dict):
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _write_json(path: Path, data: dict):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _formal_completion_status(metrics: dict | None, paths: dict):
    if not isinstance(metrics, dict):
        return "failed", "training_function_returned_non_dict_metrics"
    if not (paths["result_dir"] / "final_metrics.json").exists():
        return "failed", "missing_final_metrics_json"
    final_eval = metrics.get("final_eval")
    if not isinstance(final_eval, dict):
        return "failed", "missing_final_eval"
    if final_eval.get("eval_status") != "success":
        return "failed", final_eval.get("eval_failure_reason") or "final_evaluation_failed"
    for key in ["accuracy", "macro_f1", "loss"]:
        if not _is_finite_number(final_eval.get(key)):
            return "failed", f"invalid_final_eval_{key}"
    if int(metrics.get("effective_global_rounds", -1)) != int(metrics.get("total_global_rounds", -2)):
        return "failed", "incomplete_effective_global_rounds"
    return "success", None


def _metadata(args, cfg, paths, audit, status, failure_reason, start_time, end_time, metrics=None):
    stage2_audit = audit.get("stage2", {}) if audit else {}
    stage2_metadata = stage2_audit.get("stage2_metadata")
    return {
        "run_type": paths["run_type"],
        "tag": paths["tag"],
        "dataset": paths["dataset"],
        "status": status,
        "failure_reason": failure_reason,
        "original_config_path": str(_resolve(args.config)),
        "stage3_only_config_path": str(paths["config_snapshot"]),
        "stage1_dir": str(paths["stage1_dir"]),
        "stage2_dir": str(paths["stage2_dir"]),
        "output_root": str(paths["output_root"]),
        "run_dir": str(paths["run_dir"]),
        "result_dir": str(paths["result_dir"]),
        "model_dir": str(paths["model_dir"]),
        "git_branch": _git_branch(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "python_executable": sys.executable,
        "cli_arguments": vars(args),
        "start_time": start_time,
        "end_time": end_time,
        "runtime_seconds": None if end_time is None else float(time.time() - _metadata.start_monotonic),
        "seed": int(cfg.get("seed", 42)),
        "scheduler": cfg.get("training", {}).get("scheduler", "proposed_cluster_coverage"),
        "training_mode": "mmbind_fusion_split_learning",
        "binding_mode": cfg.get("binding", {}).get("type", "label_random"),
        "fusion_mode": cfg.get("fusion", {}).get("type", "concat_mlp"),
        "estimated_Q": stage2_audit.get("estimated_Q"),
        "stage2_discovery_status": stage2_audit.get("discovery_status"),
        "stage2_source_metadata": stage2_metadata,
        "stage2_adaptive_discovery_freeze_sha": None if not stage2_metadata else stage2_metadata.get("git_commit"),
        "metrics": metrics,
    }


_metadata.start_monotonic = 0.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Stage 3 only: train fusion Split Learning from frozen Stage1/Stage2 inputs.")
    parser.add_argument("--config", required=True, help="Path to yaml config")
    parser.add_argument("--stage1-dir", required=True, help="Frozen 01_dataset_partition directory")
    parser.add_argument("--stage2-dir", required=True, help="Frozen 02_cluster_results directory")
    parser.add_argument("--output-root", required=True, help="Root directory for isolated Stage3 outputs")
    parser.add_argument("--tag", required=True, help="Run tag under output-root/<dataset>/")
    parser.add_argument("--run-type", choices=["codex_test", "user_formal"], required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    run_cfg, paths = build_stage3_only_run(
        cfg,
        stage1_dir=args.stage1_dir,
        stage2_dir=args.stage2_dir,
        output_root=args.output_root,
        tag=args.tag,
        run_type=args.run_type,
    )
    audit = audit_stage3_inputs(run_cfg, paths["stage1_dir"], paths["stage2_dir"])

    paths["run_dir"].mkdir(parents=True, exist_ok=False)
    paths["result_dir"].mkdir(parents=True, exist_ok=False)
    paths["model_dir"].mkdir(parents=True, exist_ok=False)
    _write_yaml(paths["config_snapshot"], run_cfg)

    start = _utc_now()
    _metadata.start_monotonic = time.time()
    try:
        set_seed(int(run_cfg.get("seed", 42)))
        device = select_device(run_cfg.get("device", "auto"))
        metrics = run_mmbind_fusion_stage3_split_training(run_cfg, ROOT, device)
    except Exception as exc:
        end = _utc_now()
        _write_json(paths["metadata"], _metadata(args, run_cfg, paths, audit, "failed", str(exc), start, end))
        raise

    end = _utc_now()
    status, failure_reason = _formal_completion_status(metrics, paths)
    _write_json(paths["metadata"], _metadata(args, run_cfg, paths, audit, status, failure_reason, start, end, metrics=metrics))
    if status != "success":
        raise RuntimeError(f"Stage3-only run did not complete successfully: {failure_reason}")
    print("Stage 3 only finished.")
    print(f"run_type={paths['run_type']}")
    print(f"stage1_dir={paths['stage1_dir']}")
    print(f"stage2_dir={paths['stage2_dir']}")
    print(f"run_dir={paths['run_dir']}")
    print(f"final_metrics={metrics}")


if __name__ == "__main__":
    main()
