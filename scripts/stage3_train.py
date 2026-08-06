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

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from MSL.learning.fusion_sl import run_mmbind_fusion_stage3_split_training
from MSL.evaluation.plot_training_curves import write_training_curves
from MSL.utils.config import load_config, normalize_experiment_config, save_config_artifacts
from MSL.utils.device import select_device
from MSL.utils.results import (
    cluster_assignment_scope,
    dataset_result_name,
    experiment_config_signature,
    resolve_stage_paths,
)
from MSL.utils.seed import set_seed


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


def build_stage3_run(cfg: dict, stage1_dir, stage2_dir, output_root, attempt=1):
    if not output_root:
        raise ValueError("--output-root is required.")

    stage1_path = _resolve(stage1_dir)
    stage2_path = _resolve(stage2_dir)
    output_root_path = _resolve(output_root)

    dataset_name = _validate_path_component(dataset_result_name(cfg), "dataset")
    scope = _validate_path_component(cluster_assignment_scope(cfg), "cluster assignment scope")
    objective = str(cfg.get("fusion", {}).get("training_objective", "objective"))
    loss_component = _validate_path_component(objective, "fusion training objective")
    config_signature = experiment_config_signature(cfg)
    seed = int(cfg.get("seed", 42))
    attempt = int(attempt)
    if attempt <= 0:
        raise ValueError(f"attempt must be a positive integer, got {attempt}.")
    seed_component = _validate_path_component(f"seed-{seed}", "seed")
    attempt_component = _validate_path_component(f"attempt-{attempt:02d}", "attempt")
    for input_label, input_path in [("Stage1", stage1_path), ("Stage2", stage2_path)]:
        if _paths_overlap(output_root_path, input_path):
            raise ValueError(f"output_root must not overlap {input_label} input directory: {input_path}")

    run_dir = (
        output_root_path
        / scope
        / dataset_name
        / loss_component
        / attempt_component
        / seed_component
    ).resolve()
    if not _is_relative_to(run_dir, output_root_path):
        raise ValueError(f"Stage3 run directory escaped output_root: {run_dir}")
    for input_label, input_path in [("Stage1", stage1_path), ("Stage2", stage2_path)]:
        if _paths_overlap(run_dir, input_path):
            raise ValueError(f"Stage3 run directory must not overlap {input_label} input directory: {input_path}")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing Stage3 run directory: {run_dir}")

    run_cfg = dict(cfg)
    run_cfg["partition"] = {**run_cfg.get("partition", {}), "output_dir": str(stage1_path)}
    run_cfg["cluster"] = {**run_cfg.get("cluster", {}), "output_dir": str(stage2_path)}
    run_cfg["result"] = {**run_cfg.get("result", {}), "output_dir": str(run_dir)}
    run_cfg["result_model"] = {**run_cfg.get("result_model", {}), "output_dir": str(run_dir)}
    run_cfg["stage3"] = {
        "stage1_dir": str(stage1_path),
        "stage2_dir": str(stage2_path),
        "output_root": str(output_root_path),
        "run_dir": str(run_dir),
        "cluster_assignment_scope": scope,
        "config_signature": config_signature,
        "loss": objective,
        "seed": seed,
        "attempt": attempt,
    }
    paths = {
        "stage1_dir": stage1_path,
        "stage2_dir": stage2_path,
        "output_root": output_root_path,
        "run_dir": run_dir,
        "metadata": run_dir / "stage3_metadata.json",
        "run_id": attempt_component,
        "dataset": dataset_name,
        "cluster_assignment_scope": scope,
        "config_signature": config_signature,
        "loss": objective,
        "seed": seed,
        "attempt": attempt,
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
        raise FileNotFoundError(f"Missing Stage2 cluster directory: {stage2_dir}")
    pred_path = stage2_dir / "pred_cluster.csv"
    true_path = stage2_dir / "true_cluster.csv"
    assignment_source = str(
        cfg.get("training", {}).get("cluster_assignment_source", "pred_cluster")
    ).strip().lower()
    if assignment_source not in {"pred_cluster", "true_cluster"}:
        raise ValueError(
            "training.cluster_assignment_source must be 'pred_cluster' or 'true_cluster', "
            f"got {assignment_source!r}."
        )
    assignment_path = pred_path if assignment_source == "pred_cluster" else true_path
    assignment_column = assignment_source
    metadata_path = stage2_dir / "stage2_metadata.json"
    _require_readable_file(assignment_path, f"Stage2 {assignment_path.name}")
    stage2_metadata = None
    stage2_metadata_read_error = None
    if metadata_path.exists():
        try:
            stage2_metadata = _load_json(metadata_path, "Stage2 stage2_metadata.json")
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            stage2_metadata_read_error = f"{type(exc).__name__}: {exc}"
    encoder_dir = stage2_dir / "pretrained_encoders"
    if not encoder_dir.exists() or not encoder_dir.is_dir():
        raise FileNotFoundError(f"Missing Stage2 pretrained_encoders directory: {encoder_dir}")

    raw_metrics = stage2_metadata.get("metrics", {}) if isinstance(stage2_metadata, dict) else {}
    metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
    method = (
        stage2_metadata.get("cluster_method") or metrics.get("method")
        if isinstance(stage2_metadata, dict)
        else None
    )
    discovery_status = metrics.get("discovery_status")
    raw_reported_estimated_q = metrics.get("estimated_Q", metrics.get("estimated_num_clusters"))
    try:
        reported_estimated_q = (
            None if raw_reported_estimated_q is None else int(raw_reported_estimated_q)
        )
    except (TypeError, ValueError):
        reported_estimated_q = None

    rows = _read_csv_rows(assignment_path)
    if not rows:
        raise ValueError(f"Stage2 {assignment_path.name} is empty.")
    if assignment_column not in rows[0]:
        raise ValueError(f"Stage2 {assignment_path.name} must contain {assignment_column}.")
    assignment_ids = [row.get("client_id") for row in rows]
    if any(not client_id for client_id in assignment_ids):
        raise ValueError(f"Stage2 {assignment_path.name} must contain non-empty client_id values.")
    if len(set(assignment_ids)) != len(assignment_ids):
        raise ValueError(f"Stage2 {assignment_path.name} contains duplicate client_id values.")
    stage1_ids = set(stage1_audit["client_ids"])
    assignment_set = set(assignment_ids)
    if assignment_set != stage1_ids:
        missing = sorted(stage1_ids - assignment_set)
        unknown = sorted(assignment_set - stage1_ids)
        raise ValueError(f"Stage2 client IDs mismatch Stage1: missing={missing}, unknown={unknown}")

    clusters = []
    for row in rows:
        try:
            cluster_id = int(row[assignment_column])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid {assignment_column} for client {row.get('client_id')}: {row.get(assignment_column)}"
            ) from exc
        if cluster_id < 0:
            raise ValueError(f"{assignment_column} must be non-negative, got {cluster_id}")
        clusters.append(cluster_id)
    cluster_ids = sorted(set(clusters))
    estimated_q = len(cluster_ids)

    encoder_files = sorted(encoder_dir.glob("*_encoder.pt"))
    encoder_client_ids = [path.name[: -len("_encoder.pt")] for path in encoder_files]
    if set(encoder_client_ids) != stage1_ids:
        missing = sorted(stage1_ids - set(encoder_client_ids))
        unknown = sorted(set(encoder_client_ids) - stage1_ids)
        raise ValueError(f"Stage2 pretrained encoder IDs mismatch Stage1: missing={missing}, unknown={unknown}")
    for client_id in sorted(stage1_ids):
        _require_readable_file(encoder_dir / f"{client_id}_encoder.pt", f"Stage2 encoder for {client_id}")

    true_cluster_audit = {
        "available": bool(true_path.exists()),
        "path": str(true_path) if true_path.exists() else None,
        "client_ids_match_stage1": None,
        "num_rows": None,
        "read_error": None,
    }
    if true_path.exists():
        try:
            true_rows = _read_csv_rows(true_path)
            true_ids = [row.get("client_id") for row in true_rows]
            true_cluster_audit.update(
                {
                    "client_ids_match_stage1": bool(
                        len(true_ids) == len(set(true_ids))
                        and set(true_ids) == stage1_ids
                    ),
                    "num_rows": len(true_rows),
                }
            )
        except (OSError, UnicodeError, csv.Error) as exc:
            true_cluster_audit["read_error"] = f"{type(exc).__name__}: {exc}"

    return {
        "cluster_assignment_source": assignment_source,
        "cluster_assignment_path": str(assignment_path),
        "cluster_assignment_column": assignment_column,
        "pred_cluster_path": str(pred_path),
        "true_cluster_path": str(true_path) if true_path.exists() else None,
        "true_cluster_audit": true_cluster_audit,
        "metadata_path": str(metadata_path) if metadata_path.exists() else None,
        "metadata_read_error": stage2_metadata_read_error,
        "pretrained_encoders_dir": str(encoder_dir),
        "client_ids": sorted(assignment_ids),
        "num_clients": len(assignment_ids),
        "cluster_ids": cluster_ids,
        "estimated_Q": estimated_q,
        "reported_estimated_Q": reported_estimated_q,
        "reported_estimated_Q_matches_pred_cluster": (
            None
            if reported_estimated_q is None
            else reported_estimated_q == estimated_q
        ),
        "discovery_status": discovery_status,
        "method": method,
        "stage2_metadata": stage2_metadata,
    }


def _write_json(path: Path, data: dict):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _formal_completion_status(metrics: dict | None, paths: dict):
    if not isinstance(metrics, dict):
        return "failed", "training_function_returned_non_dict_metrics"
    run_dir = paths["run_dir"]
    required_files = [
        "source_config.config",
        "resolved_config.config",
        "train_log.csv",
        "final_metrics.json",
        "last_model.pt",
        "training_curves.png",
    ]
    for name in required_files:
        if not (run_dir / name).exists():
            return "failed", f"missing_{name}"
    run_test = bool(metrics.get("evaluation_mode") != "test_deferred")
    if not run_test:
        if metrics.get("test_eval_status") != "deferred":
            return "failed", "deferred_run_must_defer_test"
        if int(metrics.get("test_evaluation_count", -1)) != 0:
            return "failed", "deferred_test_evaluation_count_must_equal_zero"
        if metrics.get("official_result") is not None:
            return "failed", "deferred_run_must_not_claim_official_result"
        if not _is_finite_number(metrics.get("best_round")):
            return "failed", "missing_best_round"
        return "success", None
    if metrics.get("test_eval_status") != "success":
        return "failed", metrics.get("test_eval_failure_reason") or "test_evaluation_failed"
    for key in ["test_accuracy", "test_macro_f1", "test_weighted_f1", "test_loss"]:
        if not _is_finite_number(metrics.get(key)):
            return "failed", f"invalid_{key}"
    if metrics.get("checkpoint") != "last_model.pt":
        return "failed", "official_checkpoint_must_be_last_model"
    if metrics.get("selected_by") != "fixed_rounds_no_validation":
        return "failed", "official_checkpoint_selection_must_be_fixed_rounds"
    if int(metrics.get("test_evaluation_count", -1)) != 1:
        return "failed", "test_evaluation_count_must_equal_one"
    executed_rounds = int(metrics.get("executed_global_rounds", -1))
    configured_rounds = int(metrics.get("configured_global_rounds", -1))
    if executed_rounds <= 0 or configured_rounds <= 0 or executed_rounds > configured_rounds:
        return "failed", "invalid_executed_global_rounds"
    if not _is_finite_number(metrics.get("best_round")):
        return "failed", "missing_best_round"
    return "success", None


def _metadata(args, cfg, paths, audit, status, failure_reason, start_time, end_time, metrics=None):
    stage2_audit = audit.get("stage2", {}) if audit else {}
    stage2_metadata = stage2_audit.get("stage2_metadata")
    training_cfg = cfg.get("training", {})
    metrics = metrics if isinstance(metrics, dict) else None
    return {
        "stage": "stage3_train",
        "run_id": paths["run_id"],
        "cluster_assignment_scope": paths["cluster_assignment_scope"],
        "config_signature": paths["config_signature"],
        "attempt": paths["attempt"],
        "dataset": paths["dataset"],
        "status": status,
        "failure_reason": failure_reason,
        "original_config_path": str(_resolve(args.config)),
        "stage1_dir": str(paths["stage1_dir"]),
        "stage2_dir": str(paths["stage2_dir"]),
        "output_root": str(paths["output_root"]),
        "run_dir": str(paths["run_dir"]),
        "git_branch": _git_branch(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "python_executable": sys.executable,
        "cli_arguments": vars(args),
        "start_time": start_time,
        "end_time": end_time,
        "runtime_seconds": None if end_time is None else float(time.time() - _metadata.start_monotonic),
        "seed": int(cfg.get("seed", 42)),
        "scheduler": cfg.get("training", {}).get("scheduler", "balanced_cluster_round_robin"),
        "training_mode": "mmbind_fusion_split_learning",
        "binding_mode": cfg.get("binding", {}).get("type", "label_random"),
        "fusion_mode": cfg.get("fusion", {}).get("type", "concat_mlp"),
        "fusion_training_objective": cfg.get("fusion", {}).get(
            "training_objective",
            "label_random_ce",
        ),
        "mmbind_training_config": cfg.get("fusion", {}).get("mmbind", {}),
        "split_protocol": cfg.get("dataset", {}).get("split_protocol"),
        "split_subjects": {
            split_name: cfg.get("dataset", {}).get(f"{split_name}_subjects")
            for split_name in ("train", "test")
        },
        "evaluation_mode": (
            "formal_test"
            if bool(cfg.get("evaluation", {}).get("run_test", True))
            else "test_deferred"
        ),
        "configured_global_rounds": int(training_cfg.get("global_rounds", 0)),
        "executed_global_rounds": None if metrics is None else metrics.get("executed_global_rounds"),
        "best_round": None if metrics is None else metrics.get("best_round"),
        "stop_round": None if metrics is None else metrics.get("stop_round"),
        "stop_reason": None if metrics is None else metrics.get("stop_reason"),
        "checkpoint_selection": "fixed_rounds_no_validation",
        "test_evaluation_count": None if metrics is None else metrics.get("test_evaluation_count"),
        "device": None if metrics is None else metrics.get("device"),
        "estimated_Q": stage2_audit.get("estimated_Q"),
        "stage2_discovery_status": stage2_audit.get("discovery_status"),
        "stage2_git_commit": None if not isinstance(stage2_metadata, dict) else stage2_metadata.get("git_commit"),
        "stage2_source_metadata": stage2_metadata,
        "input_audit": audit,
        "config_snapshot": cfg,
        "metrics": metrics,
    }


_metadata.start_monotonic = 0.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Stage 3: train fusion Split Learning from frozen Stage1/Stage2 inputs.")
    parser.add_argument("--config", required=True, help="Path to INI-style .config file")
    parser.add_argument(
        "--seed",
        type=int,
        help="Override the Stage3 experiment seed in memory; does not modify the source .config or affect Stage1/Stage2",
    )
    parser.add_argument(
        "--fusion-training-objective",
        choices=["label_random_ce", "mmbind_weighted_contrastive"],
        help="Override fusion.training_objective in memory and record it in resolved_config.config",
    )
    parser.add_argument("--stage1-dir", help="Optional override for stage3.stage1_dir")
    parser.add_argument("--stage2-dir", help="Optional override for stage3.stage2_dir")
    parser.add_argument("--output-root", help="Optional override for stage3.output_root")
    parser.add_argument(
        "--attempt",
        type=int,
        help="Optional override for stage3.attempt; the run fails instead of overwriting an existing attempt",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = normalize_experiment_config(load_config(args.config))
    stage3_cfg = cfg.get("stage3", {})
    stage1_dir = args.stage1_dir or stage3_cfg.get("stage1_dir")
    stage2_dir = args.stage2_dir or stage3_cfg.get("stage2_dir")
    if not stage1_dir or not stage2_dir:
        # 新格式 config 不写路径：自动从 base_dir + 数据集 + 协议生成。
        resolved = resolve_stage_paths(cfg, ROOT)
        stage1_dir = stage1_dir or resolved["stage1_dir"]
        stage2_dir = stage2_dir or resolved["stage2_dir"]
    output_root = args.output_root or stage3_cfg.get("output_root")
    if not output_root:
        output_root = resolve_stage_paths(cfg, ROOT)["output_dir"]
    if not stage1_dir or not stage2_dir:
        raise ValueError(
            "Set stage3.stage1_dir and stage3.stage2_dir in the .config file or pass CLI overrides."
        )
    attempt = args.attempt if args.attempt is not None else int(stage3_cfg.get("attempt", 1))
    resolved_seed = int(args.seed) if args.seed is not None else int(cfg.get("seed", 42))
    cfg = {**cfg, "seed": resolved_seed}
    if args.fusion_training_objective is not None:
        cfg["fusion"] = {
            **cfg.get("fusion", {}),
            "training_objective": args.fusion_training_objective,
        }
    # attempt 自动递增：同 loss 目录已存在时自动尝试下一个 attempt，避免覆盖旧结果。
    run_cfg = None
    paths = None
    for candidate_attempt in range(attempt, attempt + 100):
        try:
            run_cfg, paths = build_stage3_run(
                cfg,
                stage1_dir=stage1_dir,
                stage2_dir=stage2_dir,
                output_root=output_root,
                attempt=candidate_attempt,
            )
            attempt = candidate_attempt
            break
        except FileExistsError:
            continue
    if run_cfg is None or paths is None:
        raise FileExistsError(f"Too many existing Stage3 attempt directories under {output_root}.")
    audit = audit_stage3_inputs(run_cfg, paths["stage1_dir"], paths["stage2_dir"])

    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    save_config_artifacts(args.config, run_cfg, paths["run_dir"])

    start = _utc_now()
    _metadata.start_monotonic = time.time()
    try:
        set_seed(int(run_cfg.get("seed", 42)))
        device = select_device(run_cfg.get("device", "auto"))
        metrics = run_mmbind_fusion_stage3_split_training(run_cfg, ROOT, device)
        write_training_curves(paths["run_dir"])
    except Exception as exc:
        end = _utc_now()
        _write_json(paths["metadata"], _metadata(args, run_cfg, paths, audit, "failed", str(exc), start, end))
        raise

    end = _utc_now()
    status, failure_reason = _formal_completion_status(metrics, paths)
    _write_json(paths["metadata"], _metadata(args, run_cfg, paths, audit, status, failure_reason, start, end, metrics=metrics))
    if status != "success":
        raise RuntimeError(f"Stage3 run did not complete successfully: {failure_reason}")
    print("Stage 3 finished.")
    print(f"stage1_dir={paths['stage1_dir']}")
    print(f"stage2_dir={paths['stage2_dir']}")
    print(f"run_dir={paths['run_dir']}")
    print(f"final_metrics={metrics}")


if __name__ == "__main__":
    main()
