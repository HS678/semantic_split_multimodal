import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from MSL.discovery.clustering import run_kmeans
from MSL.learning.cluster_feasibility import (
    repair_cluster_feasibility,
    validate_cluster_feasibility,
)
from MSL.learning.fusion_sl import run_mmbind_fusion_stage3_split_training
from MSL.utils.device import select_device
from MSL.utils.seed import set_seed
from experiments.common import (
    RQ2_METHODS,
    add_common_run_args,
    assert_no_mock_pipeline_imports,
    find_stage1_dir,
    find_stage2_dir,
    formal_result_dir,
    load_fingerprint_npz,
    project_root,
    resolved_cfg,
    runtime_metadata,
    stable_config_hash,
    write_json,
)


# 构建 RQ2 run 的 deterministic key，保留给 metadata 和兼容调用使用。
def rq2_run_key(dataset: str, fold: int | None, seed: int, method: str, global_rounds: int | None) -> str:
    round_part = "formal" if global_rounds is None else f"rounds{int(global_rounds)}"
    return f"{dataset}_fold{fold or 0:02d}_seed{int(seed)}_{method}_{round_part}"


# 返回层级化的 RQ2 运行目录。
def rq2_run_dir(results_root: Path, dataset: str, fold: int | None, seed: int, method: str, global_rounds: int | None) -> Path:
    run_dir = formal_result_dir(results_root, "rq2", dataset, method, fold, seed)
    if global_rounds is not None:
        run_dir = run_dir / f"rounds_{int(global_rounds)}"
    return run_dir


# 写入 KMeans 或 oracle topology assignment 文件。
def write_assignment_csv(path: Path, client_ids, labels, column: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["client_id", column])
        writer.writeheader()
        for client_id, label in zip(client_ids, labels):
            writer.writerow({"client_id": str(client_id), column: int(label)})


# 将 adaptive Stage2 的 pretrained encoders 复用到派生 topology 目录。
def link_pretrained_encoders(adaptive_dir: Path, target_dir: Path) -> None:
    source = adaptive_dir / "pretrained_encoders"
    target = target_dir / "pretrained_encoders"
    if target.exists():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source, target, target_is_directory=True)
    except OSError:
        shutil.copytree(source, target)


# 为 RQ2 方法解析 cluster_dir 和 assignment source。
def prepare_method_topology(method: str, stage1_dir: Path, adaptive_dir: Path, topology_dir: Path, seed: int, r: int) -> tuple[Path, str, dict]:
    payload = load_fingerprint_npz(adaptive_dir)
    client_ids = payload["client_ids"]
    if method == "randomsl":
        raw = payload["pred_cluster"].astype(int)
        return adaptive_dir, "pred_cluster", {
            "feasibility_checked": False,
            "feasibility_repair_applied": None,
            "num_reassigned_clients": None,
            "cluster_sizes_before": None,
            "cluster_sizes_after": None,
            "violating_clusters_before": None,
            "raw_cluster_assignment": {str(client_id): int(label) for client_id, label in zip(client_ids, raw)},
            "training_cluster_assignment": None,
            "yield_definition": normalized_yield_definition(),
        }
    if method == "ours":
        raw = payload["pred_cluster"].astype(int)
        return _prepare_cluster_based_topology(
            method,
            stage1_dir,
            adaptive_dir,
            topology_dir,
            payload,
            raw,
            "pred_cluster",
            r,
            allow_repair=True,
        )
    if method == "oracle":
        true = payload["true_cluster"].astype(int)
        report = validate_cluster_feasibility(payload["fingerprints"], true, r)
        metadata = {
            "feasibility_checked": True,
            "feasibility_repair_applied": False,
            "num_reassigned_clients": 0,
            "cluster_sizes_before": {str(k): int(v) for k, v in report.cluster_sizes.items()},
            "cluster_sizes_after": {str(k): int(v) for k, v in report.cluster_sizes.items()},
            "violating_clusters_before": [int(v) for v in report.violating_clusters],
            "raw_cluster_assignment": {str(client_id): int(label) for client_id, label in zip(client_ids, true)},
            "training_cluster_assignment": {str(client_id): int(label) for client_id, label in zip(client_ids, true)},
            "yield_definition": normalized_yield_definition(),
        }
        return adaptive_dir, "true_cluster", metadata
    if method.startswith("kmeans"):
        k = int(method.replace("kmeans", ""))
        pred = run_kmeans(payload["fingerprints"], k, seed=seed)
        return _prepare_cluster_based_topology(
            method,
            stage1_dir,
            adaptive_dir,
            topology_dir,
            payload,
            pred,
            "pred_cluster",
            r,
            allow_repair=True,
        )
    raise ValueError(f"Unsupported RQ2 method: {method}")


# 为 cluster-based RQ2 方法生成 raw/training assignment 和 repair metadata。
def _prepare_cluster_based_topology(method, stage1_dir, adaptive_dir, topology_dir, payload, raw_assignment, assignment_source, r, allow_repair):
    topology_dir = Path(topology_dir)
    client_ids = payload["client_ids"]
    result = repair_cluster_feasibility(
        payload["fingerprints"],
        raw_assignment,
        r=int(r),
        client_ids=client_ids,
    ) if allow_repair else None
    if result is None:
        report = validate_cluster_feasibility(payload["fingerprints"], raw_assignment, r)
        training_assignment = raw_assignment
        metadata = {
            "feasibility_checked": True,
            "feasibility_repair_applied": False,
            "num_reassigned_clients": 0,
            "cluster_sizes_before": {str(k): int(v) for k, v in report.cluster_sizes.items()},
            "cluster_sizes_after": {str(k): int(v) for k, v in report.cluster_sizes.items()},
            "violating_clusters_before": [int(v) for v in report.violating_clusters],
        }
    else:
        training_assignment = result.training_assignment
        metadata = result.to_metadata()
    write_assignment_csv(topology_dir / "raw_cluster_assignment.csv", client_ids, raw_assignment, "raw_cluster")
    write_assignment_csv(topology_dir / "pred_cluster.csv", client_ids, training_assignment, "pred_cluster")
    write_assignment_csv(topology_dir / "true_cluster.csv", client_ids, payload["true_cluster"], "true_cluster")
    link_pretrained_encoders(adaptive_dir, topology_dir)
    metadata.update(
        {
            "raw_cluster_assignment": {str(client_id): int(label) for client_id, label in zip(client_ids, raw_assignment)},
            "training_cluster_assignment": {str(client_id): int(label) for client_id, label in zip(client_ids, training_assignment)},
            "topology_dir": str(topology_dir.resolve()),
            "assignment_source": assignment_source,
            "yield_definition": normalized_yield_definition(),
        }
    )
    write_json(topology_dir / "feasibility_metadata.json", metadata)
    return topology_dir.resolve(), assignment_source, metadata


# 将 RQ2 方法转换为共享 trainer 的 policy 配置。
def configure_method(cfg: dict, method: str, stage1_dir: Path, cluster_dir: Path, assignment_source: str, run_dir: Path, ckpt_dir: Path, global_rounds: int | None) -> dict:
    cfg = dict(cfg)
    cfg["partition"] = {**dict(cfg.get("partition", {})), "output_dir": str(stage1_dir)}
    cfg["cluster"] = {**dict(cfg.get("cluster", {})), "output_dir": str(cluster_dir)}
    cfg["training"] = {
        **dict(cfg.get("training", {})),
        "cluster_assignment_source": assignment_source,
        "scheduler": "random" if method == "randomsl" else "balanced_cluster_round_robin",
    }
    if global_rounds is not None:
        cfg["training"]["global_rounds"] = int(global_rounds)
    cfg["result"] = {**dict(cfg.get("result", {})), "output_dir": str(run_dir)}
    cfg["result_model"] = {**dict(cfg.get("result_model", {})), "output_dir": str(ckpt_dir)}
    cfg["rq2_method"] = method
    return cfg


# 返回 normalized pseudo yield 的固定定义。
def normalized_yield_definition() -> dict:
    return {
        "name": "pseudo_samples_over_requested_tuple_budget",
        "formula": "pseudo_batch_size_per_round / (binding.batch_size * attempted_local_steps)",
        "numerator": "actual complete same-label pseudo multimodal tuples built in the round",
        "denominator": "configured requested pseudo tuple budget for that round",
    }


# 从训练日志提取每轮正式结果字段。
def summarize_train_log(path: Path, binding_batch_size: int) -> dict:
    rows = []
    if not path.exists():
        return {
            "selected_client_count_per_round": [],
            "selected_cluster_counts_per_round": [],
            "coverage_per_round": [],
            "empty_cluster_count_per_round": [],
            "candidate_labels_per_round": [],
            "pseudo_batch_size_per_round": [],
            "normalized_pseudo_yield_per_round": [],
            "empty_binding_per_round": [],
            "training_loss_per_round": [],
            "loss_mean": None,
            "loss_std": None,
            "loss_variance": None,
        }
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    losses = [float(row["loss"]) for row in rows if row.get("loss") not in {None, ""}]
    selected_counts = [int(float(row.get("K_t") or row.get("clients_per_round") or 0)) for row in rows]
    per_cluster = [json.loads(row.get("per_cluster_selected_json") or "{}") for row in rows]
    expected_clusters = [json.loads(row.get("expected_cluster_ids") or "[]") for row in rows]
    attempted_steps = [int(float(row.get("attempted_local_steps") or 1)) for row in rows]
    pseudo_sizes = [int(float(row.get("pseudo_batch_size") or 0)) for row in rows]
    return {
        "selected_client_count_per_round": selected_counts,
        "selected_cluster_counts_per_round": per_cluster,
        "coverage_per_round": [float(row.get("coverage") or 0.0) for row in rows],
        "empty_cluster_count_per_round": [
            int(sum(1 for cluster_id in expected if int(counts.get(str(cluster_id), 0)) == 0))
            for expected, counts in zip(expected_clusters, per_cluster)
        ],
        "candidate_labels_per_round": [json.loads(row.get("common_labels_json") or "[]") for row in rows],
        "pseudo_batch_size_per_round": pseudo_sizes,
        "normalized_pseudo_yield_per_round": [
            float(pseudo / max(1, int(binding_batch_size) * int(steps)))
            for pseudo, steps in zip(pseudo_sizes, attempted_steps)
        ],
        "empty_binding_per_round": [bool(int(float(row.get("empty_binding_round") or 0))) for row in rows],
        "training_loss_per_round": losses,
        "loss_mean": float(np_mean(losses)) if losses else None,
        "loss_std": float(np_std(losses)) if len(losses) > 1 else 0.0,
        "loss_variance": float(np_var(losses)) if len(losses) > 1 else 0.0,
    }


# 计算均值，避免额外引入 pandas 依赖。
def np_mean(values):
    return sum(values) / max(1, len(values))


# 计算总体标准差，基于原始 round loss。
def np_std(values):
    mean = np_mean(values)
    return (sum((value - mean) ** 2 for value in values) / max(1, len(values))) ** 0.5


# 计算总体方差，基于原始 round loss。
def np_var(values):
    mean = np_mean(values)
    return sum((value - mean) ** 2 for value in values) / max(1, len(values))


# 运行单个 RQ2 方法并保存最终 metrics。
def expected_rq2_config_hash(dataset: str, fold: int | None, seed: int, method: str, results_root: Path, global_rounds: int | None) -> str:
    root = project_root()
    cfg = resolved_cfg(dataset, fold, seed)
    stage1_dir = find_stage1_dir(root, cfg)
    adaptive_dir = find_stage2_dir(root, stage1_dir, "adaptive_isodata")
    r = int(cfg.get("training", {}).get("clients_per_cluster_per_round", 2))
    run_dir = rq2_run_dir(results_root, dataset, fold, seed, method, global_rounds)
    topology_dir = run_dir / "topology"
    cluster_dir, assignment_source, _ = prepare_method_topology(method, stage1_dir, adaptive_dir, topology_dir, seed, r)
    ckpt_dir = run_dir / "checkpoints"
    cfg = configure_method(cfg, method, stage1_dir, cluster_dir, assignment_source, run_dir, ckpt_dir, global_rounds)
    return stable_config_hash(
        {
            "dataset": dataset,
            "fold": fold,
            "seed": int(seed),
            "method": method,
            "global_rounds": cfg.get("training", {}).get("global_rounds"),
            "r": r,
            "cfg": cfg,
        }
    )


# 运行单个 RQ2 方法并保存最终 metrics。
def run_one(dataset: str, fold: int | None, seed: int, method: str, results_root: Path, device_name: str, global_rounds: int | None) -> dict:
    assert_no_mock_pipeline_imports()
    root = project_root()
    cfg = resolved_cfg(dataset, fold, seed)
    stage1_dir = find_stage1_dir(root, cfg)
    adaptive_dir = find_stage2_dir(root, stage1_dir, "adaptive_isodata")
    r = int(cfg.get("training", {}).get("clients_per_cluster_per_round", 2))
    run_name = rq2_run_key(dataset, fold, seed, method, global_rounds)
    run_dir = rq2_run_dir(results_root, dataset, fold, seed, method, global_rounds)
    topology_dir = run_dir / "topology"
    cluster_dir, assignment_source, feasibility_metadata = prepare_method_topology(method, stage1_dir, adaptive_dir, topology_dir, seed, r)
    ckpt_dir = run_dir / "checkpoints"
    cfg = configure_method(cfg, method, stage1_dir, cluster_dir, assignment_source, run_dir, ckpt_dir, global_rounds)
    config_hash = stable_config_hash(
        {
            "dataset": dataset,
            "fold": fold,
            "seed": int(seed),
            "method": method,
            "global_rounds": cfg.get("training", {}).get("global_rounds"),
            "r": r,
            "cfg": cfg,
        }
    )
    metadata = runtime_metadata(root, dataset, fold, seed, method)
    set_seed(seed)
    device = select_device(device_name)
    start = time.time()
    try:
        metrics = run_mmbind_fusion_stage3_split_training(cfg, root, device)
        round_summary = summarize_train_log(
            run_dir / "train_log.csv",
            int(cfg.get("binding", {}).get("batch_size", cfg.get("training", {}).get("batch_size", 1))),
        )
        payload = {
            **metadata,
            "status": "success",
            "run_key": run_name,
            "fingerprint_type": cfg.get("fingerprint", {}).get("type"),
            "r": int(r),
            "Q_hat": int(metrics.get("estimated_num_clusters", 0)),
            **feasibility_metadata,
            **round_summary,
            "accuracy": metrics.get("test_accuracy"),
            "macro_f1": metrics.get("test_macro_f1"),
            "runtime_seconds": float(time.time() - start),
            "config_hash": config_hash,
            "device": str(device),
            "config_snapshot": cfg,
            "metrics": metrics,
            "run_dir": str(run_dir),
            "checkpoint_dir": str(ckpt_dir),
            "curve_path": str(run_dir / "train_log.csv"),
            "cluster_metadata_path": str(Path(cluster_dir) / "feasibility_metadata.json"),
        }
    except Exception as exc:
        payload = {
            **metadata,
            "status": "failed",
            "run_key": run_name,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            **feasibility_metadata,
            "config_hash": config_hash,
            "runtime_seconds": float(time.time() - start),
            "run_dir": str(run_dir),
            "checkpoint_dir": str(ckpt_dir),
        }
        write_json(run_dir / "failed_run.json", payload)
        return payload
    write_json(run_dir / "rq2_result.json", payload)
    return payload


# 根据用户要求检查 CUDA 是否可用。
def require_cuda_if_requested(require_cuda: bool) -> None:
    if bool(require_cuda) and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by --require-cuda, but torch.cuda.is_available() is false.")


# 解析 RQ2 单次训练参数。
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run one RQ2 method with the shared Split Learning trainer.")
    add_common_run_args(parser)
    parser.add_argument("--method", choices=RQ2_METHODS, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--global-rounds", type=int)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args(argv)


# 执行 RQ2 单次训练。
def main(argv=None):
    args = parse_args(argv)
    require_cuda_if_requested(args.require_cuda)
    result = run_one(
        args.dataset,
        args.fold,
        int(args.seed),
        args.method,
        (ROOT / args.results_root).resolve(),
        args.device,
        args.global_rounds,
    )
    print(f"RQ2 {args.method} finished: status={result['status']} run_dir={result['run_dir']}")


if __name__ == "__main__":
    main()
