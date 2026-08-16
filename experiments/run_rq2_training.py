import argparse
import csv
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from MSL.discovery.clustering import run_kmeans
from MSL.learning.fusion_sl import run_mmbind_fusion_stage3_split_training
from MSL.utils.device import select_device
from MSL.utils.seed import set_seed
from experiments.common import (
    RQ2_METHODS,
    add_common_run_args,
    assert_no_mock_pipeline_imports,
    find_stage1_dir,
    find_stage2_dir,
    load_fingerprint_npz,
    project_root,
    resolved_cfg,
    runtime_metadata,
    write_json,
)


# 为正式 run 生成不覆盖旧结果的目录。
def unique_run_dir(base: Path) -> Path:
    if not base.exists():
        return base
    for attempt in range(2, 10000):
        candidate = base.with_name(f"{base.name}_attempt{attempt:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot allocate unique result directory under {base.parent}")


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
def prepare_method_topology(method: str, stage1_dir: Path, adaptive_dir: Path, results_root: Path, seed: int) -> tuple[Path, str]:
    if method in {"ours", "randomsl"}:
        return adaptive_dir, "pred_cluster"
    if method == "oracle":
        return adaptive_dir, "true_cluster"
    if method.startswith("kmeans"):
        k = int(method.replace("kmeans", ""))
        payload = load_fingerprint_npz(adaptive_dir)
        pred = run_kmeans(payload["fingerprints"], k, seed=seed)
        topology_dir = results_root / "rq2" / "topologies" / stage1_dir.parent.name / stage1_dir.name / method
        write_assignment_csv(topology_dir / "pred_cluster.csv", payload["client_ids"], pred, "pred_cluster")
        write_assignment_csv(topology_dir / "true_cluster.csv", payload["client_ids"], payload["true_cluster"], "true_cluster")
        link_pretrained_encoders(adaptive_dir, topology_dir)
        return topology_dir.resolve(), "pred_cluster"
    raise ValueError(f"Unsupported RQ2 method: {method}")


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


# 运行单个 RQ2 方法并保存最终 metrics。
def run_one(dataset: str, fold: int | None, seed: int, method: str, results_root: Path, device_name: str, global_rounds: int | None) -> dict:
    assert_no_mock_pipeline_imports()
    root = project_root()
    cfg = resolved_cfg(dataset, fold, seed)
    stage1_dir = find_stage1_dir(root, cfg)
    adaptive_dir = find_stage2_dir(root, stage1_dir, "adaptive_isodata")
    cluster_dir, assignment_source = prepare_method_topology(method, stage1_dir, adaptive_dir, results_root, seed)
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{dataset}_fold{fold or 0:02d}_seed{seed}_{method}_{run_stamp}"
    run_dir = unique_run_dir(results_root / "rq2" / "raw" / run_name)
    ckpt_dir = results_root / "rq2" / "checkpoints" / run_dir.name
    cfg = configure_method(cfg, method, stage1_dir, cluster_dir, assignment_source, run_dir, ckpt_dir, global_rounds)
    metadata = runtime_metadata(root, dataset, fold, seed, method)
    set_seed(seed)
    device = select_device(device_name)
    try:
        metrics = run_mmbind_fusion_stage3_split_training(cfg, root, device)
        payload = {**metadata, "status": "success", "metrics": metrics, "run_dir": str(run_dir), "checkpoint_dir": str(ckpt_dir)}
    except Exception as exc:
        payload = {
            **metadata,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "run_dir": str(run_dir),
            "checkpoint_dir": str(ckpt_dir),
        }
        write_json(run_dir / "failed_run.json", payload)
        return payload
    write_json(run_dir / "rq2_result.json", payload)
    return payload


# 解析 RQ2 单次训练参数。
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run one RQ2 method with the shared Split Learning trainer.")
    add_common_run_args(parser)
    parser.add_argument("--method", choices=RQ2_METHODS, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--global-rounds", type=int)
    return parser.parse_args(argv)


# 执行 RQ2 单次训练。
def main(argv=None):
    args = parse_args(argv)
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
