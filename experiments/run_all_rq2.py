import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.common import DATASET_DEFAULTS, FORMAL_SEEDS, RQ2_METHODS, formal_folds, write_json
from experiments.run_rq2_training import expected_rq2_config_hash, rq2_run_key, run_one


# 从 RQ2 record 中取最终测试指标。
def _metric(record: dict, name: str):
    if name in record and record.get(name) is not None:
        return record.get(name)
    metrics = record.get("metrics") or {}
    if name == "accuracy":
        return metrics.get("test_accuracy")
    if name == "macro_f1":
        return metrics.get("test_macro_f1")
    if name == "final_loss":
        return metrics.get("test_loss")
    return metrics.get(name)


# 展平 per-round 数值列表。
def _flat_values(rows, name: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(name)
        if isinstance(value, list):
            values.extend(float(item) for item in value if item is not None)
        elif value is not None:
            values.append(float(value))
    return values


# 对 RQ2 raw records 计算 mean/std/count 聚合。
def aggregate(records: list[dict]) -> dict:
    groups = {}
    for record in records:
        key = (record.get("dataset"), record.get("method"))
        groups.setdefault(key, []).append(record)
    out = {}
    for (dataset, method), rows in groups.items():
        success = [row for row in rows if row.get("status") == "success"]
        item = {"count": len(success), "failed": len(rows) - len(success)}
        for metric in ["accuracy", "macro_f1", "final_loss", "loss_std"]:
            values = [float(_metric(row, metric)) for row in success if _metric(row, metric) is not None]
            item[f"{metric}_mean"] = float(statistics.mean(values)) if values else None
            item[f"{metric}_std"] = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
        for metric in ["coverage_per_round", "normalized_pseudo_yield_per_round"]:
            values = _flat_values(success, metric)
            stem = "coverage" if metric == "coverage_per_round" else "pseudo_yield"
            item[f"{stem}_mean"] = float(statistics.mean(values)) if values else None
            item[f"{stem}_std"] = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
        item["repair_rate"] = float(
            sum(1 for row in success if row.get("feasibility_repair_applied") is True) / max(1, len(success))
        )
        out[f"{dataset}_{method}"] = item
    return out


# 将 RQ2 aggregate 写成 CSV。
def write_summary_csv(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in summary.values() for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", *keys])
        writer.writeheader()
        for group, row in sorted(summary.items()):
            writer.writerow({"group": group, **row})


# 生成 RQ2 全实验 run plan。
def build_plan(datasets, seeds, methods) -> list[dict]:
    plan = []
    for dataset in datasets:
        folds = formal_folds(dataset)
        plan.append(
            {
                "dataset": dataset,
                "folds": folds,
                "seeds": [int(seed) for seed in seeds],
                "methods": list(methods),
                "runs": int(len(folds) * len(seeds) * len(methods)),
            }
        )
    return plan


# 打印正式 RQ2 实验计划。
def print_plan(plan: list[dict]) -> None:
    print("RQ2 Formal Experiment Plan")
    print("==========================")
    for item in plan:
        print(f"dataset: {item['dataset']}")
        print(f"folds: {item['folds']}")
        print(f"seeds: {item['seeds']}")
        print(f"methods: {len(item['methods'])}")
        print(f"runs: {item['runs']}")
        print("")
    print(f"TOTAL EXPECTED RUNS: {sum(item['runs'] for item in plan)}")


# 读取已存在的 run result 以支持 success resume 和 failed 记录。
def load_existing_result(results_root: Path, dataset: str, fold: int | None, seed: int, method: str, global_rounds: int | None, retry_failed: bool):
    run_dir = results_root / "rq2" / "raw" / rq2_run_key(dataset, fold, seed, method, global_rounds)
    success_path = run_dir / "rq2_result.json"
    failed_path = run_dir / "failed_run.json"
    path = success_path if success_path.exists() else failed_path
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("status") == "failed" and retry_failed:
        return None
    if payload.get("dataset") != dataset or payload.get("fold") != fold or int(payload.get("seed")) != int(seed):
        return None
    if payload.get("method") != method or not payload.get("config_hash"):
        return None
    expected_hash = expected_rq2_config_hash(dataset, fold, seed, method, results_root, global_rounds)
    if payload.get("config_hash") != expected_hash:
        return None
    return payload


# 根据用户要求检查 CUDA 是否可用。
def require_cuda_if_requested(require_cuda: bool) -> None:
    if bool(require_cuda) and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by --require-cuda, but torch.cuda.is_available() is false.")


# 解析 RQ2 全实验参数。
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run all RQ2 methods on available real artifacts.")
    parser.add_argument("--datasets", nargs="*", choices=tuple(DATASET_DEFAULTS), default=list(DATASET_DEFAULTS))
    parser.add_argument("--seeds", nargs="*", type=int, default=FORMAL_SEEDS)
    parser.add_argument("--methods", nargs="*", choices=RQ2_METHODS, default=RQ2_METHODS)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--global-rounds", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args(argv)


# 执行 RQ2 全实验并保存失败原因。
def main(argv=None):
    args = parse_args(argv)
    require_cuda_if_requested(args.require_cuda)
    results_root = (ROOT / args.results_root).resolve()
    plan = build_plan(args.datasets, args.seeds, args.methods)
    print_plan(plan)
    records = []
    for dataset in args.datasets:
        for fold in formal_folds(dataset):
            for seed in args.seeds:
                for method in args.methods:
                    record = load_existing_result(
                        results_root,
                        dataset,
                        fold,
                        int(seed),
                        method,
                        args.global_rounds,
                        args.retry_failed,
                    )
                    if record is None:
                        record = run_one(
                            dataset,
                            fold,
                            int(seed),
                            method,
                            results_root,
                            args.device,
                            args.global_rounds,
                        )
                    records.append(record)
    summary = aggregate(records)
    write_json(results_root / "rq2" / "aggregated" / "summary.json", summary)
    write_summary_csv(results_root / "rq2" / "aggregated" / "summary.csv", summary)
    expected = sum(item["runs"] for item in plan)
    failed = sum(1 for row in records if row.get("status") == "failed")
    print(f"RQ2 all finished: expected={expected} total={len(records)} success={len(records) - failed} failed={failed}")
    if len(records) != expected:
        raise RuntimeError(f"RQ2 runner missed runs: expected={expected}, actual={len(records)}")


if __name__ == "__main__":
    main()
