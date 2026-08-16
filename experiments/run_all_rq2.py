import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.common import DATASET_DEFAULTS, FORMAL_SEEDS, RQ2_METHODS, formal_folds, write_json
from experiments.run_rq2_training import run_one


# 从 RQ2 record 中取最终测试指标。
def _metric(record: dict, name: str):
    metrics = record.get("metrics") or {}
    if name == "accuracy":
        return metrics.get("test_accuracy")
    if name == "macro_f1":
        return metrics.get("test_macro_f1")
    if name == "final_loss":
        return metrics.get("test_loss")
    return metrics.get(name)


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
        for metric in ["accuracy", "macro_f1", "final_loss"]:
            values = [float(_metric(row, metric)) for row in success if _metric(row, metric) is not None]
            item[f"{metric}_mean"] = float(statistics.mean(values)) if values else None
            item[f"{metric}_std"] = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
        out[f"{dataset}_{method}"] = item
    return out


# 解析 RQ2 全实验参数。
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run all RQ2 methods on available real artifacts.")
    parser.add_argument("--datasets", nargs="*", choices=tuple(DATASET_DEFAULTS), default=list(DATASET_DEFAULTS))
    parser.add_argument("--seeds", nargs="*", type=int, default=FORMAL_SEEDS)
    parser.add_argument("--methods", nargs="*", choices=RQ2_METHODS, default=RQ2_METHODS)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--global-rounds", type=int)
    return parser.parse_args(argv)


# 执行 RQ2 全实验并保存失败原因。
def main(argv=None):
    args = parse_args(argv)
    results_root = (ROOT / args.results_root).resolve()
    records = []
    for dataset in args.datasets:
        for fold in formal_folds(dataset):
            for seed in args.seeds:
                for method in args.methods:
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
    write_json(results_root / "rq2" / "aggregated" / "summary.json", aggregate(records))
    print(f"RQ2 all finished: total={len(records)} failed={sum(1 for row in records if row.get('status') == 'failed')}")


if __name__ == "__main__":
    main()
