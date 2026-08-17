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

from experiments.common import DATASET_DEFAULTS, RQ1_METHODS, formal_result_dir, formal_run_grid, write_json
from experiments.run_rq1_discovery import run_one


# 对 RQ1 raw records 计算 mean/std/count 聚合。
def aggregate(records: list[dict]) -> dict:
    groups = {}
    for record in records:
        key = (record.get("dataset"), record.get("method"))
        groups.setdefault(key, []).append(record)
    out = {}
    for (dataset, method), rows in groups.items():
        success = [row for row in rows if row.get("status") == "success"]
        item = {"count": len(success), "failed": len(rows) - len(success)}
        for metric in ["ARI", "NMI", "hungarian_ACC", "cluster_count_error", "Q_hat", "M"]:
            values = [float(row[metric]) for row in success if metric in row]
            item[f"{metric}_mean"] = float(statistics.mean(values)) if values else None
            item[f"{metric}_std"] = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
        out[f"{dataset}_{method}"] = item
    return out


# 解析 RQ1 全实验参数。
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run all RQ1 discovery jobs on available real artifacts.")
    parser.add_argument("--datasets", nargs="*", choices=tuple(DATASET_DEFAULTS), default=list(DATASET_DEFAULTS))
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--results-root", default="results")
    return parser.parse_args(argv)


# 执行 RQ1 全实验并保存失败原因。
def main(argv=None):
    args = parse_args(argv)
    results_root = (ROOT / args.results_root).resolve()
    records = []
    for dataset in args.datasets:
        for fold, seed in formal_run_grid(dataset, args.seeds):
            for method in RQ1_METHODS:
                try:
                    records.append(run_one(dataset, fold, int(seed), method, results_root))
                except Exception as exc:
                    failed = {
                        "dataset": dataset,
                        "fold": fold,
                        "seed": int(seed),
                        "method": method,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                    records.append(failed)
                    write_json(
                        formal_result_dir(results_root, "rq1", dataset, method, fold, seed) / "failed_run.json",
                        failed,
                    )
    write_json(results_root / "rq1" / "aggregated" / "summary.json", aggregate(records))
    print(f"RQ1 all finished: total={len(records)} failed={sum(1 for row in records if row.get('status') == 'failed')}")


if __name__ == "__main__":
    main()
