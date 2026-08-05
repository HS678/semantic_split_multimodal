import argparse
import json
import re
import statistics
from pathlib import Path

import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic_split_multimodal.utils.config import load_config


METRICS = [
    "test_accuracy",
    "test_balanced_accuracy",
    "test_macro_f1",
    "test_weighted_f1",
    "test_binary_f1",
]

FOLD_PATTERN = re.compile(r"fold(\d+)_v1$")


def _load_record(run_dir: Path) -> dict | None:
    metrics_path = run_dir / "final_metrics.json"
    config_path = run_dir / "resolved_config.config"
    metadata_path = run_dir / "stage3_metadata.json"
    if not all(path.is_file() for path in (metrics_path, config_path, metadata_path)):
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    cfg = load_config(config_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("status") != "success"
        or metrics.get("test_eval_status") != "success"
        or int(metrics.get("test_evaluation_count", 0)) != 1
        or not metrics.get("official_result")
    ):
        return None
    dataset_cfg = cfg.get("dataset", {})
    split_protocol = str(dataset_cfg.get("split_protocol", ""))
    fold_match = FOLD_PATTERN.search(split_protocol)
    return {
        "dataset": str(dataset_cfg.get("type", "unknown")),
        "seed": int(cfg.get("seed", 0)),
        "attempt": int(cfg.get("stage3", {}).get("attempt", 0)),
        "fold": int(fold_match.group(1)) if fold_match else None,
        "split_protocol": split_protocol,
        "run_dir": str(run_dir.resolve()),
        "metrics": {name: metrics.get(name) for name in METRICS},
    }


def _mean_std(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "values": []}
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "values": [float(value) for value in values],
    }


def _aggregate(records: list[dict]) -> dict:
    aggregates = {}
    for metric in METRICS:
        values = [record["metrics"][metric] for record in records]
        aggregates[metric] = _mean_std([float(v) for v in values if v is not None])
    return aggregates


def build_dataset_summary(records: list[dict]) -> dict:
    records = sorted(
        records,
        key=lambda record: (
            record["fold"] is None,
            record["fold"] if record["fold"] is not None else record["seed"],
        ),
    )
    has_folds = any(record["fold"] is not None for record in records)
    dimension = "fold" if has_folds else "seed"
    return {
        "dimension": dimension,
        "runs": records,
        "aggregate": _aggregate(records),
        "num_runs": len(records),
    }


def build_summary(results_root: Path) -> dict:
    records = []
    for path in sorted(results_root.glob("experiments/**/final_metrics.json")):
        record = _load_record(path.parent)
        if record is not None:
            records.append(record)

    by_dataset = {}
    for dataset in sorted({record["dataset"] for record in records}):
        matches = [record for record in records if record["dataset"] == dataset]
        by_dataset[dataset] = build_dataset_summary(matches)
    return {
        "results_root": str(results_root.resolve()),
        "datasets": by_dataset,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate multi-fold / multi-seed Stage3 results under a results root."
    )
    parser.add_argument(
        "--results-root",
        default=str(ROOT / "local" / "results_test1"),
        help="Results root; defaults to local/results_test1.",
    )
    parser.add_argument(
        "--dataset",
        help="Only aggregate the given dataset (e.g. uci_har); default aggregates all.",
    )
    args = parser.parse_args()
    results_root = Path(args.results_root).resolve()
    summary = build_summary(results_root)
    if args.dataset:
        dataset_key = str(args.dataset).strip().lower()
        if dataset_key not in summary["datasets"]:
            print(f"WARNING: no completed runs found for dataset={dataset_key}")
        selected = {
            dataset_key: summary["datasets"].get(
                dataset_key,
                {"dimension": "seed", "runs": [], "aggregate": {}, "num_runs": 0},
            )
        }
    else:
        selected = summary["datasets"]
    summary_dir = results_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    for dataset, dataset_summary in selected.items():
        path = summary_dir / f"{dataset}.json"
        path.write_text(json.dumps(dataset_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        agg = dataset_summary["aggregate"]
        acc_mean = agg.get("test_accuracy", {}).get("mean")
        wf1_mean = agg.get("test_weighted_f1", {}).get("mean")
        print(
            f"{dataset:10s} {dataset_summary['dimension']:6s} "
            f"runs={dataset_summary['num_runs']:2d} "
            f"acc={acc_mean} wf1={wf1_mean}"
        )
    summary_path = summary_dir / "summary.json"
    full_summary = build_summary(results_root)
    summary_path.write_text(json.dumps(full_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"summary_dir={summary_dir}")


if __name__ == "__main__":
    main()
