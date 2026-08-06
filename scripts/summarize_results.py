import argparse
import json
import re
from pathlib import Path

import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from MSL.utils.config import load_config


# 论文指标：acc=Accuracy, macro_f1, weighted_f1。
METRIC_KEYS = {
    "acc": "test_accuracy",
    "macro_f1": "test_macro_f1",
    "weighted_f1": "test_weighted_f1",
}

FOLD_PATTERN = re.compile(r"fold(\d+)")


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
        "metrics": {key: metrics.get(field) for key, field in METRIC_KEYS.items()},
    }


def _mean(values: list[float]) -> float | None:
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def build_dataset_summary(records: list[dict]) -> dict:
    records = sorted(
        records,
        key=lambda record: (
            record["fold"] is None,
            record["fold"] if record["fold"] is not None else record["seed"],
        ),
    )
    summary = {
        "average": {
            key: _mean([record["metrics"][key] for record in records])
            for key in METRIC_KEYS
        }
    }
    for record in records:
        key = f"fold{record['fold']}" if record["fold"] is not None else f"seed{record['seed']}"
        summary[key] = {
            metric_key: record["metrics"][metric_key]
            for metric_key in METRIC_KEYS
        }
    return summary


def build_summary(results_root: Path) -> dict:
    records = []
    for path in sorted(results_root.glob("experiments/**/final_metrics.json")):
        record = _load_record(path.parent)
        if record is not None:
            records.append(record)

    # run 级聚合：每个 run（seed/fold 文件夹的父级 = attempt 目录）一个 summary.json，与 seed 同级。
    run_summaries = {}
    for record in records:
        run_dir = Path(record["run_dir"]).parents[0]
        run_summaries.setdefault(run_dir, []).append(record)
    for run_dir, run_records in run_summaries.items():
        run_summary = build_dataset_summary(run_records)
        (run_dir / "summary.json").write_text(
            json.dumps(run_summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    by_dataset = {}
    for dataset in sorted({record["dataset"] for record in records}):
        matches = [record for record in records if record["dataset"] == dataset]
        by_dataset[dataset] = build_dataset_summary(matches)
    return {
        "datasets": by_dataset,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate multi-fold / multi-seed Stage3 results under a results root."
    )
    parser.add_argument(
        "--results-root",
        default=str(ROOT / "local" / "results_msl"),
        help="Results root; defaults to local/results_msl.",
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
                {"average": {}},
            )
        }
    else:
        selected = summary["datasets"]
    summary_dir = results_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    for dataset, dataset_summary in selected.items():
        path = summary_dir / f"{dataset}.json"
        path.write_text(json.dumps(dataset_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        average = dataset_summary.get("average", {})
        acc_mean = average.get("acc")
        wf1_mean = average.get("weighted_f1")
        runs = len([key for key in dataset_summary if key != "average"])
        print(
            f"{dataset:10s} runs={runs:2d} "
            f"acc={acc_mean} wf1={wf1_mean}"
        )
    summary_path = summary_dir / "summary.json"
    full_summary = build_summary(results_root)
    summary_path.write_text(json.dumps(full_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"summary_dir={summary_dir}")


if __name__ == "__main__":
    main()
