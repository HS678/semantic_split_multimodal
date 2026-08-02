import argparse
import json
from pathlib import Path
import statistics
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


def _load_record(path: Path) -> dict | None:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    run_dir = path.parent
    config_path = run_dir / "resolved_config.config"
    metadata_path = run_dir / "stage3_metadata.json"
    if not config_path.is_file() or not metadata_path.is_file():
        return None
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
    test_sessions = dataset_cfg.get("test_sessions", [])
    return {
        "dataset": str(dataset_cfg.get("type", "unknown")),
        "seed": int(cfg.get("seed", 0)),
        "attempt": int(cfg.get("stage3", {}).get("attempt", 0)),
        "config_signature": cfg.get("stage3", {}).get("config_signature"),
        "split_protocol": dataset_cfg.get("split_protocol"),
        "test_session": int(test_sessions[0]) if len(test_sessions) == 1 else None,
        "run_dir": str(run_dir.resolve()),
        "best_round": metrics.get("best_round"),
        "stop_round": metrics.get("stop_round"),
        "stop_reason": metrics.get("stop_reason"),
        "metrics": {name: metrics.get(name) for name in METRICS},
        "confusion_matrix": metrics.get("test_confusion_matrix"),
        "test_num_eval_samples": metrics.get("test_num_eval_samples"),
    }


def _mean_std(values: list[float]) -> dict:
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "values": [float(value) for value in values],
    }


def _sum_confusions(records: list[dict]) -> list[list[int]] | None:
    matrices = [record.get("confusion_matrix") for record in records]
    if not matrices or any(matrix is None for matrix in matrices):
        return None
    rows = len(matrices[0])
    if any(len(matrix) != rows or any(len(row) != rows for row in matrix) for matrix in matrices):
        raise ValueError("Cannot aggregate confusion matrices with different shapes.")
    return [
        [sum(int(matrix[row][col]) for matrix in matrices) for col in range(rows)]
        for row in range(rows)
    ]


def build_summary(results_root: Path) -> dict:
    records = []
    for path in sorted(results_root.glob("experiments/oracle_true_cluster/**/final_metrics.json")):
        record = _load_record(path)
        if record is not None:
            records.append(record)
    by_dataset = {}
    for dataset in ["uci_har", "mhealth", "pamap2", "cmu_mosei"]:
        matches = [record for record in records if record["dataset"] == dataset]
        by_dataset[dataset] = {
            "status": "complete" if matches else "missing",
            "runs": matches,
        }

    iemocap_records = sorted(
        [record for record in records if record["dataset"] == "iemocap"],
        key=lambda record: record["test_session"] or 0,
    )
    completed_sessions = sorted(
        record["test_session"] for record in iemocap_records if record["test_session"] is not None
    )
    aggregate_metrics = {}
    for metric in METRICS:
        values = [record["metrics"][metric] for record in iemocap_records]
        values = [float(value) for value in values if value is not None]
        aggregate_metrics[metric] = _mean_std(values) if values else None
    by_dataset["iemocap"] = {
        "status": "complete" if completed_sessions == [1, 2, 3, 4, 5] else "incomplete",
        "completed_test_sessions": completed_sessions,
        "missing_test_sessions": sorted(set(range(1, 6)) - set(completed_sessions)),
        "fold_runs": iemocap_records,
        "five_fold_metrics": aggregate_metrics,
        "aggregate_confusion_matrix": _sum_confusions(iemocap_records),
    }
    return {
        "protocol": "oracle_true_cluster_current_development",
        "results_root": str(results_root.resolve()),
        "datasets": by_dataset,
    }


def _fmt(value) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def write_markdown(summary: dict, path: Path) -> None:
    lines = [
        "# 五数据集正式结果汇总",
        "",
        "> 所有数值由 `final_metrics.json` 自动读取；缺失实验不会生成占位指标。",
        "",
        "| 数据集 | 状态 | Accuracy | Balanced Acc/UA | Macro-F1 | Weighted-F1 | Binary-F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in ["uci_har", "mhealth", "pamap2", "cmu_mosei"]:
        item = summary["datasets"][dataset]
        run = item["runs"][0] if item["runs"] else None
        metrics = {} if run is None else run["metrics"]
        lines.append(
            f"| {dataset} | {item['status']} | {_fmt(metrics.get('test_accuracy'))} | "
            f"{_fmt(metrics.get('test_balanced_accuracy'))} | {_fmt(metrics.get('test_macro_f1'))} | "
            f"{_fmt(metrics.get('test_weighted_f1'))} | {_fmt(metrics.get('test_binary_f1'))} |"
        )
    iemocap = summary["datasets"]["iemocap"]
    fold_metrics = iemocap["five_fold_metrics"]
    def fold_value(metric):
        value = fold_metrics.get(metric)
        return "—" if value is None else f"{value['mean']:.4f} ± {value['std']:.4f}"
    lines.append(
        f"| iemocap (5-fold) | {iemocap['status']} | {fold_value('test_accuracy')} | "
        f"{fold_value('test_balanced_accuracy')} | {fold_value('test_macro_f1')} | "
        f"{fold_value('test_weighted_f1')} | — |"
    )
    lines.extend(
        [
            "",
            "## IEMOCAP 折状态",
            "",
            f"- 已完成 test Session：{iemocap['completed_test_sessions']}",
            f"- 缺失 test Session：{iemocap['missing_test_sessions']}",
            "",
            "## 运行记录",
            "",
        ]
    )
    for dataset, item in summary["datasets"].items():
        runs = item.get("runs", item.get("fold_runs", []))
        for run in runs:
            lines.append(f"- `{dataset}`：`{run['run_dir']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Summarize completed formal results without fabricating missing runs.")
    parser.add_argument(
        "--results-root",
        default=str(ROOT / "local" / "results"),
        help="Results root; defaults to local/results.",
    )
    args = parser.parse_args()
    results_root = Path(args.results_root).resolve()
    summary = build_summary(results_root)
    json_path = results_root / "formal_summary.json"
    markdown_path = results_root / "formal_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(summary, markdown_path)
    print(f"json_summary={json_path}")
    print(f"markdown_summary={markdown_path}")
    print(f"iemocap_status={summary['datasets']['iemocap']['status']}")


if __name__ == "__main__":
    main()
