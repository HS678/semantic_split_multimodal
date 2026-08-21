# C2 curve-mode convergence metrics computed after training from test_curve.csv.
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.common import DATASET_PROTOCOLS, formal_run_grid, protocol_hash, runtime_metadata, write_json
from experiments.training import training_run_dir


THRESHOLDS = (0.7, 0.8, 0.9)


def read_curve_csv(path: Path) -> list[dict]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing curve metrics file: {path}")
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "round": int(float(row["round"])),
            "test_macro_f1": float(row["test_macro_f1"]),
            "test_accuracy": float(row["test_accuracy"]),
            "test_weighted_f1": float(row["test_weighted_f1"]),
        }
        for row in rows
    ]


def compute_curve_metrics(rows: list[dict]) -> dict:
    if len(rows) < 3:
        raise RuntimeError("Convergence metrics require at least 3 curve checkpoints.")
    ordered = sorted(rows, key=lambda row: int(row["round"]))
    m_ref = float(statistics.mean(float(row["test_macro_f1"]) for row in ordered[-3:]))
    out = {
        "M_ref": m_ref,
        "num_checkpoints": int(len(ordered)),
        "last_three_rounds": [int(row["round"]) for row in ordered[-3:]],
    }
    for threshold in THRESHOLDS:
        target = float(threshold * m_ref)
        reached = [
            int(row["round"])
            for row in ordered
            if float(row["test_macro_f1"]) >= target
        ]
        if not reached:
            raise RuntimeError(f"Curve never reaches R{int(threshold * 100)} target={target}.")
        out[f"R{int(threshold * 100)}"] = int(reached[0])
    return out


def _std(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def aggregate_convergence(
    results_root: Path,
    datasets: list[str] | None = None,
    methods: list[str] | None = None,
    seeds: list[int] | None = None,
) -> dict:
    datasets = list(DATASET_PROTOCOLS) if datasets is None else list(datasets)
    methods = ["randomsl", "ours", "oracle"] if methods is None else list(methods)
    records = []
    by_run = {}
    for dataset in datasets:
        for fold, seed in formal_run_grid(dataset, seeds):
            for method in methods:
                run_dir = training_run_dir(results_root, dataset, fold, int(seed), method, None)
                metrics = compute_curve_metrics(read_curve_csv(run_dir / "test_curve.csv"))
                record = {
                    **runtime_metadata(ROOT, dataset, fold, int(seed), method),
                    "protocol_hash": protocol_hash(),
                    "evaluation_mode": "curve",
                    "run_dir": str(run_dir),
                    **metrics,
                }
                records.append(record)
                by_run[(dataset, fold, int(seed), method)] = record

    for record in records:
        key = (record["dataset"], record["fold"], int(record["seed"]))
        random_record = by_run.get((*key, "randomsl"))
        if random_record is None:
            raise RuntimeError(f"Missing matched RandomSL curve for {key}.")
        if record["method"] == "randomsl":
            record["S90"] = 1.0
        else:
            record["S90"] = float(random_record["R90"] / record["R90"])

    groups = {}
    for record in records:
        groups.setdefault((record["dataset"], record["method"]), []).append(record)

    summary = {}
    for (dataset, method), rows in groups.items():
        item = {"count": len(rows), "failed": 0}
        for metric in ["M_ref", "R70", "R80", "R90", "S90"]:
            values = [float(row[metric]) for row in rows]
            item[f"{metric}_mean"] = float(statistics.mean(values)) if values else None
            item[f"{metric}_std"] = _std(values) if values else 0.0
        item["per_run"] = [
            {
                "dataset": row["dataset"],
                "fold": row["fold"],
                "seed": int(row["seed"]),
                "method": row["method"],
                "M_ref": float(row["M_ref"]),
                "R70": int(row["R70"]),
                "R80": int(row["R80"]),
                "R90": int(row["R90"]),
                "S90": float(row["S90"]),
            }
            for row in rows
        ]
        summary[f"{dataset}_{method}"] = item
    return {"records": records, "summary": summary}


def write_summary_csv(path: Path, summary: dict) -> None:
    fields = ["group", "count", "failed", "M_ref_mean", "M_ref_std", "R70_mean", "R70_std", "R80_mean", "R80_std", "R90_mean", "R90_std", "S90_mean", "S90_std"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group, row in sorted(summary.items()):
            writer.writerow({"group": group, **{field: row.get(field) for field in fields if field != "group"}})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate C2 curve-mode convergence metrics.")
    parser.add_argument("--results-root", default="results/c2_curve")
    parser.add_argument("--datasets", nargs="*", choices=tuple(DATASET_PROTOCOLS), default=list(DATASET_PROTOCOLS))
    parser.add_argument("--methods", nargs="*", choices=("randomsl", "ours", "oracle"), default=["randomsl", "ours", "oracle"])
    parser.add_argument("--seeds", nargs="*", type=int)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    results_root = (ROOT / args.results_root).resolve()
    payload = aggregate_convergence(results_root, args.datasets, args.methods, args.seeds)
    write_json(results_root / "convergence" / "summary.json", payload["summary"])
    write_json(results_root / "convergence" / "records.json", {"records": payload["records"]})
    write_summary_csv(results_root / "convergence" / "summary.csv", payload["summary"])
    print(f"convergence aggregation finished: records={len(payload['records'])}")


if __name__ == "__main__":
    main()
