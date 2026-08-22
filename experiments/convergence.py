# C2 convergence metrics with common Oracle-stable Macro-F1 targets.
import argparse
import csv
import json
import math
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


C2_V2_METHODS = (
    "randomsl",
    "kmeans2",
    "kmeans3",
    "kmeans4",
    "kmeans5",
    "auto_kmeans",
    "gmm_bic",
    "ours",
    "oracle",
)
COMMON_TARGET_LEVELS = (60, 70, 80)


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
    """Legacy self-relative metric retained only for compatibility."""

    if len(rows) < 3:
        raise RuntimeError("Convergence metrics require at least 3 curve checkpoints.")
    ordered = sorted(rows, key=lambda row: int(row["round"]))
    m_ref = float(statistics.mean(float(row["test_macro_f1"]) for row in ordered[-3:]))
    out = {
        "definition": "legacy_self_relative",
        "M_ref": m_ref,
        "num_checkpoints": int(len(ordered)),
        "last_three_rounds": [int(row["round"]) for row in ordered[-3:]],
    }
    for threshold in (0.7, 0.8, 0.9):
        target = float(threshold * m_ref)
        reached = [
            int(row["round"])
            for row in ordered
            if float(row["test_macro_f1"]) >= target
        ]
        out[f"R{int(threshold * 100)}"] = int(reached[0]) if reached else math.nan
    return out


def oracle_stable_target(oracle_rows: list[dict]) -> dict:
    if len(oracle_rows) < 3:
        raise RuntimeError("Oracle common target requires at least 3 curve checkpoints.")
    ordered = sorted(oracle_rows, key=lambda row: int(row["round"]))
    last3 = ordered[-3:]
    stable = float(statistics.mean(float(row["test_macro_f1"]) for row in last3))
    return {
        "oracle_stable_macro_f1": stable,
        "oracle_last3_rounds": [int(row["round"]) for row in last3],
        "target60": float(0.60 * stable),
        "target70": float(0.70 * stable),
        "target80": float(0.80 * stable),
    }


def first_common_target_crossing(rows: list[dict], target: float) -> tuple[float, float, bool]:
    ordered = sorted(rows, key=lambda row: int(row["round"]))
    for row in ordered:
        macro_f1 = float(row["test_macro_f1"])
        if macro_f1 >= float(target):
            return float(int(row["round"])), macro_f1, True
    return math.nan, math.nan, False


def build_v2_run_manifest(
    *,
    output_root: Path,
    legacy_curve_root: Path,
    new_curve_root: Path,
    datasets: list[str] | None = None,
    methods: list[str] | None = None,
    seeds: list[int] | None = None,
) -> list[dict]:
    datasets = list(DATASET_PROTOCOLS) if datasets is None else list(datasets)
    methods = list(C2_V2_METHODS) if methods is None else list(methods)
    records = []
    for dataset in datasets:
        for fold, seed in formal_run_grid(dataset, seeds):
            for method in methods:
                source_root = legacy_curve_root if method in {"randomsl", "ours", "oracle"} else new_curve_root
                run_dir = training_run_dir(source_root, dataset, fold, int(seed), method, None)
                records.append(
                    {
                        "dataset": dataset,
                        "fold": fold,
                        "seed": int(seed),
                        "method": method,
                        "evaluation_mode": "curve",
                        "source_root": str(source_root),
                        "run_dir": str(run_dir),
                        "curve_file": str(run_dir / "test_curve.csv"),
                        "source_kind": "legacy_reused" if method in {"randomsl", "ours", "oracle"} else "v2_new_run",
                        "v2_output_root": str(output_root),
                    }
                )
    return records


def load_v2_run_manifest(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload["runs"] if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("run manifest must be a list or an object with a runs list.")
    return records


def _matched_key(record: dict) -> tuple:
    return (record["dataset"], record.get("fold"), int(record["seed"]))


def aggregate_common_targets(manifest_records: list[dict]) -> dict:
    rows_by_key_method = {}
    manifest_by_key_method = {}
    for item in manifest_records:
        if str(item.get("evaluation_mode", "curve")) != "curve":
            raise ValueError(f"Common target aggregation only accepts curve records: {item}")
        key = (*_matched_key(item), item["method"])
        rows_by_key_method[key] = read_curve_csv(Path(item["curve_file"]))
        manifest_by_key_method[key] = dict(item)

    out_records = []
    for dataset, fold, seed in sorted({_matched_key(item) for item in manifest_records}):
        oracle_key = (dataset, fold, int(seed), "oracle")
        if oracle_key not in rows_by_key_method:
            raise RuntimeError(f"Missing matched Oracle curve for {(dataset, fold, seed)}.")
        target_info = oracle_stable_target(rows_by_key_method[oracle_key])
        for method in C2_V2_METHODS:
            key = (dataset, fold, int(seed), method)
            if key not in rows_by_key_method:
                raise RuntimeError(f"Missing method curve for {key}.")
            record = {
                **runtime_metadata(ROOT, dataset, fold, int(seed), method),
                "protocol_hash": protocol_hash(),
                "definition": "common_oracle_target",
                "evaluation_mode": "curve",
                "source_kind": manifest_by_key_method[key].get("source_kind"),
                "run_dir": manifest_by_key_method[key].get("run_dir"),
                "curve_file": manifest_by_key_method[key].get("curve_file"),
                "oracle_stable_macro_f1": float(target_info["oracle_stable_macro_f1"]),
                "oracle_last3_round_1": int(target_info["oracle_last3_rounds"][0]),
                "oracle_last3_round_2": int(target_info["oracle_last3_rounds"][1]),
                "oracle_last3_round_3": int(target_info["oracle_last3_rounds"][2]),
                "target60": float(target_info["target60"]),
                "target70": float(target_info["target70"]),
                "target80": float(target_info["target80"]),
                "Rmax": int(max(row["round"] for row in rows_by_key_method[key])),
            }
            for level in COMMON_TARGET_LEVELS:
                round_value, macro_f1, reached = first_common_target_crossing(
                    rows_by_key_method[key],
                    record[f"target{level}"],
                )
                record[f"R{level}"] = round_value
                record[f"macro_f1_at_R{level}"] = macro_f1
                record[f"reached{level}"] = bool(reached)
                record[f"status{level}"] = "reached" if reached else "not_reached"
            out_records.append(record)

    by_run = {
        (*_matched_key(record), record["method"]): record
        for record in out_records
    }
    for record in out_records:
        random_record = by_run.get((*_matched_key(record), "randomsl"))
        if random_record is None:
            raise RuntimeError(f"Missing matched RandomSL curve for {_matched_key(record)}.")
        for level in COMMON_TARGET_LEVELS:
            if record["method"] == "randomsl":
                speedup = 1.0 if record[f"reached{level}"] else math.nan
            elif random_record[f"reached{level}"] and record[f"reached{level}"]:
                speedup = float(random_record[f"R{level}"] / record[f"R{level}"])
            else:
                speedup = math.nan
            record[f"S{level}"] = speedup

    return {"records": out_records, "summary": summarize_common_targets(out_records)}


def _finite(values):
    return [float(value) for value in values if value is not None and not math.isnan(float(value))]


def _mean(values):
    values = _finite(values)
    return float(statistics.mean(values)) if values else math.nan


def _std(values):
    values = _finite(values)
    return float(statistics.pstdev(values)) if len(values) > 1 else (0.0 if values else math.nan)


def summarize_common_targets(records: list[dict]) -> dict:
    groups = {}
    for record in records:
        groups.setdefault((record["dataset"], record["method"]), []).append(record)
    summary = {}
    for (dataset, method), rows in groups.items():
        item = {"count": len(rows), "failed": 0}
        for level in COMMON_TARGET_LEVELS:
            reached_rows = [row for row in rows if bool(row[f"reached{level}"])]
            item[f"R{level}_mean"] = _mean([row[f"R{level}"] for row in reached_rows])
            item[f"R{level}_std"] = _std([row[f"R{level}"] for row in reached_rows])
            item[f"S{level}_mean"] = _mean([row[f"S{level}"] for row in rows])
            item[f"S{level}_std"] = _std([row[f"S{level}"] for row in rows])
            item[f"reach_rate{level}"] = float(len(reached_rows) / max(1, len(rows)))
        item["per_run"] = [
            {
                "dataset": row["dataset"],
                "fold": row["fold"],
                "seed": int(row["seed"]),
                "method": row["method"],
                **{f"R{level}": row[f"R{level}"] for level in COMMON_TARGET_LEVELS},
                **{f"S{level}": row[f"S{level}"] for level in COMMON_TARGET_LEVELS},
                **{f"reached{level}": row[f"reached{level}"] for level in COMMON_TARGET_LEVELS},
            }
            for row in rows
        ]
        summary[f"{dataset}_{method}"] = item
    return summary


def _csv_value(value):
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return value


def write_records_csv(path: Path, records: list[dict]) -> None:
    fields = [
        "dataset", "fold", "seed", "method", "oracle_stable_macro_f1",
        "oracle_last3_round_1", "oracle_last3_round_2", "oracle_last3_round_3",
        "target60", "target70", "target80",
        "R60", "R70", "R80",
        "macro_f1_at_R60", "macro_f1_at_R70", "macro_f1_at_R80",
        "reached60", "reached70", "reached80",
        "S60", "S70", "S80",
        "Rmax", "status60", "status70", "status80",
        "source_kind", "run_dir", "curve_file",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def write_summary_csv(path: Path, summary: dict) -> None:
    fields = ["group", "count", "failed"]
    for level in COMMON_TARGET_LEVELS:
        fields.extend([
            f"R{level}_mean", f"R{level}_std",
            f"S{level}_mean", f"S{level}_std",
            f"reach_rate{level}",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group, row in sorted(summary.items()):
            writer.writerow({"group": group, **{field: _csv_value(row.get(field)) for field in fields if field != "group"}})


def _format_round(mean_value, std_value, reach_rate):
    if mean_value is None or (isinstance(mean_value, float) and math.isnan(mean_value)):
        return "N.R."
    text = f"{float(mean_value):.1f} ± {float(std_value):.1f}"
    if float(reach_rate) < 1.0:
        text += f" ({float(reach_rate):.0%})"
    return text


def write_table_artifacts(table_dir: Path, summary: dict) -> None:
    table_dir = Path(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in DATASET_PROTOCOLS:
        for method in C2_V2_METHODS:
            item = summary.get(f"{dataset}_{method}")
            if not item:
                continue
            rows.append(
                {
                    "Dataset": dataset,
                    "Method": method,
                    "60%": _format_round(item.get("R60_mean"), item.get("R60_std"), item.get("reach_rate60", 0.0)),
                    "70%": _format_round(item.get("R70_mean"), item.get("R70_std"), item.get("reach_rate70", 0.0)),
                    "80%": _format_round(item.get("R80_mean"), item.get("R80_std"), item.get("reach_rate80", 0.0)),
                }
            )
    csv_path = table_dir / "table_common_targets.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Dataset", "Method", "60%", "70%", "80%"])
        writer.writeheader()
        writer.writerows(rows)

    tex_lines = [
        "\\begin{tabular}{llccc}",
        "\\hline",
        "Dataset & Method & 60\\% & 70\\% & 80\\% \\\\",
        "\\hline",
    ]
    for row in rows:
        method = "\\textbf{Ours}" if row["Method"] == "ours" else row["Method"]
        tex_lines.append(f"{row['Dataset']} & {method} & {row['60%']} & {row['70%']} & {row['80%']} \\\\")
    tex_lines.extend(["\\hline", "\\end{tabular}", ""])
    (table_dir / "table_common_targets.tex").write_text("\n".join(tex_lines), encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_height = max(3.0, 0.28 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    ax.axis("off")
    ax.set_title("Communication Rounds to Oracle-Relative Stable Macro-F1", pad=14)
    table = ax.table(
        cellText=[[row["Dataset"], row["Method"], row["60%"], row["70%"], row["80%"]] for row in rows],
        colLabels=["Dataset", "Method", "60%", "70%", "80%"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.18)
    for row_index, row in enumerate(rows, start=1):
        if row["Method"] == "ours":
            for col in range(5):
                table[(row_index, col)].set_text_props(weight="bold")
    fig.tight_layout()
    fig.savefig(table_dir / "table_common_targets.png", dpi=220, bbox_inches="tight")
    fig.savefig(table_dir / "table_common_targets.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate C2 curve-mode common Oracle target metrics.")
    parser.add_argument("--manifest", required=True, help="Explicit v2 run manifest JSON; no recursive result globbing is used.")
    parser.add_argument("--output-root", default="results/c2_v2_common_oracle")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_root = (ROOT / args.output_root).resolve()
    common_dir = output_root / "common_targets"
    manifest_records = load_v2_run_manifest(ROOT / args.manifest)
    payload = aggregate_common_targets(manifest_records)
    write_json(common_dir / "records.json", {"records": payload["records"]})
    write_json(common_dir / "summary.json", payload["summary"])
    write_records_csv(common_dir / "records.csv", payload["records"])
    write_summary_csv(common_dir / "summary.csv", payload["summary"])
    write_table_artifacts(output_root / "tables", payload["summary"])
    print(f"common-target convergence aggregation finished: records={len(payload['records'])}")


if __name__ == "__main__":
    main()
