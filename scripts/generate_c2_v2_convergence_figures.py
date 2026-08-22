#!/usr/bin/env python
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = ROOT / "results" / "c2_v2_common_oracle"
OUT_DIR = V2_ROOT / "convergence_figures"

ALL_METHODS = [
    "randomsl",
    "kmeans2",
    "kmeans3",
    "kmeans4",
    "kmeans5",
    "auto_kmeans",
    "gmm_bic",
    "ours",
    "oracle",
]
MAIN_METHODS = ["randomsl", "auto_kmeans", "gmm_bic", "ours", "oracle"]
DATASETS = ["uci_har", "mhealth", "pamap2", "iemocap"]
DISPLAY = {
    "randomsl": "RandomSL",
    "kmeans2": "KMeans-2",
    "kmeans3": "KMeans-3",
    "kmeans4": "KMeans-4",
    "kmeans5": "KMeans-5",
    "auto_kmeans": "Auto-KMeans",
    "gmm_bic": "GMM+BIC",
    "ours": "Ours",
    "oracle": "Oracle-SL",
}
DATASET_DISPLAY = {
    "uci_har": "UCI-HAR",
    "mhealth": "MHEALTH",
    "pamap2": "PAMAP2",
    "iemocap": "IEMOCAP",
}
COLORS = {
    "randomsl": "#F1C40F",
    "kmeans2": "#7B3294",
    "kmeans3": "#8C564B",
    "kmeans4": "#E377C2",
    "kmeans5": "#17BECF",
    "auto_kmeans": "#1F77B4",
    "gmm_bic": "#2CA02C",
    "ours": "#F28E2B",
    "oracle": "#D62728",
}
LINESTYLES = {
    method: "-"
    for method in ALL_METHODS
}
TARGET_LINE_STYLES = {
    "60": {"color": "#777777", "linestyle": ":", "label": "60% target"},
    "70": {"color": "#555555", "linestyle": "--", "label": "70% target"},
    "80": {"color": "#333333", "linestyle": "-.", "label": "80% target"},
}


def _mean(values):
    return sum(values) / len(values)


def _std(values):
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_curve(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [
            {
                "round": int(float(row["round"])),
                "test_macro_f1": float(row["test_macro_f1"]),
            }
            for row in csv.DictReader(handle)
        ]


def _curve_manifest_records():
    manifest = _read_json(V2_ROOT / "manifest.json")
    records = [
        row
        for row in manifest["runs"]
        if row["evaluation_mode"] == "curve"
    ]
    return records


def _common_target_records():
    with (V2_ROOT / "common_targets" / "records.csv").open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _run_key(row):
    return (row["dataset"], row.get("fold"), int(row["seed"]))


def build_curve_summary():
    manifest_records = _curve_manifest_records()
    expected_methods = set(ALL_METHODS)
    methods_seen = {row["method"] for row in manifest_records}
    if methods_seen != expected_methods:
        raise RuntimeError(f"Unexpected curve methods: seen={sorted(methods_seen)}, expected={sorted(expected_methods)}")

    source_paths = defaultdict(list)
    trajectories = defaultdict(lambda: defaultdict(list))
    run_counts_by_dataset = defaultdict(set)
    for record in manifest_records:
        if "/formal/" in str(record["expected_output_path"]):
            raise RuntimeError(f"Curve manifest points at formal path: {record}")
        curve_path = Path(record["expected_output_path"]) / "test_curve.csv"
        if not curve_path.exists():
            raise FileNotFoundError(f"Missing curve file: {curve_path}")
        rows = _read_curve(curve_path)
        if not rows:
            raise RuntimeError(f"Empty curve file: {curve_path}")
        run_key = _run_key(record)
        run_counts_by_dataset[record["dataset"]].add(run_key)
        source_paths[(record["dataset"], record["method"])].append(str(curve_path))
        for row in rows:
            if math.isnan(row["test_macro_f1"]):
                raise RuntimeError(f"NaN macro-F1 in {curve_path} round={row['round']}")
            trajectories[(record["dataset"], record["method"])][row["round"]].append(row["test_macro_f1"])

    summary_rows = []
    summary_json = {}
    for dataset in DATASETS:
        expected_run_count = len(run_counts_by_dataset[dataset])
        if expected_run_count <= 0:
            raise RuntimeError(f"No curve runs for dataset={dataset}")
        for method in ALL_METHODS:
            per_round = trajectories[(dataset, method)]
            if not per_round:
                raise RuntimeError(f"Missing trajectory for dataset={dataset}, method={method}")
            items = []
            for round_idx in sorted(per_round):
                values = per_round[round_idx]
                if len(values) != expected_run_count:
                    raise RuntimeError(
                        f"Unexpected run count for {dataset}/{method}/round {round_idx}: "
                        f"{len(values)} != {expected_run_count}"
                    )
                item = {
                    "dataset": dataset,
                    "method": method,
                    "round": int(round_idx),
                    "test_macro_f1_mean": float(_mean(values)),
                    "test_macro_f1_std": float(_std(values)),
                    "number_of_runs": int(len(values)),
                }
                summary_rows.append(item)
                items.append(item)
            summary_json[f"{dataset}_{method}"] = {
                "source_test_curve_paths": source_paths[(dataset, method)],
                "rounds": items,
            }
    return summary_rows, summary_json, {key: len(value) for key, value in run_counts_by_dataset.items()}


def target_means():
    records = _common_target_records()
    targets = defaultdict(lambda: defaultdict(list))
    source_counts = defaultdict(int)
    for row in records:
        dataset = row["dataset"]
        method = row["method"]
        if method not in ALL_METHODS:
            raise RuntimeError(f"Unexpected method in common target records: {method}")
        if method != "oracle":
            continue
        source_counts[dataset] += 1
        for level in ("60", "70", "80"):
            targets[dataset][level].append(float(row[f"target{level}"]))
    out = {}
    for dataset in DATASETS:
        if source_counts[dataset] <= 0:
            raise RuntimeError(f"No matched oracle target rows for dataset={dataset}")
        out[dataset] = {
            f"target{level}": float(_mean(targets[dataset][level]))
            for level in ("60", "70", "80")
        }
        out[dataset]["source"] = "mean_of_matched_oracle_targets"
        out[dataset]["matched_oracle_target_count"] = int(source_counts[dataset])
    return out


def write_curve_summary(summary_rows, summary_json):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "method", "round", "test_macro_f1_mean", "test_macro_f1_std", "number_of_runs"]
    with (OUT_DIR / "curve_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    with (OUT_DIR / "curve_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_json, handle, indent=2, sort_keys=True)


def _summary_index(summary_rows):
    out = defaultdict(list)
    for row in summary_rows:
        out[(row["dataset"], row["method"])].append(row)
    for key in out:
        out[key] = sorted(out[key], key=lambda row: row["round"])
    return out


def _plot_dataset(
    ax,
    dataset,
    methods,
    summary_idx,
    target_info=None,
    title=None,
    legend=True,
    show_targets=True,
    annotate_targets=False,
):
    for method in methods:
        rows = summary_idx[(dataset, method)]
        rounds = [row["round"] for row in rows]
        means = [row["test_macro_f1_mean"] for row in rows]
        ax.plot(
            rounds,
            means,
            label=DISPLAY[method],
            color=COLORS[method],
            linestyle=LINESTYLES.get(method, "-"),
            linewidth=2.7 if method == "ours" else 2.0,
        )
    if show_targets:
        if target_info is None:
            raise RuntimeError("target_info is required when show_targets=True")
        x_min, x_max = ax.get_xlim()
        x_text = x_max - (x_max - x_min) * 0.02
        for level in ("60", "70", "80"):
            style = TARGET_LINE_STYLES[level]
            value = target_info[f"target{level}"]
            ax.axhline(
                value,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.0,
                alpha=0.85,
            )
            if annotate_targets:
                ax.text(
                    x_text,
                    value,
                    style["label"],
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    color=style["color"],
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
                )
    ax.set_title(title or DATASET_DISPLAY[dataset])
    ax.set_xlabel("Communication Rounds")
    ax.set_ylabel("Test Macro-F1")
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0.0, 1.0)


def write_figures(summary_rows, targets):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/MSL_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_idx = _summary_index(summary_rows)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), sharey=True)
    for ax, dataset in zip(axes.reshape(-1), DATASETS):
        _plot_dataset(ax, dataset, MAIN_METHODS, summary_idx, targets[dataset], title=dataset, legend=False)
    handles, labels = axes.reshape(-1)[0].get_legend_handles_labels()
    # Build one clean legend containing methods and target-line labels.
    line_handles = []
    line_labels = []
    for method in MAIN_METHODS:
        line_handles.append(plt.Line2D([0], [0], color=COLORS[method], linestyle=LINESTYLES.get(method, "-"), linewidth=2.0))
        line_labels.append(DISPLAY[method])
    for level, style in [("60", ":"), ("70", "--"), ("80", "-.")]:
        line_handles.append(plt.Line2D([0], [0], color="#333333", linestyle=style, linewidth=1.0))
        line_labels.append(f"{level}% Oracle target")
    fig.legend(line_handles, line_labels, loc="lower center", ncol=4, frameon=True)
    fig.suptitle("C2 Convergence: Communication Rounds to Oracle-Relative Stable Macro-F1", y=0.995)
    fig.tight_layout(rect=(0, 0.08, 1, 0.965))
    fig.savefig(OUT_DIR / "fig3_convergence_main.png", dpi=240)
    fig.savefig(OUT_DIR / "fig3_convergence_main.pdf")
    plt.close(fig)

    for dataset in DATASETS:
        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        _plot_dataset(ax, dataset, ALL_METHODS, summary_idx, targets[dataset], title=f"{dataset} full C1-to-C2 propagation", legend=False)
        handles = [
            plt.Line2D([0], [0], color=COLORS[method], linestyle=LINESTYLES.get(method, "-"), linewidth=2.0 if method in {"ours", "oracle"} else 1.55)
            for method in ALL_METHODS
        ]
        labels = [DISPLAY[method] for method in ALL_METHODS]
        for level, style in [("60", ":"), ("70", "--"), ("80", "-.")]:
            handles.append(plt.Line2D([0], [0], color="#333333", linestyle=style, linewidth=1.0))
            labels.append(f"{level}% Oracle target")
        ax.legend(handles, labels, loc="lower right", fontsize=8, frameon=True, ncol=2)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"fig3_full_{dataset}.png", dpi=240)
        fig.savefig(OUT_DIR / f"fig3_full_{dataset}.pdf")
        plt.close(fig)


def write_two_figure_spec(summary_rows, targets):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/MSL_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_idx = _summary_index(summary_rows)

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), sharey=True)
    for ax, dataset in zip(axes.reshape(-1), DATASETS):
        _plot_dataset(
            ax,
            dataset,
            MAIN_METHODS,
            summary_idx,
            targets[dataset],
            title=DATASET_DISPLAY[dataset],
            legend=False,
            show_targets=True,
            annotate_targets=True,
        )
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=COLORS[method],
            linestyle="-",
            linewidth=2.7 if method == "ours" else 2.0,
        )
        for method in MAIN_METHODS
    ]
    labels = [DISPLAY[method] for method in MAIN_METHODS]
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=True)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT_DIR / "fig3_convergence_main_v3.png", dpi=240)
    fig.savefig(OUT_DIR / "fig3_convergence_main_v3.pdf")
    plt.close(fig)

    for dataset in DATASETS:
        fig, ax = plt.subplots(figsize=(8.4, 5.2))
        _plot_dataset(
            ax,
            dataset,
            ALL_METHODS,
            summary_idx,
            target_info=None,
            title=DATASET_DISPLAY[dataset],
            legend=False,
            show_targets=False,
            annotate_targets=False,
        )
        handles = [
            plt.Line2D(
                [0],
                [0],
                color=COLORS[method],
                linestyle="-",
                linewidth=2.7 if method == "ours" else 2.0,
            )
            for method in ALL_METHODS
        ]
        labels = [DISPLAY[method] for method in ALL_METHODS]
        ax.legend(handles, labels, loc="lower right", fontsize=8, frameon=True, ncol=2)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"fig3_full_{dataset}_v2.png", dpi=240)
        fig.savefig(OUT_DIR / f"fig3_full_{dataset}_v2.pdf")
        plt.close(fig)


def write_metadata(summary_json, run_counts, targets):
    metadata = {
        "datasets": DATASETS,
        "main_methods": MAIN_METHODS,
        "full_methods": ALL_METHODS,
        "aggregation_unit": "run trajectory by dataset/method/communication_round",
        "rounds": {
            key: [row["round"] for row in value["rounds"]]
            for key, value in summary_json.items()
        },
        "target_source": "results/c2_v2_common_oracle/common_targets/records.csv",
        "visualization_target": "mean_of_matched_oracle_targets",
        "target_values_used_for_visualization": targets,
        "source_test_curve_paths_counts": {
            key: len(value["source_test_curve_paths"])
            for key, value in summary_json.items()
        },
        "dataset_run_counts": run_counts,
        "mean_std_rule": "For each dataset/method/round, compute mean and population std over matched fold/seed run trajectory values.",
        "formal_results_read": False,
        "common_target_records_modified": False,
    }
    with (OUT_DIR / "figure_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return metadata


def write_metadata_v3(summary_json, targets):
    metadata = {
        "figure_a_methods": MAIN_METHODS,
        "figure_b_methods": ALL_METHODS,
        "display_names": DISPLAY,
        "color_map": COLORS,
        "linestyle_map": {method: "-" for method in ALL_METHODS},
        "target_line_styles": TARGET_LINE_STYLES,
        "aggregation_rule": "For each dataset/method/round, aggregate test_macro_f1 over matched fold/seed run trajectories; keep mean, population std, and run_count.",
        "source_curve_summary_path": str(OUT_DIR / "curve_summary.csv"),
        "target_source_for_figure_a": "results/c2_v2_common_oracle/common_targets/records.csv; visualization target is the dataset mean over matched oracle rows only.",
        "target_values_used_for_figure_a": targets,
        "figure_b_targets": "none",
        "std_band": "not drawn; curve_summary still records std values",
        "sanity_spec": {
            "figure_a_panels": 4,
            "figure_a_methods_per_panel": len(MAIN_METHODS),
            "figure_a_targets_per_panel": 3,
            "figure_a_targets_in_legend": False,
            "figure_b_separate_figures": 4,
            "figure_b_methods_per_figure": len(ALL_METHODS),
            "figure_b_targets": False,
        },
        "source_test_curve_paths_counts": {
            key: len(value["source_test_curve_paths"])
            for key, value in summary_json.items()
        },
    }
    with (OUT_DIR / "figure_metadata_v3.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return metadata


def main():
    summary_rows, summary_json, run_counts = build_curve_summary()
    targets = target_means()
    write_curve_summary(summary_rows, summary_json)
    write_two_figure_spec(summary_rows, targets)
    write_metadata(summary_json, run_counts, targets)
    write_metadata_v3(summary_json, targets)
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
