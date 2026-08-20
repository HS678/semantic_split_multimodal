# RQ2 training-round budget sensitivity 实验入口和聚合分析。
import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.common import (
    build_protocol_manifest,
    find_clients_dir,
    find_discovery_dir,
    formal_run_grid,
    fold_result_component,
    project_root,
    resolved_cfg,
    runtime_metadata,
    seed_result_component,
    stable_config_hash,
    write_json,
)
from experiments.training import (
    configure_method,
    prepare_method_topology,
    resolve_method_policy,
    summarize_train_log,
)
from MSL.training import train_msl_split_learning
from MSL.utils import safe_result_component, select_device, set_seed


SENSITIVITY_PROTOCOL_VERSION = "round_budget_sensitivity_multidataset_v1_2026_08_19"
DEFAULT_DATASET = "mhealth"
DEFAULT_DATASETS = ("mhealth",)
ALL_DATASETS = ("uci_har", "mhealth", "pamap2", "iemocap")
DEFAULT_METHODS = ("ours", "randomsl", "oracle")
DATASET_ROUND_BUDGETS = {
    "uci_har": (25, 50, 100, 150, 200),
    "mhealth": (25, 50, 100, 150, 200),
    "pamap2": (25, 50, 100, 150, 200, 250, 300),
    "iemocap": (25, 50, 100, 150, 200, 250, 300),
}
DEFAULT_ROUND_BUDGETS = DATASET_ROUND_BUDGETS[DEFAULT_DATASET]


# 读取已完成正式 RQ2 写出的 protocol manifest；缺失时才从当前代码构建。
def load_formal_protocol_manifest() -> dict:
    path = ROOT / "results" / "protocol_manifest.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return build_protocol_manifest(None)


# 返回 sensitivity 单次 run 的隔离结果目录。
def sensitivity_run_dir(results_root: Path, dataset: str, method: str, round_budget: int, fold: int | None, seed: int) -> Path:
    return (
        Path(results_root)
        / safe_result_component(dataset)
        / safe_result_component(method)
        / f"rounds_{int(round_budget)}"
        / fold_result_component(fold)
        / seed_result_component(seed)
    )


# 构建本 sensitivity 实验的独立 manifest。
def build_sensitivity_manifest(
    *,
    results_root: Path | None,
    datasets,
    methods,
    dataset_round_budgets: dict[str, list[int]],
    run_grids: dict[str, list[tuple[int | None, int]]],
) -> dict:
    formal_manifest = load_formal_protocol_manifest()
    dataset_payload = {}
    for dataset in datasets:
        grid = run_grids[str(dataset)]
        dataset_cfg = resolved_cfg(str(dataset), grid[0][0], grid[0][1])
        dataset_payload[str(dataset)] = {
            "run_grid": [
                {"fold": fold, "seed": int(seed)}
                for fold, seed in grid
            ],
            "round_budgets": [int(value) for value in dataset_round_budgets[str(dataset)]],
            "clients_per_round": int(dataset_cfg["training"]["clients_per_round"]),
            "formal_global_rounds": int(dataset_cfg["training"]["global_rounds"]),
            "local_steps": int(dataset_cfg["training"]["local_steps"]),
        }
    payload = {
        "sensitivity_protocol_version": SENSITIVITY_PROTOCOL_VERSION,
        "formal_protocol_version": formal_manifest["protocol_version"],
        "formal_protocol_hash": formal_manifest["protocol_hash"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "results_root": None if results_root is None else str(Path(results_root)),
        "datasets": dataset_payload,
        "methods": [str(method) for method in methods],
        "budget_axis": "global_rounds",
        "independent_runs": True,
        "test_policy": "test_once_after_each_independent_run",
        "fixed_training_parameters": {
            "model": "from formal RQ2 protocol",
            "optimizer": "from formal RQ2 protocol",
            "learning_rate": "from formal RQ2 protocol",
            "local_steps": "from formal RQ2 protocol per dataset",
            "partition": "from formal RQ2 artifacts",
            "discovery": "from formal RQ2 adaptive_isodata artifacts",
            "clients_per_round": "from formal RQ2 protocol per dataset",
        },
    }
    payload["sensitivity_protocol_hash"] = sensitivity_protocol_hash(payload)
    return payload


# 对 sensitivity manifest 计算稳定 hash，忽略输出位置和时间戳。
def sensitivity_protocol_hash(manifest: dict) -> str:
    payload = dict(manifest)
    payload.pop("timestamp", None)
    payload.pop("results_root", None)
    payload.pop("sensitivity_protocol_hash", None)
    return stable_config_hash(payload)


# 计算 sensitivity 单次 run 的配置 hash，用于 resume 校验。
def expected_sensitivity_config_hash(
    dataset: str,
    fold: int | None,
    seed: int,
    method: str,
    round_budget: int,
    results_root: Path,
) -> str:
    root = project_root()
    cfg = resolved_cfg(dataset, fold, seed)
    clients_dir = find_clients_dir(root, cfg)
    adaptive_dir = find_discovery_dir(root, clients_dir, "adaptive_isodata")
    policy = resolve_method_policy(method)
    run_dir = sensitivity_run_dir(results_root, dataset, method, round_budget, fold, seed)
    topology_dir = run_dir / "topology"
    cluster_dir = topology_dir if policy.method == "ours" or policy.k is not None else adaptive_dir
    assignment_source = policy.assignment_source
    ckpt_dir = run_dir / "checkpoints"
    cfg = configure_method(
        cfg,
        method,
        clients_dir,
        cluster_dir,
        assignment_source,
        run_dir,
        ckpt_dir,
        int(round_budget),
    )
    return stable_config_hash(
        {
            "experiment": "round_budget_sensitivity",
            "dataset": dataset,
            "fold": fold,
            "seed": int(seed),
            "method": method,
            "round_budget": int(round_budget),
            "clients_per_round": int(cfg["training"]["clients_per_round"]),
            "cfg": cfg,
        }
    )


# 读取可复用的已有 sensitivity result。
def load_existing_result(results_root: Path, dataset: str, method: str, round_budget: int, fold: int | None, seed: int, retry_failed: bool):
    run_dir = sensitivity_run_dir(results_root, dataset, method, round_budget, fold, seed)
    success_path = run_dir / "result.json"
    failed_path = run_dir / "failed_run.json"
    path = success_path if success_path.exists() else failed_path
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("status") == "failed" and retry_failed:
        return None
    if payload.get("dataset") != dataset or payload.get("method") != method:
        return None
    if payload.get("fold") != fold or int(payload.get("seed")) != int(seed):
        return None
    if int(payload.get("round_budget", -1)) != int(round_budget):
        return None
    expected_hash = expected_sensitivity_config_hash(dataset, fold, seed, method, round_budget, results_root)
    if payload.get("config_hash") != expected_hash:
        return None
    if payload.get("formal_protocol_hash") != load_formal_protocol_manifest().get("protocol_hash"):
        return None
    return payload


# 运行一个独立 round-budget sensitivity training run。
def run_one_sensitivity(
    *,
    dataset: str,
    fold: int | None,
    seed: int,
    method: str,
    round_budget: int,
    results_root: Path,
    device_name: str,
    sensitivity_manifest: dict,
) -> dict:
    root = project_root()
    cfg = resolved_cfg(dataset, fold, seed)
    clients_dir = find_clients_dir(root, cfg)
    adaptive_dir = find_discovery_dir(root, clients_dir, "adaptive_isodata")
    clients_per_round = int(cfg["training"]["clients_per_round"])
    run_dir = sensitivity_run_dir(results_root, dataset, method, round_budget, fold, seed)
    topology_dir = run_dir / "topology"
    cluster_dir, assignment_source, feasibility_metadata = prepare_method_topology(
        method,
        clients_dir,
        adaptive_dir,
        topology_dir,
        seed,
        clients_per_round,
    )
    ckpt_dir = run_dir / "checkpoints"
    cfg = configure_method(
        cfg,
        method,
        clients_dir,
        cluster_dir,
        assignment_source,
        run_dir,
        ckpt_dir,
        int(round_budget),
    )
    config_hash = stable_config_hash(
        {
            "experiment": "round_budget_sensitivity",
            "dataset": dataset,
            "fold": fold,
            "seed": int(seed),
            "method": method,
            "round_budget": int(round_budget),
            "clients_per_round": clients_per_round,
            "cfg": cfg,
        }
    )
    metadata = runtime_metadata(root, dataset, fold, seed, method)
    set_seed(seed)
    device = select_device(device_name)
    start = time.time()
    try:
        metrics = train_msl_split_learning(cfg, root, device)
        round_summary = summarize_train_log(
            run_dir / "train_log.csv",
            int(cfg.get("binding", {}).get("batch_size", cfg.get("training", {}).get("batch_size", 1))),
        )
        payload = {
            **metadata,
            "status": "success",
            "experiment": "round_budget_sensitivity",
            "sensitivity_protocol_version": sensitivity_manifest["sensitivity_protocol_version"],
            "sensitivity_protocol_hash": sensitivity_manifest["sensitivity_protocol_hash"],
            "protocol_version": sensitivity_manifest["formal_protocol_version"],
            "protocol_hash": sensitivity_manifest["formal_protocol_hash"],
            "formal_protocol_version": sensitivity_manifest["formal_protocol_version"],
            "formal_protocol_hash": sensitivity_manifest["formal_protocol_hash"],
            "round_budget": int(round_budget),
            "clients_per_round": int(clients_per_round),
            "client_budget_policy": "fixed_total_clients_per_round",
            "test_policy": "test_once_after_independent_training_run",
            "Q_hat": int(metrics.get("estimated_num_clusters", 0)),
            **feasibility_metadata,
            **round_summary,
            "accuracy": metrics.get("test_accuracy"),
            "macro_f1": metrics.get("test_macro_f1"),
            "weighted_f1": metrics.get("test_weighted_f1"),
            "final_loss": metrics.get("test_loss"),
            "test_classification_loss": metrics.get("test_classification_loss"),
            "modality_full_coverage_rate": metrics.get("modality_full_coverage_rate"),
            "modality_coverage_mean": metrics.get("modality_coverage_mean"),
            "runtime_seconds": float(time.time() - start),
            "config_hash": config_hash,
            "device": str(device),
            "config_snapshot": cfg,
            "metrics": metrics,
            "run_dir": str(run_dir),
            "checkpoint_dir": str(ckpt_dir),
            "curve_path": str(run_dir / "train_log.csv"),
            "cluster_metadata_path": str(Path(cluster_dir) / "feasibility_metadata.json"),
        }
    except Exception as exc:
        payload = {
            **metadata,
            "status": "failed",
            "experiment": "round_budget_sensitivity",
            "sensitivity_protocol_version": sensitivity_manifest["sensitivity_protocol_version"],
            "sensitivity_protocol_hash": sensitivity_manifest["sensitivity_protocol_hash"],
            "protocol_version": sensitivity_manifest["formal_protocol_version"],
            "protocol_hash": sensitivity_manifest["formal_protocol_hash"],
            "formal_protocol_version": sensitivity_manifest["formal_protocol_version"],
            "formal_protocol_hash": sensitivity_manifest["formal_protocol_hash"],
            "round_budget": int(round_budget),
            "clients_per_round": int(clients_per_round),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            **feasibility_metadata,
            "config_hash": config_hash,
            "runtime_seconds": float(time.time() - start),
            "run_dir": str(run_dir),
            "checkpoint_dir": str(ckpt_dir),
        }
        write_json(run_dir / "failed_run.json", payload)
        return payload
    write_json(run_dir / "result.json", payload)
    return payload


# 安全取单个 record 的最终测试指标。
def _metric(record: dict, name: str):
    if record.get(name) is not None:
        return record.get(name)
    metrics = record.get("metrics") or {}
    if name == "accuracy":
        return metrics.get("test_accuracy")
    if name == "macro_f1":
        return metrics.get("test_macro_f1")
    if name == "weighted_f1":
        return metrics.get("test_weighted_f1")
    if name == "final_loss":
        return metrics.get("test_loss")
    return metrics.get(name)


# 对 sensitivity records 按 dataset/method/round_budget 聚合。
def aggregate(records: list[dict]) -> list[dict]:
    groups = {}
    for record in records:
        key = (record.get("dataset"), record.get("method"), int(record.get("round_budget", -1)))
        groups.setdefault(key, []).append(record)
    rows = []
    for (dataset, method, round_budget), group in sorted(groups.items()):
        success = [row for row in group if row.get("status") == "success"]
        item = {
            "dataset": dataset,
            "method": method,
            "round_budget": int(round_budget),
            "count": int(len(success)),
            "failed": int(len(group) - len(success)),
        }
        for metric in [
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "final_loss",
            "modality_full_coverage_rate",
            "modality_coverage_mean",
        ]:
            values = [float(_metric(row, metric)) for row in success if _metric(row, metric) is not None]
            item[f"{metric}_mean"] = float(statistics.mean(values)) if values else None
            item[f"{metric}_std"] = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
        item["modality_full_coverage_mean"] = item["modality_full_coverage_rate_mean"]
        item["modality_full_coverage_std"] = item["modality_full_coverage_rate_std"]
        rows.append(item)
    return rows


# 计算 performance-vs-budget 的 normalized trapezoidal AUC。
def budget_auc(summary_rows: list[dict]) -> list[dict]:
    groups = {}
    for row in summary_rows:
        groups.setdefault((row["dataset"], row["method"]), []).append(row)
    out = []
    for (dataset, method), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda item: int(item["round_budget"]))
        item = {"dataset": dataset, "method": method}
        for metric in ["accuracy", "macro_f1"]:
            points = [
                (float(row["round_budget"]), row.get(f"{metric}_mean"))
                for row in rows
                if row.get(f"{metric}_mean") is not None
            ]
            if len(points) < 2:
                item[f"{metric}_budget_auc"] = None
                continue
            area = 0.0
            for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
                area += (x1 - x0) * (float(y0) + float(y1)) / 2.0
            item[f"{metric}_budget_auc"] = float(area / max(1.0, points[-1][0] - points[0][0]))
        out.append(item)
    return out


# 计算离散 rounds-to-target。
def rounds_to_target(summary_rows: list[dict], target_accuracy: float | None) -> list[dict]:
    if target_accuracy is None:
        return []
    groups = {}
    for row in summary_rows:
        groups.setdefault((row["dataset"], row["method"]), []).append(row)
    out = []
    for (dataset, method), rows in sorted(groups.items()):
        reached = "not_reached"
        for row in sorted(rows, key=lambda item: int(item["round_budget"])):
            value = row.get("accuracy_mean")
            if value is not None and float(value) >= float(target_accuracy):
                reached = int(row["round_budget"])
                break
        out.append(
            {
                "dataset": dataset,
                "method": method,
                "target_accuracy": float(target_accuracy),
                "rounds_to_target": reached,
            }
        )
    return out


# 写 CSV rows，字段按所有行 union 稳定排序。
def write_rows_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    preferred = ["dataset", "method", "round_budget", "count", "failed"]
    fieldnames = [key for key in preferred if key in fieldnames] + [
        key for key in fieldnames if key not in preferred
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# 解析最终要运行的数据集列表，保留 --dataset 单数据集兼容入口。
def resolve_datasets(dataset: str | None, datasets) -> list[str]:
    selected = list(datasets) if datasets else [dataset or DEFAULT_DATASET]
    out = []
    for value in selected:
        key = str(value).strip().lower()
        if key not in ALL_DATASETS:
            raise ValueError(f"Unsupported sensitivity dataset: {value!r}.")
        if key not in out:
            out.append(key)
    return out


# 返回每个数据集的 round budget 列表。
def resolve_dataset_round_budgets(datasets, round_budget_override) -> dict[str, list[int]]:
    if round_budget_override:
        budgets = [int(value) for value in round_budget_override]
        return {str(dataset): budgets for dataset in datasets}
    return {
        str(dataset): [int(value) for value in DATASET_ROUND_BUDGETS[str(dataset)]]
        for dataset in datasets
    }


# 打印 sensitivity run plan。
def print_plan(datasets, methods, dataset_round_budgets, run_grids) -> None:
    print("round-budget Sensitivity Experiment Plan")
    print("=======================================")
    print(f"methods: {list(methods)}")
    total = 0
    for dataset in datasets:
        run_grid = run_grids[str(dataset)]
        round_budgets = dataset_round_budgets[str(dataset)]
        runs = len(methods) * len(round_budgets) * len(run_grid)
        total += runs
        print(f"dataset: {dataset}")
        print(f"round_budgets: {[int(value) for value in round_budgets]}")
        print(f"folds: {sorted({fold for fold, _ in run_grid}, key=lambda value: -1 if value is None else int(value))}")
        print(f"seeds: {sorted({int(seed) for _, seed in run_grid})}")
        print(f"runs: {runs}")
        print("")
    print(f"TOTAL EXPECTED RUNS: {total}")


# 根据参数解析正式 fold/seed grid。
def build_run_grid(dataset: str, seeds, folds):
    base = formal_run_grid(dataset, seeds)
    if folds is None:
        return base
    wanted = {int(fold) for fold in folds}
    return [(fold, seed) for fold, seed in base if fold is not None and int(fold) in wanted]


# 检查 CUDA 是否可用。
def require_cuda_if_requested(require_cuda: bool) -> None:
    if bool(require_cuda) and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by --require-cuda, but torch.cuda.is_available() is false.")


# 解析 sensitivity CLI。
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run RQ2 training-round budget sensitivity experiments.")
    parser.add_argument("--dataset", choices=ALL_DATASETS, default=DEFAULT_DATASET)
    parser.add_argument("--datasets", nargs="*", choices=ALL_DATASETS)
    parser.add_argument("--methods", nargs="*", choices=DEFAULT_METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument("--round-budgets", nargs="*", type=int)
    parser.add_argument("--folds", nargs="*", type=int)
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--results-root", default="results/sensitivity/round_budget")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--target-accuracy", type=float)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args(argv)


# 执行全部 sensitivity runs 并写出 summary/AUC/target 文件。
def main(argv=None):
    args = parse_args(argv)
    require_cuda_if_requested(args.require_cuda)
    datasets = resolve_datasets(args.dataset, args.datasets)
    dataset_round_budgets = resolve_dataset_round_budgets(datasets, args.round_budgets)
    all_budgets = [value for budgets in dataset_round_budgets.values() for value in budgets]
    if any(int(value) <= 0 for value in all_budgets):
        raise ValueError("--round-budgets must be positive integers.")
    results_root = (ROOT / args.results_root).resolve()
    run_grids = {
        dataset: build_run_grid(dataset, args.seeds, args.folds)
        for dataset in datasets
    }
    empty = [dataset for dataset, grid in run_grids.items() if not grid]
    if empty:
        raise ValueError(f"No runs selected by dataset/folds/seeds for datasets: {empty}.")
    manifest = build_sensitivity_manifest(
        results_root=results_root,
        datasets=datasets,
        methods=args.methods,
        dataset_round_budgets=dataset_round_budgets,
        run_grids=run_grids,
    )
    print_plan(datasets, args.methods, dataset_round_budgets, run_grids)
    if args.plan_only:
        print(f"sensitivity_protocol_hash={manifest['sensitivity_protocol_hash']}")
        return
    write_json(results_root / "sensitivity_manifest.json", manifest)
    records = []
    for dataset in datasets:
        for method in args.methods:
            for round_budget in dataset_round_budgets[dataset]:
                for fold, seed in run_grids[dataset]:
                    record = load_existing_result(
                        results_root,
                        dataset,
                        method,
                        int(round_budget),
                        fold,
                        int(seed),
                        args.retry_failed,
                    )
                    if record is None:
                        record = run_one_sensitivity(
                            dataset=dataset,
                            fold=fold,
                            seed=int(seed),
                            method=method,
                            round_budget=int(round_budget),
                            results_root=results_root,
                            device_name=args.device,
                            sensitivity_manifest=manifest,
                        )
                    records.append(record)
    summary = aggregate(records)
    auc_rows = budget_auc(summary)
    target_rows = rounds_to_target(summary, args.target_accuracy)
    write_json(results_root / "runs.json", {"runs": records})
    write_json(results_root / "summary.json", {"rows": summary})
    write_rows_csv(results_root / "summary.csv", summary)
    write_json(results_root / "budget_auc.json", {"rows": auc_rows})
    write_rows_csv(results_root / "budget_auc.csv", auc_rows)
    if target_rows:
        write_json(results_root / "rounds_to_target.json", {"rows": target_rows})
        write_rows_csv(results_root / "rounds_to_target.csv", target_rows)
    expected = sum(
        len(args.methods) * len(dataset_round_budgets[dataset]) * len(run_grids[dataset])
        for dataset in datasets
    )
    failed = sum(1 for row in records if row.get("status") == "failed")
    print(
        "round-budget sensitivity finished: "
        f"expected={expected} total={len(records)} success={len(records) - failed} failed={failed}"
    )
    print(f"results_root={results_root}")
    print(f"sensitivity_protocol_hash={manifest['sensitivity_protocol_hash']}")
    if len(records) != expected:
        raise RuntimeError(f"sensitivity runner missed runs: expected={expected}, actual={len(records)}")


if __name__ == "__main__":
    main()
