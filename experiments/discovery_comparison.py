# RQ1 单次 discovery comparison 入口，比较 adaptive ISODATA 与 KMeans 聚类结果。
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from MSL.discovery import run_kmeans
from experiments.common import (
    DISCOVERY_METHODS,
    add_common_run_args,
    assert_no_mock_pipeline_imports,
    find_clients_dir,
    find_discovery_dir,
    load_fingerprint_npz,
    project_root,
    protocol_hash,
    resolved_cfg,
    formal_result_dir,
    runtime_metadata,
    run_type_metadata,
    write_json,
)
from MSL.evaluation import discovery_metrics


# 将 client assignment 写为 csv，供 KMeans-SL baseline 复用。
def write_assignment_csv(path: Path, client_ids, labels, column: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["client_id", column])
        writer.writeheader()
        for client_id, label in zip(client_ids, labels):
            writer.writerow({"client_id": str(client_id), column: int(label)})


# 运行单个 discovery comparison 方法并保存原始结果。
def run_one(dataset: str, fold: int | None, seed: int, method: str, results_root: Path) -> dict:
    assert_no_mock_pipeline_imports()
    root = project_root()
    cfg = resolved_cfg(dataset, fold, seed)
    clients_dir = find_clients_dir(root, cfg)
    adaptive_dir = find_discovery_dir(root, clients_dir, "adaptive_isodata")
    payload = load_fingerprint_npz(adaptive_dir)
    fingerprints = payload["fingerprints"]
    client_ids = payload["client_ids"]
    true = payload["true_cluster"].astype(int)

    if method == "adaptive_isodata":
        pred = payload["pred_cluster"].astype(int)
    elif method.startswith("kmeans"):
        k = int(method.replace("kmeans", ""))
        pred = run_kmeans(fingerprints, k, seed=seed)
    else:
        raise ValueError(f"Unsupported discovery method: {method}")

    metrics = discovery_metrics(true, pred)
    cluster_sizes = {str(int(label)): int((pred == label).sum()) for label in sorted(set(pred.tolist()))}
    raw = {
        **runtime_metadata(root, dataset, fold, seed, method),
        "protocol_hash": protocol_hash(),
        **run_type_metadata(results_root),
        "fingerprint": "reused_discovery_fingerprints",
        "clients_dir": str(clients_dir),
        "adaptive_discovery_dir": str(adaptive_dir),
        "M": int(metrics["true_Q"]),
        "Q_hat": int(metrics["estimated_Q"]),
        "ARI": float(metrics["ARI"]),
        "NMI": float(metrics["NMI"]),
        "Purity": metrics["pred_cluster_purity"],
        "hungarian_ACC": float(metrics["hungarian_ACC"]),
        "cluster_count_error": int(metrics["abs_Q_error"]),
        "cluster_assignments": {str(client_id): int(label) for client_id, label in zip(client_ids, pred)},
        "cluster_sizes": cluster_sizes,
        "diagnostics": metrics,
        "status": "success",
    }
    run_dir = formal_result_dir(results_root, "discovery", dataset, method, fold, seed)
    raw_path = run_dir / "discovery_result.json"
    write_json(raw_path, raw)

    if method.startswith("kmeans"):
        artifact_dir = run_dir / "artifacts"
        write_assignment_csv(artifact_dir / "pred_cluster.csv", client_ids, pred, "pred_cluster")
        write_assignment_csv(artifact_dir / "true_cluster.csv", client_ids, true, "true_cluster")
        raw["artifact_dir"] = str(artifact_dir)
        write_json(raw_path, raw)
    raw["run_dir"] = str(run_dir)
    write_json(raw_path, raw)
    return raw


# 解析 discovery comparison 单次运行参数。
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run discovery comparison on existing real pipeline artifacts.")
    add_common_run_args(parser)
    parser.add_argument("--method", choices=DISCOVERY_METHODS, default="adaptive_isodata")
    return parser.parse_args(argv)


# 执行单次 discovery comparison。
def main(argv=None):
    args = parse_args(argv)
    result = run_one(
        args.dataset,
        args.fold,
        int(args.seed),
        args.method,
        (ROOT / args.results_root).resolve(),
    )
    print(f"Discovery {args.method} finished: status={result['status']} Q_hat={result['Q_hat']}")


if __name__ == "__main__":
    main()
