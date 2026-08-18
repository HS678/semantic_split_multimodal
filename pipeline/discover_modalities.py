import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "MSL").is_dir():
            return parent
    raise RuntimeError("Cannot locate project root containing src/MSL.")


ROOT = _project_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from MSL.pretrain import discover_modalities
from MSL.utils import select_device
from MSL.protocol import (
    add_experiment_args,
    load_experiment_config_from_args,
    print_resolved_config,
    save_resolved_config_artifact,
    modality_discovery_config_snapshot,
)
from MSL.utils import dataset_result_name, resolve_pipeline_paths, safe_result_component
from MSL.utils import set_seed


def _resolve(path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _cluster_method_name(cfg: dict) -> str:
    method = str(cfg.get("cluster", {}).get("method", "adaptive_isodata")).lower()
    if method == "adaptive":
        method = "adaptive_isodata"
    if method not in {"kmeans", "adaptive_isodata"}:
        raise ValueError("cluster.method must be 'kmeans' or 'adaptive_isodata'.")
    return safe_result_component(method)


def build_discovery_run(cfg: dict, clients_dir, output_root):
    partition_dir = _resolve(clients_dir)
    if not partition_dir.exists():
        raise FileNotFoundError(f"Missing client preparation directory: {partition_dir}")
    if not (partition_dir / "train_clients").exists():
        raise FileNotFoundError(f"Missing train_clients under client preparation directory: {partition_dir}")

    output_root_path = _resolve(output_root)
    dataset_name = safe_result_component(dataset_result_name(cfg))
    partition_name = safe_result_component(partition_dir.name)
    method_name = _cluster_method_name(cfg)
    cluster_dir = (output_root_path / dataset_name / partition_name / method_name).resolve()
    if cluster_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing modality discovery output directory: {cluster_dir}")

    run_cfg = dict(cfg)
    run_cfg["partition"] = {**run_cfg.get("partition", {}), "output_dir": str(partition_dir)}
    run_cfg["cluster"] = {**run_cfg.get("cluster", {}), "output_dir": str(cluster_dir), "method": method_name}
    run_cfg["result"] = {**run_cfg.get("result", {}), "output_dir": str(cluster_dir)}
    run_cfg["discovery"] = {
        "clients_dir": str(partition_dir),
        "output_root": str(output_root_path),
        "cluster_dir": str(cluster_dir),
        "dataset": dataset_name,
        "partition_signature": partition_name,
        "cluster_method": method_name,
    }
    return run_cfg, {
        "partition_dir": partition_dir,
        "output_root": output_root_path,
        "cluster_dir": cluster_dir,
        "dataset": dataset_name,
        "partition_signature": partition_name,
        "cluster_method": method_name,
    }


def _write_metadata(paths: dict, cfg: dict, metrics: dict | None, runtime_seconds: float):
    paths["cluster_dir"].mkdir(parents=True, exist_ok=True)
    config_snapshot = modality_discovery_config_snapshot(cfg)
    metadata = {
        "pipeline_step": "modality_discovery",
        "dataset": paths["dataset"],
        "partition_signature": paths["partition_signature"],
        "cluster_method": paths["cluster_method"],
        "clients_dir": str(paths["partition_dir"]),
        "output_root": str(paths["output_root"]),
        "cluster_dir": str(paths["cluster_dir"]),
        "git_commit": _git_commit(),
        "runtime_seconds": float(runtime_seconds),
        "seed": int(cfg.get("seed", 42)),
        "config_snapshot": config_snapshot,
        "metrics": metrics,
    }
    with (paths["cluster_dir"] / "discovery_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Modality discovery: discover modality clusters from a frozen client preparation partition.")
    add_experiment_args(parser, include_seed=True)
    parser.add_argument("--clients-dir", help="Optional override for discovery.clients_dir")
    parser.add_argument("--output-root", help="Optional override for discovery.output_root")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_experiment_config_from_args(args)
    if args.print_config:
        print_resolved_config(cfg)
        return
    discovery_cfg = cfg.get("discovery", {})
    clients_dir = args.clients_dir or discovery_cfg.get("clients_dir")
    output_root = args.output_root or discovery_cfg.get("output_root")
    if not clients_dir or not output_root:
        # 新格式 config 不写路径：自动从 base_dir + 数据集 + 协议生成。
        resolved = resolve_pipeline_paths(cfg, ROOT)
        clients_dir = clients_dir or resolved["clients_dir"]
        output_root = output_root or resolved["discovery_dir"].parents[2]
    cfg, paths = build_discovery_run(
        cfg,
        clients_dir=clients_dir,
        output_root=output_root,
    )
    paths["cluster_dir"].mkdir(parents=True, exist_ok=True)
    save_resolved_config_artifact(modality_discovery_config_snapshot(cfg), paths["cluster_dir"])

    start = time.time()
    set_seed(int(cfg.get("seed", 42)))
    device = select_device(cfg.get("device", "auto"))
    metrics = discover_modalities(cfg, ROOT, device)
    runtime_seconds = time.time() - start
    _write_metadata(paths, cfg, metrics, runtime_seconds)

    print("Modality discovery finished.")
    print(f"clients_dir={paths['partition_dir']}")
    print(f"cluster_dir={paths['cluster_dir']}")
    print(f"estimated_Q={metrics['estimated_Q']}")
    print(f"abs_Q_error={metrics['abs_Q_error']}")
    print(f"discovery_status={metrics['discovery_status']}")
    print(f"ACC={metrics['ACC']:.6f}, NMI={metrics['NMI']:.6f}, ARI={metrics['ARI']:.6f}")


if __name__ == "__main__":
    main()
