import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "MSL").is_dir():
            return parent
    raise RuntimeError("Cannot locate project root containing src/MSL.")


ROOT = _project_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MSL.pretrain import discover_modalities
from MSL.protocol import DATASET_PROTOCOLS
from MSL.utils import dataset_result_name, safe_result_component
from MSL.utils import select_device, set_seed
from experiments.common import (
    apply_experiment_overrides,
    build_experiment_config,
    find_clients_dir,
    with_repeated_seed_split_signature,
)


def _resolve(path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


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
    clients_dir = _resolve(clients_dir)
    if not clients_dir.exists():
        raise FileNotFoundError(f"Missing client preparation directory: {clients_dir}")
    if not (clients_dir / "train_clients").exists():
        raise FileNotFoundError(f"Missing train_clients under client preparation directory: {clients_dir}")

    output_root = _resolve(output_root)
    dataset_name = safe_result_component(dataset_result_name(cfg))
    partition_name = safe_result_component(clients_dir.name)
    method_name = _cluster_method_name(cfg)
    discovery_dir = (output_root / dataset_name / partition_name / method_name).resolve()
    if discovery_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing modality discovery output directory: {discovery_dir}")

    run_cfg = dict(cfg)
    run_cfg["partition"] = {**run_cfg.get("partition", {}), "output_dir": str(clients_dir)}
    run_cfg["cluster"] = {**run_cfg.get("cluster", {}), "output_dir": str(discovery_dir), "method": method_name}
    run_cfg["result"] = {**run_cfg.get("result", {}), "output_dir": str(discovery_dir)}
    run_cfg["discovery"] = {
        "clients_dir": str(clients_dir),
        "output_root": str(output_root),
        "discovery_dir": str(discovery_dir),
        "dataset": dataset_name,
        "partition_signature": partition_name,
        "cluster_method": method_name,
    }
    return run_cfg, {
        "clients_dir": clients_dir,
        "output_root": output_root,
        "discovery_dir": discovery_dir,
        "dataset": dataset_name,
        "partition_signature": partition_name,
        "cluster_method": method_name,
    }


def _config_snapshot(cfg: dict) -> dict:
    return {
        "config_scope": "modality_discovery",
        "seed": cfg.get("seed"),
        "device": cfg.get("device"),
        "num_classes": cfg.get("num_classes"),
        "dataset": cfg.get("dataset"),
        "partition": cfg.get("partition"),
        "pretrain": cfg.get("pretrain"),
        "fingerprint": cfg.get("fingerprint"),
        "cluster": cfg.get("cluster"),
        "fingerprint_visualization": cfg.get("fingerprint_visualization"),
        "runtime_overrides": cfg.get("runtime_overrides"),
    }


def _write_metadata(paths: dict, cfg: dict, metrics: dict | None, runtime_seconds: float):
    paths["discovery_dir"].mkdir(parents=True, exist_ok=True)
    metadata = {
        "pipeline_step": "modality_discovery",
        "dataset": paths["dataset"],
        "partition_signature": paths["partition_signature"],
        "cluster_method": paths["cluster_method"],
        "clients_dir": str(paths["clients_dir"]),
        "output_root": str(paths["output_root"]),
        "discovery_dir": str(paths["discovery_dir"]),
        "git_commit": _git_commit(),
        "runtime_seconds": float(runtime_seconds),
        "seed": int(cfg.get("seed", 42)),
        "config_snapshot": _config_snapshot(cfg),
        "metrics": metrics,
    }
    with (paths["discovery_dir"] / "discovery_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    with (paths["discovery_dir"] / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(_config_snapshot(cfg), handle, indent=2, ensure_ascii=False, sort_keys=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Discover modality clusters from prepared clients.")
    parser.add_argument("--dataset", choices=tuple(DATASET_PROTOCOLS), required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", default="results/pipeline/discovery")
    parser.add_argument("--fingerprint-type", choices=["encoder", "signal", "hybrid"])
    parser.add_argument("--clients-dir")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = build_experiment_config(dataset_type=args.dataset, seed=args.seed, device=args.device)
    cfg = apply_experiment_overrides(cfg, fold=args.fold)
    if args.fold is None and DATASET_PROTOCOLS[str(args.dataset)]["fold_count"] is None:
        cfg = with_repeated_seed_split_signature(cfg, args.seed)
    if args.fingerprint_type is not None:
        cfg["fingerprint"] = {**dict(cfg.get("fingerprint", {})), "type": str(args.fingerprint_type)}

    clients_dir = _resolve(args.clients_dir) if args.clients_dir else find_clients_dir(ROOT, cfg)
    cfg, paths = build_discovery_run(cfg, clients_dir=clients_dir, output_root=args.output_root)

    start = time.time()
    set_seed(int(cfg.get("seed", 42)))
    device = select_device(cfg.get("device", "auto"))
    metrics = discover_modalities(cfg, ROOT, device)
    runtime_seconds = time.time() - start
    _write_metadata(paths, cfg, metrics, runtime_seconds)

    print("Modality discovery finished.")
    print(f"clients_dir={paths['clients_dir']}")
    print(f"discovery_dir={paths['discovery_dir']}")
    print(f"estimated_Q={metrics['estimated_Q']}")
    print(f"abs_Q_error={metrics['abs_Q_error']}")
    print(f"discovery_status={metrics['discovery_status']}")
    print(f"ACC={metrics['ACC']:.6f}, NMI={metrics['NMI']:.6f}, ARI={metrics['ARI']:.6f}")


if __name__ == "__main__":
    main()
