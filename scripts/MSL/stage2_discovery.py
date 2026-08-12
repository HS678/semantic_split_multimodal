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

from MSL.learning.pretrain import run_stage2_discovery
from MSL.utils.config import save_config_artifacts
from MSL.utils.device import select_device
from MSL.utils.experiment_args import add_experiment_args, load_experiment_config_from_args, print_resolved_config
from MSL.utils.results import dataset_result_name, resolve_stage_paths, safe_result_component
from MSL.utils.seed import set_seed


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


def build_stage2_run(cfg: dict, stage1_dir, output_root):
    partition_dir = _resolve(stage1_dir)
    if not partition_dir.exists():
        raise FileNotFoundError(f"Missing Stage1 directory: {partition_dir}")
    if not (partition_dir / "train_clients").exists():
        raise FileNotFoundError(f"Missing train_clients under Stage1 directory: {partition_dir}")

    output_root_path = _resolve(output_root)
    dataset_name = safe_result_component(dataset_result_name(cfg))
    partition_name = safe_result_component(partition_dir.name)
    method_name = _cluster_method_name(cfg)
    cluster_dir = (output_root_path / dataset_name / partition_name / method_name).resolve()
    if cluster_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing Stage2 output directory: {cluster_dir}")

    run_cfg = dict(cfg)
    run_cfg["partition"] = {**run_cfg.get("partition", {}), "output_dir": str(partition_dir)}
    run_cfg["cluster"] = {**run_cfg.get("cluster", {}), "output_dir": str(cluster_dir), "method": method_name}
    run_cfg["result"] = {**run_cfg.get("result", {}), "output_dir": str(cluster_dir)}
    run_cfg["stage2"] = {
        "stage1_dir": str(partition_dir),
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
    metadata = {
        "stage": "stage2_discovery",
        "dataset": paths["dataset"],
        "partition_signature": paths["partition_signature"],
        "cluster_method": paths["cluster_method"],
        "stage1_dir": str(paths["partition_dir"]),
        "output_root": str(paths["output_root"]),
        "cluster_dir": str(paths["cluster_dir"]),
        "git_commit": _git_commit(),
        "runtime_seconds": float(runtime_seconds),
        "seed": int(cfg.get("seed", 42)),
        "config_snapshot": cfg,
        "metrics": metrics,
    }
    with (paths["cluster_dir"] / "stage2_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Stage 2: discover modality clusters from a frozen Stage1 partition.")
    add_experiment_args(parser, include_seed=True)
    parser.add_argument("--stage1-dir", help="Optional override for stage2.stage1_dir")
    parser.add_argument("--output-root", help="Optional override for stage2.output_root")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg, source_path = load_experiment_config_from_args(args)
    if args.print_config:
        print_resolved_config(cfg)
        return
    stage2_cfg = cfg.get("stage2", {})
    stage1_dir = args.stage1_dir or stage2_cfg.get("stage1_dir")
    output_root = args.output_root or stage2_cfg.get("output_root")
    if not stage1_dir or not output_root:
        # 新格式 config 不写路径：自动从 base_dir + 数据集 + 协议生成。
        resolved = resolve_stage_paths(cfg, ROOT)
        stage1_dir = stage1_dir or resolved["stage1_dir"]
        output_root = output_root or resolved["stage2_dir"].parents[2]
    cfg, paths = build_stage2_run(
        cfg,
        stage1_dir=stage1_dir,
        output_root=output_root,
    )
    paths["cluster_dir"].mkdir(parents=True, exist_ok=True)
    save_config_artifacts(source_path, cfg, paths["cluster_dir"])

    start = time.time()
    set_seed(int(cfg.get("seed", 42)))
    device = select_device(cfg.get("device", "auto"))
    metrics = run_stage2_discovery(cfg, ROOT, device)
    runtime_seconds = time.time() - start
    _write_metadata(paths, cfg, metrics, runtime_seconds)

    print("Stage 2 finished.")
    print(f"stage1_dir={paths['partition_dir']}")
    print(f"cluster_dir={paths['cluster_dir']}")
    print(f"estimated_Q={metrics['estimated_Q']}")
    print(f"abs_Q_error={metrics['abs_Q_error']}")
    print(f"discovery_status={metrics['discovery_status']}")
    print(f"ACC={metrics['ACC']:.6f}, NMI={metrics['NMI']:.6f}, ARI={metrics['ARI']:.6f}")


if __name__ == "__main__":
    main()
