import argparse
import json
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic_split_multimodal.learning.pretrain import run_stage2_discovery
from semantic_split_multimodal.utils.config import load_config
from semantic_split_multimodal.utils.device import select_device
from semantic_split_multimodal.utils.results import dataset_result_name
from semantic_split_multimodal.utils.seed import set_seed


CODEX_RESULTS_ROOT = (ROOT / "local" / "results" / "codex").resolve()


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


def _timestamp_ms():
    now = datetime.now()
    return now.strftime("%y_%m_%d_%H_%M_%S_") + f"{now.microsecond // 1000:03d}"


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


def build_stage2_only_run(
    cfg: dict,
    stage1_dir,
    output_root,
    tag=None,
    run_type="codex_test",
    cluster_dir=None,
    result_dir=None,
    allow_existing=False,
):
    run_type = str(run_type)
    if run_type not in {"codex_test", "user_formal"}:
        raise ValueError("run_type must be 'codex_test' or 'user_formal'.")

    partition_dir = _resolve(stage1_dir)
    if not partition_dir.exists():
        raise FileNotFoundError(f"Missing Stage1 directory: {partition_dir}")
    if not (partition_dir / "train_clients").exists():
        raise FileNotFoundError(f"Missing train_clients under Stage1 directory: {partition_dir}")

    output_root_path = _resolve(output_root)
    if run_type == "codex_test" and not _is_relative_to(output_root_path, CODEX_RESULTS_ROOT):
        raise ValueError(f"codex_test output_root must be under {CODEX_RESULTS_ROOT}, got {output_root_path}")
    if run_type == "user_formal" and output_root is None:
        raise ValueError("user_formal requires an explicit output_root.")

    dataset_name = dataset_result_name(cfg)
    run_tag = str(tag or _timestamp_ms())
    if cluster_dir is None and result_dir is None:
        run_dir = output_root_path / dataset_name / run_tag
        cluster_path = run_dir / "02_cluster_results"
        result_path = run_dir / "02_discovery_logs"
    elif cluster_dir is not None and result_dir is not None:
        cluster_path = _resolve(cluster_dir)
        result_path = _resolve(result_dir)
        run_dir = result_path.parent
    else:
        raise ValueError("--cluster-dir and --result-dir must be provided together when using compatibility mode.")

    for output_dir in [cluster_path, result_path]:
        if output_dir.exists() and not allow_existing:
            raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")

    run_cfg = dict(cfg)
    run_cfg["partition"] = {**run_cfg.get("partition", {}), "output_dir": str(partition_dir)}
    run_cfg["cluster"] = {**run_cfg.get("cluster", {}), "output_dir": str(cluster_path)}
    run_cfg["result"] = {**run_cfg.get("result", {}), "output_dir": str(result_path)}
    run_cfg["stage2_only"] = {
        "run_type": run_type,
        "stage1_dir": str(partition_dir),
        "output_root": str(output_root_path),
        "run_dir": str(run_dir),
        "cluster_dir": str(cluster_path),
        "result_dir": str(result_path),
        "tag": run_tag,
    }
    return run_cfg, {
        "partition_dir": partition_dir,
        "output_root": output_root_path,
        "run_dir": run_dir,
        "cluster_dir": cluster_path,
        "result_dir": result_path,
        "run_type": run_type,
        "tag": run_tag,
    }


def _write_metadata(paths: dict, cfg: dict, metrics: dict | None, runtime_seconds: float):
    paths["cluster_dir"].mkdir(parents=True, exist_ok=True)
    paths["result_dir"].mkdir(parents=True, exist_ok=True)
    metadata = {
        "stage": "stage2_discovery_only",
        "run_type": paths["run_type"],
        "tag": paths["tag"],
        "stage1_dir": str(paths["partition_dir"]),
        "output_root": str(paths["output_root"]),
        "run_dir": str(paths["run_dir"]),
        "cluster_dir": str(paths["cluster_dir"]),
        "result_dir": str(paths["result_dir"]),
        "git_commit": _git_commit(),
        "runtime_seconds": float(runtime_seconds),
        "seed": int(cfg.get("seed", 42)),
        "adaptive_parameters": dict(cfg.get("cluster", {}).get("adaptive", {})),
        "metrics": metrics,
    }
    for path in [paths["cluster_dir"] / "stage2_only_metadata.json", paths["result_dir"] / "stage2_only_metadata.json"]:
        with path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    with (paths["result_dir"] / "stage2_only_config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with (paths["cluster_dir"] / "stage2_only_config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Stage 2 only: run modality discovery from a read-only Stage1 partition."
    )
    parser.add_argument("--config", required=True, help="Path to yaml config")
    parser.add_argument("--stage1-dir", help="Existing Stage 1 partition directory, read-only input")
    parser.add_argument("--output-root", help="Root directory for isolated Stage2-only outputs")
    parser.add_argument("--tag", help="Run tag under output-root/<dataset>/")
    parser.add_argument("--run-type", choices=["codex_test", "user_formal"], default="codex_test")
    parser.add_argument("--partition-dir", help="Compatibility alias for --stage1-dir")
    parser.add_argument("--cluster-dir", help="Compatibility output directory for 02_cluster_results")
    parser.add_argument("--result-dir", help="Compatibility output directory for Stage2 logs")
    parser.add_argument("--allow-existing", action="store_true", help="Allow writing into existing output dirs")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    stage1_dir = args.stage1_dir or args.partition_dir
    if not stage1_dir:
        raise ValueError("--stage1-dir is required.")
    if not args.output_root:
        if args.run_type == "codex_test":
            output_root = CODEX_RESULTS_ROOT / "stage2_smoke"
        else:
            raise ValueError("--output-root is required when run_type=user_formal.")
    else:
        output_root = args.output_root

    cfg = load_config(args.config)
    cfg, paths = build_stage2_only_run(
        cfg,
        stage1_dir=stage1_dir,
        output_root=output_root,
        tag=args.tag,
        run_type=args.run_type,
        cluster_dir=args.cluster_dir,
        result_dir=args.result_dir,
        allow_existing=args.allow_existing,
    )
    paths["cluster_dir"].mkdir(parents=True, exist_ok=True)
    paths["result_dir"].mkdir(parents=True, exist_ok=True)

    start = time.time()
    set_seed(int(cfg.get("seed", 42)))
    device = select_device(cfg.get("device", "auto"))
    metrics = run_stage2_discovery(cfg, ROOT, device)
    runtime_seconds = time.time() - start
    _write_metadata(paths, cfg, metrics, runtime_seconds)

    print("Stage 2 only finished.")
    print(f"run_type={paths['run_type']}")
    print(f"stage1_dir={paths['partition_dir']}")
    print(f"output_root={paths['output_root']}")
    print(f"cluster_dir={paths['cluster_dir']}")
    print(f"result_dir={paths['result_dir']}")
    print(f"estimated_Q={metrics['estimated_Q']}")
    print(f"abs_Q_error={metrics['abs_Q_error']}")
    print(f"discovery_status={metrics['discovery_status']}")
    print(f"ACC={metrics['ACC']:.6f}, NMI={metrics['NMI']:.6f}, ARI={metrics['ARI']:.6f}")


if __name__ == "__main__":
    main()
