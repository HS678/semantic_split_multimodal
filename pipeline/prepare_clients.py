# Pipeline 第二阶段：从数据集 loader 输出生成单模态 client partition。
import argparse
import json
import sys
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

from MSL.data import prepare_clients
from MSL.protocol import DATASET_PROTOCOLS
from MSL.utils import set_seed
from experiments.common import build_experiment_config, apply_experiment_overrides, with_repeated_seed_split_signature


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Prepare single-modality clients.")
    parser.add_argument("--dataset", choices=tuple(DATASET_PROTOCOLS), required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="results/pipeline/clients")
    parser.add_argument("--clients", type=int, default=10)
    return parser.parse_args(argv)


def _write_config_snapshot(cfg: dict, output_dir: str | Path) -> None:
    snapshot = {
        "config_scope": "client_preparation",
        "seed": cfg.get("seed"),
        "num_classes": cfg.get("num_classes"),
        "dataset": cfg.get("dataset"),
        "partition": cfg.get("partition"),
        "runtime_overrides": cfg.get("runtime_overrides"),
    }
    with (Path(output_dir) / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2, ensure_ascii=False, sort_keys=True)


def main(argv=None):
    args = parse_args(argv)
    cfg = build_experiment_config(dataset_type=args.dataset, seed=args.seed, clients=args.clients)
    cfg = apply_experiment_overrides(cfg, fold=args.fold)
    if args.fold is None and DATASET_PROTOCOLS[str(args.dataset)]["fold_count"] is None:
        cfg = with_repeated_seed_split_signature(cfg, args.seed)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    cfg["partition"] = {
        **dict(cfg.get("partition", {})),
        "output_dir": str((output_root / args.dataset).resolve()),
        "auto_signature_dir": True,
    }
    set_seed(int(cfg.get("seed", 42)))
    info = prepare_clients(cfg, ROOT)
    _write_config_snapshot(cfg, info["output_dir"])
    print(f"Client preparation finished. Saved data partition to: {info['output_dir']}")
    print(f"num_clients={info['num_clients']}, clients_per_modality={info['clients_per_modality']}")


if __name__ == "__main__":
    main()
