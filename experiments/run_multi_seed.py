import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from utils.config import load_config
from utils.seed import set_seed
from utils.device import select_device
from data.synthetic_dataset import make_synthetic_paired_dataset, split_train_test, build_client_pool
from data.real_dataset_adapter import load_real_paired_dataset
from data.uci_har_adapter import load_uci_har_dataset
from trainers.stage2_trainer import Stage2Trainer


def _prepare_dataset(cfg):
    dataset_cfg = cfg.get("dataset", {})
    ds_type = dataset_cfg.get("type", "synthetic").lower()

    if ds_type == "synthetic":
        full = make_synthetic_paired_dataset(
            num_samples=cfg["train_samples"] + cfg["test_samples"],
            num_modalities=cfg["num_modalities"],
            num_classes=cfg["num_classes"],
            input_dim=cfg["input_dim"],
            seed=cfg["seed"],
        )
        return split_train_test(full, train_ratio=cfg["train_split_ratio"])

    if ds_type == "real":
        return load_real_paired_dataset(cfg)

    if ds_type == "uci_har":
        split = load_uci_har_dataset(cfg, ROOT)
        return {"train": split["train"], "test": split["test"]}

    raise ValueError(f"Unsupported dataset.type: {ds_type}")


def _parse_seeds(seed_arg, cfg):
    if seed_arg:
        return [int(x.strip()) for x in seed_arg.split(",") if x.strip()]
    if "train_seeds" in cfg and cfg["train_seeds"]:
        return [int(x) for x in cfg["train_seeds"]]
    return [int(cfg["seed"])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/uci_har.yaml")
    parser.add_argument("--seeds", type=str, default="", help="comma-separated seeds, e.g. 42,123,2024")
    parser.add_argument("--out", type=str, default="experiments/results/multi_seed_summary.json")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    seeds = _parse_seeds(args.seeds, base_cfg)
    out_path = (ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for seed in seeds:
        cfg = copy.deepcopy(base_cfg)
        cfg["seed"] = int(seed)
        set_seed(cfg["seed"])
        device = select_device(cfg.get("device", "auto"))

        split = _prepare_dataset(cfg)
        train_clients_raw = build_client_pool(split["train"], cfg)
        trainer = Stage2Trainer(cfg, train_clients_raw, split["test"], device)
        metrics = trainer.run()

        row = {"seed": int(seed), **metrics}
        all_rows.append(row)

    top1 = np.array([r["top1_acc"] for r in all_rows], dtype=np.float64)
    f1 = np.array([r["macro_f1"] for r in all_rows], dtype=np.float64)
    summary = {
        "config": str(Path(args.config)),
        "num_runs": len(all_rows),
        "runs": all_rows,
        "aggregate": {
            "top1_acc_mean": float(top1.mean()),
            "top1_acc_std": float(top1.std(ddof=0)),
            "macro_f1_mean": float(f1.mean()),
            "macro_f1_std": float(f1.std(ddof=0)),
        },
    }

    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Multi-seed summary ===")
    print(f"runs: {len(all_rows)}")
    print(f"top1_acc: {summary['aggregate']['top1_acc_mean']:.4f} +/- {summary['aggregate']['top1_acc_std']:.4f}")
    print(f"macro_f1: {summary['aggregate']['macro_f1_mean']:.4f} +/- {summary['aggregate']['macro_f1_std']:.4f}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
