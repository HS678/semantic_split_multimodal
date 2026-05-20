import argparse
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from utils.config import load_config
from utils.seed import set_seed
from data.synthetic_dataset import make_synthetic_paired_dataset, split_train_test, build_client_pool
from data.real_dataset_adapter import load_real_paired_dataset
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
        split = split_train_test(full, train_ratio=cfg["train_split_ratio"])
        print("dataset source: synthetic")
        return split

    if ds_type == "real":
        split = load_real_paired_dataset(cfg)
        print(f"dataset source: real ({cfg['dataset'].get('root', '')})")
        return split

    raise ValueError(f"Unsupported dataset.type: {ds_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = torch.device(cfg.get("device", "cpu"))

    split = _prepare_dataset(cfg)
    train_clients_raw = build_client_pool(split["train"], cfg)

    trainer = Stage2Trainer(cfg, train_clients_raw, split["test"], device)
    trainer.run()


if __name__ == "__main__":
    main()
