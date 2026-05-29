import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from utils.config import load_config
from utils.seed import set_seed
from utils.device import select_device
from data.synthetic_dataset import make_synthetic_paired_dataset, split_train_test, build_client_pool
from data.real_dataset_adapter import load_real_paired_dataset
from data.uci_har_adapter import load_uci_har_dataset
from trainers.stage2_trainer import Stage2Trainer


def _str_to_bool(x: str) -> bool:
    v = str(x).strip().lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value from: {x}")


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

    if ds_type == "uci_har":
        split = load_uci_har_dataset(cfg, ROOT)
        print(f"dataset source: uci_har ({split['root']})")
        return {"train": split["train"], "test": split["test"]}

    raise ValueError(f"Unsupported dataset.type: {ds_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--lambda_align", type=float, default=None)
    parser.add_argument("--use_oracle_clusters_for_training", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # backward compatibility
    if "lambda_align" not in cfg:
        cfg["lambda_align"] = float(cfg.get("lambda_supcon", 0.0))
    if "clustering" not in cfg:
        cfg["clustering"] = {}
    if "use_oracle_clusters_for_training" not in cfg["clustering"]:
        cfg["clustering"]["use_oracle_clusters_for_training"] = True

    if args.lambda_align is not None:
        cfg["lambda_align"] = float(args.lambda_align)
    if args.use_oracle_clusters_for_training is not None:
        cfg["clustering"]["use_oracle_clusters_for_training"] = _str_to_bool(args.use_oracle_clusters_for_training)

    set_seed(cfg["seed"])
    device = select_device(cfg.get("device", "auto"))
    print(f"compute device: {device}")

    split = _prepare_dataset(cfg)
    train_clients_raw = build_client_pool(split["train"], cfg)

    trainer = Stage2Trainer(cfg, train_clients_raw, split["test"], device)
    trainer.run()


if __name__ == "__main__":
    main()
