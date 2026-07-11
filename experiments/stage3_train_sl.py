import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from trainers.split_multimodal_trainer import run_stage3_split_training
from utils.config import load_config
from utils.device import select_device
from utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Stage 3: split multimodal learning")
    parser.add_argument("--config", required=True, help="Path to yaml config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = select_device(cfg.get("device", "auto"))
    metrics = run_stage3_split_training(cfg, ROOT, device)
    print("Stage 3 finished.")
    print(f"final_metrics={metrics}")


if __name__ == "__main__":
    main()
