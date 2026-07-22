import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from trainers.split_multimodal_trainer import run_stage3_split_training
from trainers.unpaired_split_multimodal_trainer import run_unpaired_stage3_split_training
from utils.config import load_config
from utils.device import select_device
from utils.results import configure_result_run
from utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Stage 3: split multimodal learning")
    parser.add_argument("--config", required=True, help="Path to yaml config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = configure_result_run(cfg, ROOT, stage="stage3_train_sl", create_new=False)
    set_seed(int(cfg.get("seed", 42)))
    device = select_device(cfg.get("device", "auto"))
    mode = str(cfg.get("training", {}).get("multimodal_mode", "pseudo_paired_concat")).lower()
    if mode == "pseudo_paired_concat":
        metrics = run_stage3_split_training(cfg, ROOT, device)
    elif mode == "unpaired_shared_semantic":
        metrics = run_unpaired_stage3_split_training(cfg, ROOT, device)
    else:
        raise ValueError(
            "training.multimodal_mode must be 'pseudo_paired_concat' "
            "or 'unpaired_shared_semantic'."
        )
    print("Stage 3 finished.")
    print(f"run_dir={cfg['results']['run_dir']}")
    print(f"final_metrics={metrics}")


if __name__ == "__main__":
    main()
