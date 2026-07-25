import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic_split_multimodal.learning.fusion_sl import run_mmbind_fusion_stage3_split_training
from semantic_split_multimodal.learning.baseline_unpaired import run_unpaired_stage3_split_training
from semantic_split_multimodal.utils.config import load_config
from semantic_split_multimodal.utils.device import select_device
from semantic_split_multimodal.utils.results import configure_result_run
from semantic_split_multimodal.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Stage 3: split multimodal learning")
    parser.add_argument("--config", required=True, help="Path to yaml config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = configure_result_run(cfg, ROOT, stage="stage3_train", create_new=False)
    set_seed(int(cfg.get("seed", 42)))
    device = select_device(cfg.get("device", "auto"))
    mode = str(cfg.get("training", {}).get("multimodal_mode", "unpaired_split_learning")).lower()
    if mode in {"mmbind_fusion_split_learning", "mmbind_fusion_sl"}:
        metrics = run_mmbind_fusion_stage3_split_training(cfg, ROOT, device)
    elif mode in {"unpaired_split_learning", "unpaired_shared_semantic"}:
        metrics = run_unpaired_stage3_split_training(cfg, ROOT, device)
    else:
        raise ValueError("training.multimodal_mode must be 'mmbind_fusion_split_learning' or 'unpaired_split_learning'.")
    print("Stage 3 finished.")
    print(f"run_dir={cfg['results']['run_dir']}")
    print(f"final_metrics={metrics}")


if __name__ == "__main__":
    main()
