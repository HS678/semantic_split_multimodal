import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic_split_multimodal.learning.pretrain import run_stage2_discovery
from semantic_split_multimodal.utils.config import load_config
from semantic_split_multimodal.utils.device import select_device
from semantic_split_multimodal.utils.results import configure_result_run
from semantic_split_multimodal.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Stage 2: pretrain encoders and cluster clients")
    parser.add_argument("--config", required=True, help="Path to yaml config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = configure_result_run(cfg, ROOT, stage="stage2_discovery", create_new=False)
    set_seed(int(cfg.get("seed", 42)))
    device = select_device(cfg.get("device", "auto"))
    metrics = run_stage2_discovery(cfg, ROOT, device)
    print("Stage 2 finished.")
    print(f"run_dir={cfg['results']['run_dir']}")
    print(f"ACC={metrics['ACC']:.6f}, NMI={metrics['NMI']:.6f}, ARI={metrics['ARI']:.6f}")


if __name__ == "__main__":
    main()
