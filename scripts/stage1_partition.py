import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from MSL.data.partitioner import run_stage1_partition
from MSL.utils.config import load_config, save_config_artifacts
from MSL.utils.results import configure_result_run
from MSL.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Stage 1: data partition")
    parser.add_argument("--config", required=True, help="Path to INI-style .config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = configure_result_run(cfg, ROOT, stage="stage1_partition", create_new=True)
    set_seed(int(cfg.get("seed", 42)))
    info = run_stage1_partition(cfg, ROOT)
    save_config_artifacts(args.config, cfg, info["output_dir"])
    print(f"Stage 1 finished. Saved data partition to: {info['output_dir']}")
    print(f"run_dir={cfg['results']['run_dir']}")
    print(f"num_clients={info['num_clients']}, clients_per_modality={info['clients_per_modality']}")


if __name__ == "__main__":
    main()
