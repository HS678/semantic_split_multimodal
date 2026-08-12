import argparse
from pathlib import Path
import sys

def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "MSL").is_dir():
            return parent
    raise RuntimeError("Cannot locate project root containing src/MSL.")


ROOT = _project_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from MSL.data.partitioner import run_stage1_partition
from MSL.utils.config import save_config_artifacts
from MSL.utils.experiment_args import add_experiment_args, load_experiment_config_from_args, print_resolved_config
from MSL.utils.results import configure_result_run
from MSL.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Stage 1: data partition")
    add_experiment_args(parser, include_seed=True)
    args = parser.parse_args()

    cfg, source_path = load_experiment_config_from_args(args)
    if args.print_config:
        print_resolved_config(cfg)
        return
    cfg = configure_result_run(cfg, ROOT)
    set_seed(int(cfg.get("seed", 42)))
    info = run_stage1_partition(cfg, ROOT)
    save_config_artifacts(source_path, cfg, info["output_dir"])
    print(f"Stage 1 finished. Saved data partition to: {info['output_dir']}")
    print(f"run_dir={cfg['results']['run_dir']}")
    print(f"num_clients={info['num_clients']}, clients_per_modality={info['clients_per_modality']}")


if __name__ == "__main__":
    main()
