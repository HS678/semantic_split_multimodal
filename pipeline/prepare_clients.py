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

from MSL.data import prepare_clients
from MSL.protocol import (
    add_experiment_args,
    load_experiment_config_from_args,
    print_resolved_config,
    save_resolved_config_artifact,
    client_preparation_config_snapshot,
)
from MSL.utils import configure_result_run
from MSL.utils import set_seed


def main():
    parser = argparse.ArgumentParser(description="Prepare single-modality clients")
    add_experiment_args(parser, include_seed=True)
    args = parser.parse_args()

    cfg = load_experiment_config_from_args(args)
    if args.print_config:
        print_resolved_config(cfg)
        return
    cfg = configure_result_run(cfg, ROOT)
    set_seed(int(cfg.get("seed", 42)))
    info = prepare_clients(cfg, ROOT)
    save_resolved_config_artifact(client_preparation_config_snapshot(cfg), info["output_dir"])
    print(f"Client preparation finished. Saved data partition to: {info['output_dir']}")
    print(f"run_dir={cfg['results']['run_dir']}")
    print(f"num_clients={info['num_clients']}, clients_per_modality={info['clients_per_modality']}")


if __name__ == "__main__":
    main()
