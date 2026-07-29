import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic_split_multimodal.evaluation.training_curves import write_training_curves


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot Stage3 train and validation curves from a completed run.")
    parser.add_argument("--run-dir", required=True, help="Stage3 run directory containing train_log.csv and validation_log.csv")
    args = parser.parse_args(argv)
    output_path = write_training_curves(Path(args.run_dir))
    print(f"training_curves={output_path}")


if __name__ == "__main__":
    main()
