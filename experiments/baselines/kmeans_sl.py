import argparse
import sys

from experiments.msl.train import main as train_main, run_one

SUPPORTED_K = (2, 3, 4, 5)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--k", type=int, choices=SUPPORTED_K, default=2)
    known, rest = parser.parse_known_args(argv)
    if "--method" not in rest:
        rest = ["--method", f"kmeans{known.k}", *rest]
    return train_main(rest)


if __name__ == "__main__":
    main()
