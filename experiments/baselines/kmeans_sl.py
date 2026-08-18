# KMeans-SL 单次训练 wrapper，将 --k 映射到共享 training runner。
import argparse
import sys

from experiments.training import main as train_main

SUPPORTED_K = (2, 3, 4, 5)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--k", type=int, choices=SUPPORTED_K, default=2)
    known, rest = parser.parse_known_args(argv)
    if "--method" in rest:
        raise SystemExit("kmeans_sl.py fixes method from --k; use experiments/training.py for explicit methods.")
    rest = ["--method", f"kmeans{known.k}", *rest]
    return train_main(rest)


if __name__ == "__main__":
    main()
