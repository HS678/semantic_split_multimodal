# Oracle-SL 单次训练 wrapper，固定使用真实 cluster assignment。
import sys

from experiments.training import main as train_main

DEFAULT_METHOD = "oracle"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--method" in argv:
        raise SystemExit("oracle_sl.py fixes method=oracle; use experiments/training.py for explicit methods.")
    argv = ["--method", DEFAULT_METHOD, *argv]
    return train_main(argv)


if __name__ == "__main__":
    main()
