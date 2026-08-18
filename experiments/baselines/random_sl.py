# RandomSL 单次训练 wrapper，固定使用随机客户端选择策略。
import sys

from experiments.training import main as train_main

DEFAULT_METHOD = "randomsl"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--method" in argv:
        raise SystemExit("random_sl.py fixes method=randomsl; use experiments/training.py for explicit methods.")
    argv = ["--method", DEFAULT_METHOD, *argv]
    return train_main(argv)


if __name__ == "__main__":
    main()
